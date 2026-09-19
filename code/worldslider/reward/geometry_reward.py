"""Geometry reward from trajectory recovery (Section 3.4 of the paper).

A generated video is scored by recovering its camera trajectory with a
feed-forward geometry estimator (Depth Anything 3) and comparing segment
translations, expressed in the local camera frame, with the prescribed flight:

    d_{t,h}     = R_t^T (p_{t+h} - p_t)          prescribed
    d^_{t,h}    = R^_t^T (p^_{t+h} - p^_t)       recovered

over short and long offsets h in H.  A fixed rotation gauge Q maps estimator
axes into the prescribed frame; it is fitted once on the *source* video of the
flight and then frozen, so an incorrect generated trajectory cannot rotate its
errors away.  Per generated video only a positive global scale s is solved:

    e_seg = sum ||s Q d^ - d|| / sum ||d||

    r_geo = exp(-[e_seg - b]_+ / kappa) * h_dir * h_rot * h_mot

where b is the error of the same estimator on the source video (a
flight-specific floor), and the gates h_dir (translation direction),
h_rot (relative rotation) and h_mot (visual motion) are bounded in [0, 1].
If the source video itself cannot be recovered reliably the verifier abstains
and returns None; the caller then trains that sample with weight 1.

Prescribed poses for clip `<name>` are read from `<dataset>/poses/<name>.txt`
(rows `tx ty tz qx qy qz qw`, body-to-world in the x-forward / y-right / z-down
convention of worldslider/flightgrid/render.py) and the source video from
`<dataset>/videos/<name>.mp4`. Floors are cached per (estimator, clip), so a clip
here is one 93-frame window, not a whole recording.
"""
from __future__ import annotations

import json
import os
import tempfile

import cv2
import numpy as np

DATASET_DIR = os.environ.get("WORLDSLIDER_DATASET", "datasets/worldslider")
CACHE_PATH = os.environ.get("WORLDSLIDER_REWARD_CACHE", os.path.join(DATASET_DIR, "reward_floors.json"))
DA3_MODEL = os.environ.get("DA3_MODEL", "depth-anything/DA3-LARGE-1.1")

N_FRAMES = 24              # frames handed to the estimator (uniformly subsampled)
OFFSETS = (2, 8)           # segment offsets H: local direction and curvature horizon
KAPPA = 0.15               # reward temperature on the excess segment error
ABSTAIN_ERR = 0.60         # floor above which the source flight is deemed unrecoverable
DIR_LO = 0.10              # h_dir ramps from DIR_LO to 0.8 * source direction agreement
ROT_LO, ROT_HI = 0.25, 0.90   # h_rot: excess relative-rotation error (rad) where the gate closes
MOTION_FRAC = 0.60         # h_mot: generated optical flow must reach this fraction of the source

_da3 = None


# ----------------------------------------------------------------------------- estimators
def read_frames(path, max_frames=93):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)
    return [frames[i] for i in idx]


def subsample(frames, n=N_FRAMES):
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def flow_magnitude(frames):
    """Mean Farneback optical-flow magnitude at 256 px: the visual-motion measure."""
    g = [cv2.cvtColor(cv2.resize(f, (256, 256)), cv2.COLOR_BGR2GRAY) for f in frames]
    mags = [float(np.linalg.norm(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0), axis=-1).mean())
            for a, b in zip(g[:-1], g[1:])]
    return float(np.mean(mags))


def estimate_poses(frames):
    """Recovered camera centres (N,3) and camera-to-world rotations (N,3,3)."""
    global _da3
    import torch
    from depth_anything_3.api import DepthAnything3

    if _da3 is None:
        _da3 = DepthAnything3.from_pretrained(DA3_MODEL).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tmp = tempfile.mkdtemp(prefix="da3_")
    paths = []
    for i, f in enumerate(frames):
        p = os.path.join(tmp, f"{i:05d}.png")
        cv2.imwrite(p, f)
        paths.append(p)
    pred = _da3.inference(paths)
    for p in paths:
        os.remove(p)
    os.rmdir(tmp)
    E = np.asarray(pred.extrinsics, dtype=np.float64)  # world-to-camera (N,3,4) or (N,4,4)
    C = -np.einsum("nij,nj->ni", E[:, :3, :3].transpose(0, 2, 1), E[:, :3, 3])
    R_c2w = E[:, :3, :3].transpose(0, 2, 1)
    return C, R_c2w


# ----------------------------------------------------------------------------- geometry
def gt_poses(clip, n):
    from scipy.spatial.transform import Rotation

    P = np.loadtxt(os.path.join(DATASET_DIR, "poses", f"{clip}.txt"))
    idx = np.linspace(0, len(P) - 1, n).astype(int)
    return P[idx, :3], Rotation.from_quat(P[idx, 3:7]).as_matrix()


def segments(C, R_c2w, h):
    """Segment translation and relative rotation in the local frame at segment start."""
    t = np.einsum("nji,nj->ni", R_c2w[:-h], C[h:] - C[:-h])
    r = np.einsum("nji,njk->nik", R_c2w[:-h], R_c2w[h:])
    return t, r


def segment_pack(C, R_c2w):
    return {h: segments(C, R_c2w, h) for h in OFFSETS if len(C) > h}


def _cat(pack):
    hs = sorted(pack)
    return np.concatenate([pack[h][0] for h in hs], 0), hs


def kabsch(P, G):
    """Best rotation P -> G (no scale, no translation)."""
    U, _, Vt = np.linalg.svd(P.T @ G)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1
    return Vt.T @ D @ U.T


def segment_metrics(gen, gt, gauge):
    """Scale-fitted segment error, length-weighted direction cosine and mean
    relative-rotation error between two segment packs under a frozen gauge."""
    t_all, hs = _cat(gen)
    tau_all, _ = _cat(gt)
    n = min(len(t_all), len(tau_all))
    t_all, tau_all = t_all[:n], tau_all[:n]
    tm = (gauge @ t_all.T).T
    s = max(float((tm * tau_all).sum() / max((tm ** 2).sum(), 1e-12)), 1e-6)
    err = float(np.linalg.norm(s * tm - tau_all, axis=1).sum() / max(np.linalg.norm(tau_all, axis=1).sum(), 1e-9))
    w = np.linalg.norm(tau_all, axis=1)
    nm = np.linalg.norm(tm, axis=1)
    nz = (w > 1e-9) & (nm > 1e-12)
    dcos = float(((tm[nz] * tau_all[nz]).sum(1) / (nm[nz] * w[nz]) * w[nz]).sum() / w[nz].sum()) if nz.any() else 0.0
    angs = []
    for h in hs:
        if h not in gt:
            continue
        r, rho = gen[h][1], gt[h][1]
        m = min(len(r), len(rho))
        # the gauge maps recovered local quantities into the prescribed frame, so the
        # recovered relative rotation is carried over as Q R Q^T before it is compared
        # with the prescribed one; conjugating the prescribed rotation instead would
        # only agree when Q commutes with it
        Rg = np.einsum("ij,njk,lk->nil", gauge, r[:m], gauge)
        M = np.einsum("nij,njk->nik", Rg.transpose(0, 2, 1), rho[:m])
        angs.append(np.arccos(np.clip((np.trace(M, axis1=1, axis2=2) - 1.0) / 2.0, -1, 1)))
    rre = float(np.concatenate(angs).mean()) if angs else None
    return dict(err=err, dir_cos=dcos, rre=rre, scale=s)


# ----------------------------------------------------------------------------- floors
_floors = None


def _load_floors():
    global _floors
    if _floors is None:
        _floors = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    return _floors


def source_floor(clip):
    """Run the estimator on the source video of `clip`: gives the gauge Q and
    the error floor b (plus direction / rotation / motion references)."""
    floors = _load_floors()
    key = f"{DA3_MODEL.split('/')[-1]}:{clip}"
    if key in floors:
        return floors[key]
    frames = subsample(read_frames(os.path.join(DATASET_DIR, "videos", f"{clip}.mp4")))
    C, R = estimate_poses(frames)
    real = segment_pack(C, R)
    gt = segment_pack(*gt_poses(clip, len(C)))
    t_all, _ = _cat(real)
    tau_all, _ = _cat(gt)
    n = min(len(t_all), len(tau_all))
    gauge = kabsch(t_all[:n], tau_all[:n])
    m = segment_metrics(real, gt, gauge)
    ent = dict(err=m["err"], dir_cos=m["dir_cos"], rre=m["rre"], gauge=gauge.tolist(), flow=flow_magnitude(frames))
    floors[key] = ent
    try:
        on_disk = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
        on_disk.update(floors)                      # keep entries other ranks wrote meanwhile
        floors.update(on_disk)
        tmp = f"{CACHE_PATH}.tmp{os.getpid()}"      # pid-unique, then atomic replace
        json.dump(on_disk, open(tmp, "w"))
        os.replace(tmp, CACHE_PATH)
    except (OSError, ValueError):
        pass
    return ent


def _ramp(x, lo, hi):
    return float(np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


# ----------------------------------------------------------------------------- reward
def geometry_reward(frames, clip):
    """frames: list of HxWx3 uint8 BGR frames of a generated video for `clip`.
    Returns a dict with `reward` (float in [0,1], or None to abstain)."""
    base = source_floor(clip)
    if base.get("err") is None or not np.isfinite(base["err"]) or base["err"] >= ABSTAIN_ERR:
        return dict(reward=None, abstain=True, floor=base.get("err"))

    frames = subsample(frames)
    C, R = estimate_poses(frames)
    gen = segment_pack(C, R)
    gt = segment_pack(*gt_poses(clip, len(C)))
    m = segment_metrics(gen, gt, np.asarray(base["gauge"]))

    err = m["err"] if np.isfinite(m["err"]) else 10.0
    dcos = m["dir_cos"] if np.isfinite(m["dir_cos"]) else 0.0
    excess = max(err - base["err"], 0.0)
    r_traj = float(np.exp(-excess / KAPPA))
    # each gate falls back to 1 (inactive) when its reference is missing on the source
    # flight, so a missing reference can never invent a penalty
    ref_dir = base.get("dir_cos")
    ref_dir = 0.3 if ref_dir is None or not np.isfinite(ref_dir) else ref_dir
    h_dir = _ramp(dcos, DIR_LO, min(0.6, max(0.8 * ref_dir, 0.15)))
    h_rot = 1.0
    if m["rre"] is not None and base.get("rre") is not None:
        h_rot = 1.0 - _ramp(max(m["rre"] - base["rre"], 0.0), ROT_LO, ROT_HI)
    h_mot = 1.0
    flow_ref = base.get("flow")
    if flow_ref and np.isfinite(flow_ref) and flow_ref > 1e-6:
        h_mot = _ramp(flow_magnitude(frames) / flow_ref, 0.0, MOTION_FRAC)

    reward = float(r_traj * h_dir * h_rot * h_mot)
    return dict(reward=reward if np.isfinite(reward) else 0.0, r_traj=r_traj, h_dir=h_dir, h_rot=h_rot,
                h_mot=h_mot, err=err, floor=base["err"], dir_cos=dcos, rre=m["rre"], scale=m["scale"])


class RewardWeighter:
    """Rewards -> per-sample loss weights with running statistics.

    w = clip(exp((r - mu) / (beta * max(sigma, sigma_min))), w_min, w_max)

    mu and sigma are exponential moving averages across steps (the global
    batch is too small to standardise within a batch); `weights` must be called
    before `update` so a sample is never compared against statistics it moved.
    """

    def __init__(self, momentum=0.99, beta=1.0, w_min=0.9, w_max=1.15, sigma_min=0.05):
        self.m, self.beta, self.w_min, self.w_max, self.sigma_min = momentum, beta, w_min, w_max, sigma_min
        self.mean = None
        self.var = None

    def update(self, rewards):
        r = np.asarray(rewards, dtype=np.float64).ravel()
        if self.mean is None:
            self.mean, self.var = float(r.mean()), float(max(r.var(), 1e-4))
        else:
            for x in r:
                d = x - self.mean
                self.mean += (1 - self.m) * d
                self.var = self.m * self.var + (1 - self.m) * d * d
        return self

    def weights(self, rewards):
        r = np.asarray(rewards, dtype=np.float64).ravel()
        if self.mean is None:
            return np.ones_like(r)
        std = max(np.sqrt(self.var), self.sigma_min)
        return np.clip(np.exp((r - self.mean) / (self.beta * std)), self.w_min, self.w_max)


if __name__ == "__main__":
    import sys

    clip, video = sys.argv[1], sys.argv[2]
    out = geometry_reward(read_frames(video), clip)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, indent=1))
