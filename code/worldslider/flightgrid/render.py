"""FlightGrid: render a metric camera trajectory as an appearance-free control video.

A FlightGrid frame contains two structures, both fixed in world coordinates
before rendering, so every pixel motion in the control video is induced by the
prescribed camera motion alone:

  * the *flight trace*: a body-frame box corridor swept along the trajectory
    (cross-section 0.6 m x 0.2 m, banking with the recorded attitude), with a
    ring every `ring_dt` seconds so spacing along the box encodes speed.
    Each trace segment carries the time at which the camera passes it and is
    only drawn inside a temporal validity window around the current frame
    (fully visible within +-`t_full` s, gone beyond +-`t_zero` s);
  * the *spatial scaffold*: a 1 m ground and ceiling grid (red below, blue
    above) sized from the trajectory's speed and vertical extent, intermediate
    level lines (green) when the flight spans several storeys, and a wall of
    vertical bars standing in an annulus around the path.

Conventions
-----------
One row per frame: ``tx ty tz qx qy qz qw``, a body-to-world pose. Both the world
and the body frame are NED-like: x forward, y right, z down, so "up" is -z and the
identity quaternion looks along world +x. The renderer converts a pose to the
OpenCV camera frame itself (``NED_TO_CAM``: x right, y down, z forward), and the
body axes are what the flight trace banks with, so a pose file written in an
optical (OpenCV) convention must be rotated into this one before it is passed in;
flipping the sign of y and z only handles a z-up world, not an axis relabelling.

Usage
-----
    python -m worldslider.flightgrid.render poses.txt out.mp4 --fov 90 --fps 10
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass

import cv2
import imageio_ffmpeg
import numpy as np

# body/NED axes -> OpenCV camera axes (x right, y down, z forward)
NED_TO_CAM = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

# BGR colours
C_FLOOR = (60, 60, 235)     # red: ground plane grid
C_CEIL = (235, 120, 60)     # blue: ceiling plane grid
C_WALL = (60, 225, 235)     # yellow: vertical bars around the path
C_TRACE = (235, 235, 235)   # white: flight trace (box corridor)
C_LEVEL = (110, 205, 120)   # green: interior storey boundaries


@dataclass
class FlightGridConfig:
    # scaffold geometry (metres)
    grid_step: float = 1.0       # cell size of the ground / ceiling grid
    reach: float = 18.0          # how far the planes extend beyond the path
    dh_scale: float = 6.0        # half room height = dh_scale * median step
    dh_min: float = 1.5          # lower bound of the half room height
    wall_r_in: float = 1.5       # inner radius of the vertical-bar annulus
    wall_delta: float = 1.0      # thickness of the annulus
    wall_extend: float = 35.0    # inertial run-out of the wall past both ends
    wall_runout_step: float = 3.0  # coarser bar spacing on the run-out
    level_patch_radius: float = 6.0  # interior level lines only near the path
    # flight trace
    box_w: float = 0.6           # corridor width along the body right axis
    box_h: float = 0.2           # corridor height along the body down axis
    ring_dt: float = 0.3         # seconds between rings on the corridor
    t_full: float = 3.0          # trace fully visible within +-t_full seconds
    t_zero: float = 5.0          # trace invisible beyond +-t_zero seconds
    near0: float = 0.2           # near-field alpha roll-off of the trace (m)
    near1: float = 0.8
    # rendering
    line_px: int = 2             # scaffold line width
    trace_px: int = 3            # trace line width
    fade_m: float = 30.0         # depth at which scaffold lines reach the floor
    fade_floor: float = 0.15     # remaining brightness of planes at fade_m
    level_alpha: float = 0.5     # alpha of interior level lines at zero depth
    level_px: int = 2
    near_clip: float = 0.05


def quat_to_R(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def intrinsics(size: int, fov_deg: float) -> np.ndarray:
    f = size / (2 * np.tan(np.radians(fov_deg) / 2))
    return np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])


def _extrapolate(pos, length, step, k=10):
    """Continue the path past an endpoint along its heading (straight line).

    Without the run-out the annulus closes around the last pose in an arc of
    bars standing across the direction of travel; carried far enough, the arc
    lands beyond the distance fade and is never visible.
    """
    if len(pos) < 2 or length <= 0:
        return np.empty((0, 3))
    k = min(k, len(pos) - 1)
    head = pos[-1] - pos[-1 - k]
    n = np.linalg.norm(head)
    if n < 1e-6:
        return np.empty((0, 3))
    t = np.arange(step, length + step, step)[:, None]
    return pos[-1] + head / n * t


def build_segments(pos, quat, fps, cfg: FlightGridConfig):
    """World-space line segments for one clip: list of (p0, p1, colour, time)."""
    segs = []

    def span(p0, p1, col, t=None, sub=1.0):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = max(1, int(np.linalg.norm(p1 - p0) / sub))
        w = np.linspace(0, 1, n + 1)[:, None]
        P = p0 + (p1 - p0) * w
        for i in range(n):
            segs.append((P[i], P[i + 1], col, t))

    # ---- room height from speed, stacked into storeys by vertical extent ----
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dh = max(cfg.dh_min, cfg.dh_scale * float(np.median(d)) if len(d) else cfg.dh_min)
    storey_h = 2 * dh
    zmin, zmax = float(pos[:, 2].min()), float(pos[:, 2].max())
    n_storeys = max(1, int(np.ceil((zmax - zmin + 1.0) / storey_h)))
    zc = 0.5 * (zmin + zmax)
    dh = 0.5 * n_storeys * storey_h
    z_floor, z_ceil = zc + dh, zc - dh  # +z is down

    # ---- path with inertial run-out (defines the wall and the plane extent) ----
    wall_path = pos
    if cfg.wall_extend > 0:
        wall_path = np.vstack([
            _extrapolate(pos[::-1], cfg.wall_extend, cfg.grid_step)[::-1],
            pos,
            _extrapolate(pos, cfg.wall_extend, cfg.grid_step),
        ])
    lo, hi = wall_path.min(0) - cfg.reach, wall_path.max(0) + cfg.reach
    step = cfg.grid_step
    gx = np.arange(np.floor(lo[0] / step) * step, hi[0] + step, step)
    gy = np.arange(np.floor(lo[1] / step) * step, hi[1] + step, step)

    # ---- ground and ceiling grids ----
    for z, col in ((z_floor, C_FLOOR), (z_ceil, C_CEIL)):
        for x in gx:
            span([x, gy[0], z], [x, gy[-1], z], col)
        for y in gy:
            span([gx[0], y, z], [gx[-1], y, z], col)

    # ---- interior storey boundaries: level lines, only near the path ----
    pxy = wall_path[:, :2]
    for j in range(1, n_storeys):
        z = z_floor - j * storey_h
        for x in gx:
            for k in range(len(gy) - 1):
                mid = np.array([x, 0.5 * (gy[k] + gy[k + 1])])
                if np.sqrt(((pxy - mid) ** 2).sum(1)).min() < cfg.level_patch_radius:
                    span([x, gy[k], z], [x, gy[k + 1], z], C_LEVEL)
        for y in gy:
            for k in range(len(gx) - 1):
                mid = np.array([0.5 * (gx[k] + gx[k + 1]), y])
                if np.sqrt(((pxy - mid) ** 2).sum(1)).min() < cfg.level_patch_radius:
                    span([gx[k], y, z], [gx[k + 1], y, z], C_LEVEL)

    # ---- vertical bars in an annulus around the path ----
    XY = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    keep = np.zeros(len(XY), bool)
    nearest = np.zeros(len(XY), int)
    for i in range(0, len(XY), 4096):
        D = np.sqrt(((XY[i:i + 4096, None, :] - pxy[None, :, :]) ** 2).sum(-1))
        dmin = D.min(1)
        nearest[i:i + 4096] = D.argmin(1)
        keep[i:i + 4096] = (dmin > cfg.wall_r_in) & (dmin < cfg.wall_r_in + cfg.wall_delta)
    n_pre = (len(wall_path) - len(pos)) // 2 if cfg.wall_extend > 0 else 0
    in_runout = (nearest < n_pre) | (nearest >= n_pre + len(pos))
    rs = cfg.wall_runout_step
    coarse = (np.abs(np.round(XY[:, 0] / rs) * rs - XY[:, 0]) < 0.01) & \
             (np.abs(np.round(XY[:, 1] / rs) * rs - XY[:, 1]) < 0.01)
    keep &= (~in_runout) | coarse
    for x, y in XY[keep]:
        span([x, y, z_floor], [x, y, z_ceil], C_WALL)

    # ---- flight trace: body-frame box corridor with time-tick rings ----
    corners = []
    for i in range(len(pos)):
        Rb = quat_to_R(quat[i])
        yb, zb = Rb[:, 1], Rb[:, 2]  # body right, body down
        corners.append([pos[i] + sy * 0.5 * cfg.box_w * yb + sz * 0.5 * cfg.box_h * zb
                        for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    next_ring_t = 0.0
    for i in range(len(pos)):
        ti = i / fps
        if i + 1 < len(pos):
            for j in range(4):
                span(corners[i][j], corners[i + 1][j], C_TRACE, t=ti)
        if ti >= next_ring_t:
            ring = corners[i] + [corners[i][0]]
            for a, b in zip(ring[:-1], ring[1:]):
                span(a, b, C_TRACE, t=ti)
            next_ring_t += cfg.ring_dt

    info = dict(dh=round(dh, 3), n_storeys=n_storeys, z_floor=round(z_floor, 3),
                z_ceil=round(z_ceil, 3), verticals=int(keep.sum()), segments=len(segs))
    return segs, info


def _blend_line(img, ua, ub, col, px, alpha, S):
    x0 = int(max(0, min(ua[0], ub[0]) - px - 2)); x1 = int(min(S, max(ua[0], ub[0]) + px + 3))
    y0 = int(max(0, min(ua[1], ub[1]) - px - 2)); y1 = int(min(S, max(ua[1], ub[1]) + px + 3))
    if x1 <= x0 or y1 <= y0:
        return
    sub = img[y0:y1, x0:x1]
    ov = np.zeros_like(sub)
    cv2.line(ov, (int(ua[0]) - x0, int(ua[1]) - y0), (int(ub[0]) - x0, int(ub[1]) - y0),
             col, px, cv2.LINE_AA)
    m = ov.any(-1)
    if m.any():
        sub[m] = (sub[m] * (1.0 - alpha) + ov[m] * alpha).astype(np.uint8)


def render_frame(segs, pos, quat, K, size, t_now, cfg: FlightGridConfig):
    """Project the world segments into one camera and draw them, far to near."""
    img = np.zeros((size, size, 3), np.uint8)
    R = NED_TO_CAM @ quat_to_R(quat).T
    t = -R @ pos
    near = cfg.near_clip
    items = []
    for p0, p1, col, tseg in segs:
        a, b = R @ p0 + t, R @ p1 + t
        if a[2] <= near and b[2] <= near:
            continue
        if a[2] <= near:
            a = a + (b - a) * ((near - a[2]) / (b[2] - a[2] + 1e-12))
        if b[2] <= near:
            b = b + (a - b) * ((near - b[2]) / (a[2] - b[2] + 1e-12))
        ua = (a[:2] / a[2]) @ K[:2, :2].T + K[:2, 2]
        ub = (b[:2] / b[2]) @ K[:2, :2].T + K[:2, 2]
        if not (np.isfinite(ua).all() and np.isfinite(ub).all()):
            continue
        if max(abs(ua).max(), abs(ub).max()) > 1e5:
            continue
        items.append((0.5 * (a[2] + b[2]), ua, ub, col, tseg))

    for z, ua, ub, col, tseg in sorted(items, key=lambda r: -r[0]):
        u = float(np.clip(z / cfg.fade_m, 0.0, 1.0))
        if col == C_TRACE:
            # temporal validity window, then a near-field roll-off (the camera
            # flies inside the corridor); both act on alpha, not brightness
            dt = abs(tseg - t_now)
            vis = float(np.clip((cfg.t_zero - dt) / max(cfg.t_zero - cfg.t_full, 1e-6), 0.0, 1.0))
            a = vis * float(np.clip((z - cfg.near0) / max(cfg.near1 - cfg.near0, 1e-6), 0.0, 1.0))
            if a > 0.01:
                _blend_line(img, ua, ub, col, cfg.trace_px, a, size)
            continue
        if col == C_LEVEL:
            a = cfg.level_alpha * max(1.0 - u, 0.0)
            if a > 0.02:
                _blend_line(img, ua, ub, col, cfg.level_px, a, size)
            continue
        if col == C_WALL:
            c = tuple(int(v * max(1.0 - u, 0.0)) for v in col)          # bars vanish at range
        else:
            c = tuple(int(v * max(1.0 - (1.0 - cfg.fade_floor) * u, cfg.fade_floor)) for v in col)
        cv2.line(img, (int(ua[0]), int(ua[1])), (int(ub[0]), int(ub[1])), c, cfg.line_px, cv2.LINE_AA)
    return img


def render_clip(pos, quat, fps=10, size=640, fov=90.0, cfg: FlightGridConfig | None = None):
    """Yield one BGR uint8 frame per pose."""
    cfg = cfg or FlightGridConfig()
    K = intrinsics(size, fov)
    segs, _ = build_segments(pos, quat, fps, cfg)
    for i, (c, q) in enumerate(zip(pos, quat)):
        yield render_frame(segs, c, q, K, size, i / fps, cfg)


def write_mp4(frames, path, size, fps):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    w = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{size}x{size}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
         "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        w.stdin.write(f.tobytes())
    w.stdin.close()
    w.wait()


def load_poses(path, start=0, frames=None):
    """Rows ``tx ty tz qx qy qz qw``, body-to-world in the x-forward / y-right /
    z-down convention described at the top of this file. No conversion is applied."""
    P = np.loadtxt(path)
    P = P[start:] if frames is None else P[start:start + frames]
    return P[:, :3], P[:, 3:7]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses", help="text file, one 'tx ty tz qx qy qz qw' row per frame")
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=93)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--fov", type=float, default=90.0,
                    help="field of view in degrees; the CLI renders a square frame with a\n"
                         "centred principal point, which is what our clips use. Other\n"
                         "intrinsics work through render_clip(): pass your own K to\n"
                         "render_frame() instead of the one intrinsics() builds.")
    args = ap.parse_args()
    pos, quat = load_poses(args.poses, args.start, args.frames)
    write_mp4(render_clip(pos, quat, args.fps, args.size, args.fov), args.out, args.size, args.fps)
    print(f"wrote {args.out} ({len(pos)} frames)")


if __name__ == "__main__":
    main()
