"""Stage 3 - flight alignment: reward-weighted DMD from generated rollouts.

Every generator update of DMD2 produces a few-step sample.  This model decodes
that sample, recovers its camera trajectory (worldslider.reward), turns the
geometry reward into a per-sample weight

    w_k = (1 - lambda_k) + lambda_k * clip(exp((r - mu_k) / (beta * max(sigma_k, sigma_min))), w_min, w_max)

and multiplies the sample's DMD loss by it. Rewards and weights are
stop-gradient, so the score-difference target of DMD is untouched; only the
relative influence of rollouts changes. Samples whose flight cannot be verified
train with w = 1.

Counting and schedule. WARMUP and RAMP count *trainer iterations*, of which one in
every five is a student update (the other four update the fake score), and the
verifier runs on every fourth student update. lambda_k rises by 1/RAMP per iteration
and is then clipped at LAMBDA_MAX, so with the defaults it reaches the cap after
LAMBDA_MAX * RAMP = 500 iterations (100 student updates), not after RAMP of them.
During WARMUP nothing is scored at all - no reward is computed and no statistics are
collected - so it is a plain DMD stretch; the runs behind the paper used WARMUP = 0,
starting from an already distilled student whose first samples are sharp enough to
score.

Implemented as a subclass of the in-tree DMD2 model so the vendored file stays
untouched. Experiment `worldslider_flight_alignment` is the stage-2 experiment
with the model class swapped and the stage-2 student as the warm start:

    WORLDSLIDER_STUDENT=<stage-2 checkpoint dir> \
    torchrun --standalone --nproc_per_node=<N> -m scripts.train \
        --config=cosmos_transfer2/_src/interactive/configs/registry_transfer2p5.py \
        -- experiment=worldslider_flight_alignment
"""
import copy
import os
from typing import Any, Dict

import numpy as np
import torch
from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.interactive.methods.distribution_matching.dmd2 import DMD2Model
from cosmos_transfer2.experiments.worldslider.dmd_distillation import worldslider_dmd

EVERY = int(os.environ.get("WORLDSLIDER_RL_EVERY", "4"))         # score every k-th generator step
WARMUP = int(os.environ.get("WORLDSLIDER_RL_WARMUP", "0"))      # trainer iterations; the paper runs used 0
RAMP = int(os.environ.get("WORLDSLIDER_RL_RAMP", "2000"))
LAMBDA_MAX = float(os.environ.get("WORLDSLIDER_RL_LAMBDA", "0.25"))
BETA = float(os.environ.get("WORLDSLIDER_RL_BETA", "1.0"))
W_MIN = float(os.environ.get("WORLDSLIDER_RL_WMIN", "0.9"))
W_MAX = float(os.environ.get("WORLDSLIDER_RL_WMAX", "1.15"))


class RewardWeightedDMD2Model(DMD2Model):
    def __init__(self, config):
        super().__init__(config)
        self._weighter = None
        self._batch = None
        self._sample = None
        self._start_iter = None

    # -- plumbing: keep the batch (clip names) and the student's sample ------
    def single_train_step(self, data_batch: Dict[str, Any], iteration: int):
        self._batch = data_batch
        try:
            return super().single_train_step(data_batch, iteration)
        finally:
            self._batch = None

    def backward_simulation(self, *args, **kwargs):
        out = super().backward_simulation(*args, **kwargs)
        if kwargs.get("with_grad") or (len(args) >= 4 and args[3]):
            self._sample = out.detach()
        return out

    def _clip_names(self, n):
        keys = (self._batch or {}).get("__key__")
        keys = [keys] if isinstance(keys, str) else list(keys or [])
        return (keys + [None] * n)[:n]

    # -- reward ----------------------------------------------------------------
    @torch.no_grad()
    def _rewards(self, latents):
        from worldslider.reward.geometry_reward import geometry_reward

        clips = self._clip_names(latents.shape[0])
        pixels = self.decode(latents)                                     # B,C,T,H,W in [-1,1]
        pix = ((pixels.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
        rewards = []
        for i, clip in enumerate(clips):
            if clip is None:
                rewards.append(None)
                continue
            try:
                frames = [np.ascontiguousarray(pix[i, ::-1, t].transpose(1, 2, 0)) for t in range(pix.shape[2])]
                rewards.append(geometry_reward(frames, clip).get("reward"))
            except Exception as e:  # noqa: BLE001 - a failed verifier abstains
                print(f"[flight-alignment] reward failed on {clip}: {type(e).__name__}: {e}", flush=True)
                rewards.append(None)
        return rewards

    def training_step_generator(self, x0_B_C_T_H_W, condition, uncondition, iteration):
        output_batch, loss = super().training_step_generator(x0_B_C_T_H_W, condition, uncondition, iteration)
        if self._start_iter is None:
            self._start_iter = int(os.environ.get("WORLDSLIDER_RL_BASE_ITER", iteration))
        step = iteration - self._start_iter
        if step < WARMUP or iteration % EVERY or self._sample is None:
            return output_batch, loss

        from worldslider.reward.geometry_reward import RewardWeighter

        if self._weighter is None:
            self._weighter = RewardWeighter(beta=BETA, w_min=W_MIN, w_max=W_MAX)

        rewards = self._rewards(self._sample)
        scored = [r for r in rewards if r is not None]
        if not scored:
            return output_batch, loss

        # weights from the statistics *before* this step, then update them;
        # no cross-rank synchronisation (per-rank statistics converge to the
        # same distribution and a conditional collective would deadlock)
        w_scored = self._weighter.weights(scored)
        self._weighter.update(scored)
        w = np.ones(len(rewards))
        w[[i for i, r in enumerate(rewards) if r is not None]] = w_scored
        lam = min(min(max((step - WARMUP) / max(RAMP, 1), 0.0), 1.0), LAMBDA_MAX)
        w = 1.0 + lam * (w - 1.0)

        w_t = torch.as_tensor(w, device=loss.device, dtype=loss.dtype)
        if w_t.shape[0] != loss.shape[0]:
            return output_batch, loss
        output_batch["rl_lambda"] = torch.tensor(lam)
        output_batch["rl_reward_mean"] = torch.tensor(float(np.mean(scored)))
        output_batch["rl_weight_mean"] = w_t.mean().detach()
        output_batch["rl_abstain_frac"] = torch.tensor(1.0 - len(scored) / len(rewards))
        return output_batch, loss * w_t


worldslider_flight_alignment = copy.deepcopy(worldslider_dmd)
worldslider_flight_alignment["job"] = dict(project="worldslider", group="flight_alignment", name="worldslider_flight_alignment")
worldslider_flight_alignment["model"]["_target_"] = "cosmos_transfer2.experiments.worldslider.flight_alignment.RewardWeightedDMD2Model"
if os.environ.get("WORLDSLIDER_STUDENT"):
    worldslider_flight_alignment["checkpoint"]["load_path"] = os.environ["WORLDSLIDER_STUDENT"]

ConfigStore.instance().store(
    group="experiment", package="_global_", name="worldslider_flight_alignment", node=worldslider_flight_alignment
)
