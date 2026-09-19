# WorldSlider — code

Minimal, self-contained release of the parts of WorldSlider that are ours:
the FlightGrid renderer, the control-input hook into Cosmos-Transfer2.5, the
three training stages, the geometry reward, and a generation script.
Cluster launch scripts, evaluation harnesses and figure code are not included.

```
code/
├── worldslider/
│   ├── flightgrid/render.py        FlightGrid: poses -> appearance-free control video (Sec. 3.1)
│   ├── cosmos/
│   │   ├── flightgrid_control.patch  FlightGrid control input for cosmos-transfer2.5 (4 files)
│   │   ├── control_adaptation.py     stage 1: supervised control adaptation (Sec. 3.2)
│   │   ├── dmd_distillation.py       stage 2: 4-step DMD distillation (Sec. 3.3)
│   │   └── flight_alignment.py       stage 3: reward-weighted DMD from rollouts (Sec. 3.4)
│   └── reward/geometry_reward.py   trajectory recovery reward, gates, abstention, weighting
├── scripts/
│   ├── prepare_dataset.py          recordings -> 93-frame windows + FlightGrids + poses + captions
│   ├── train.sh                    torchrun commands for the three stages
│   └── generate.py                 FlightGrid + caption (+ first frame) -> parallel-world video
└── requirements.txt
```

## FlightGrid

`worldslider/flightgrid/render.py` needs only NumPy, OpenCV and ffmpeg. Poses are
body-to-world rows `tx ty tz qx qy qz qw`, with the body and the world frame both
x-forward / y-right / z-down (the identity quaternion looks along world +x). The
renderer converts to the OpenCV camera frame itself, so a pose file written in an
optical convention has to be rotated into this one first. The CLI renders a square
frame with a centred principal point; other intrinsics go through `render_clip()`.

```bash
python -m worldslider.flightgrid.render poses.txt flightgrid.mp4 --fov 90 --fps 10 --frames 93
```

The render contains the time-indexed flight trace (a body-frame box corridor
with a ring every 0.3 s, visible only inside a ±3–5 s window around the current
frame) and the metric scaffold (1 m ground/ceiling grids, storey level lines,
vertical bars in an annulus around the path). All geometry is fixed in world
coordinates, so the only motion in the control video is the camera's. With
distance, the two outer grids fade to a floor of 15 % brightness while the
vertical bars, the storey lines and the trace itself fade to nothing.

## Training

1. Clone [cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
   at commit `2ff49d0`, set up its environment and download the
   `Cosmos-Transfer2.5-2B` depth-control checkpoint as documented there.
2. `git apply worldslider/cosmos/flightgrid_control.patch` — adds the
   `control_input_flightgrid` input (dataset loader, augmentor, conditioner) and
   lets the distillation config root import user experiments.
3. Copy `worldslider/cosmos/*.py` to `cosmos_transfer2/experiments/worldslider/`
   (with an empty `__init__.py`) and put `code/` on `PYTHONPATH`.
4. Build the dataset: `python scripts/prepare_dataset.py raw/ datasets/worldslider`.
   Captions describe the environment only; the flight is given by FlightGrid. The
   script copies one caption per trajectory to all its windows — the paper's captions
   are produced per window by a vision-language model, which is not part of this
   release, so supply them yourself to reproduce that setting.
5. Run the stages: `scripts/train.sh adapt`, then `distill`
   (`WORLDSLIDER_TEACHER` = consolidated stage-1 checkpoint), then `align`
   (`WORLDSLIDER_STUDENT` = stage-2 checkpoint; needs
   [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for
   the geometry reward).

Stage 1 initialises from the official depth-control model and trains only the
control branch on FlightGrid; stage 2 distills it into a 4-step student with the
in-tree DMD2 recipe; stage 3 continues DMD with per-sample weights derived from
the recovered trajectory of each rollout (`RewardWeightedDMD2Model`).

### Paper configuration

The defaults reproduce Table 3 of the paper (training configuration) on four
GPUs; the table below lists the setting, its value in the paper and where it is
set, so that any deviation is explicit.

| Setting | Paper | Where |
|---|---|---|
| GPUs | 4 | `NGPUS=4` (default in `train.sh`) |
| Stage 1 global batch | 8 clips | `WORLDSLIDER_BS=2` × 4 GPUs (data parallel) |
| Stage 1 optimiser | AdamW, control LR 8.63e-5, weight decay 1e-3 | inherited from the cosmos-transfer2.5 base experiment |
| Stage 2/3 global batch | 4 clips | `WORLDSLIDER_BS=1` × `WORLDSLIDER_FSDP=4` |
| Stage 2/3 optimiser | student LR 1e-6, fake-score LR 2e-7, weight decay 1e-2, 1:4 student : fake-score updates, teacher CFG 3 | inherited from the in-tree DMD2 recipe |
| Verifier interval | every 4 student updates (every 20 trainer iterations) | `WORLDSLIDER_RL_EVERY=4` |
| Reward temperature β | 1 | `WORLDSLIDER_RL_BETA=1.0` |
| Weight range [w_min, w_max] | [0.9, 1.15] | `WORLDSLIDER_RL_WMIN=0.9`, `WORLDSLIDER_RL_WMAX=1.15` |
| λ_max / ramp slope | 0.25, rising by 1/2000 per trainer iteration | `WORLDSLIDER_RL_LAMBDA=0.25`, `WORLDSLIDER_RL_RAMP=2000` — λ is clipped at λ_max, so it reaches the cap after 500 iterations (100 student updates), not after 2,000 |
| Warm-up before scoring | 0 in the runs behind the paper | `WORLDSLIDER_RL_WARMUP=0` — a plain DMD stretch: nothing is scored and no statistics are collected while it lasts |
| Resolution / clip length | 640 × 640, 93 frames at 10 fps | `WORLDSLIDER_RES=480p` (square → 640 × 640), `NUM_FRAMES=93` |
| Conditioning / steps, stages 2–3 | inherited from the recipe: conditional latent frames sampled {0: 0.6, 1: 0.2, 2: 0.2}, each generator step simulates 1–4 sampling steps | set in `overrides` in `dmd_distillation.py` if you want stage-1 conditioning |

Only the stage-1 per-GPU batch size deviates from its default (`WORLDSLIDER_BS`
defaults to 1 so that a single-GPU smoke run fits); set `WORLDSLIDER_BS=2` for
the paper's global batch of 8.

Two things this release does differently from the runs behind the paper, both
deliberate:

- **Relative-rotation gate.** `geometry_reward.py` carries the recovered relative
  rotation into the prescribed frame as `Q R Q^T` before comparing it. The training
  code conjugated the prescribed rotation instead, which is not the same whenever the
  gauge does not commute with the rotation — on a synthetic flight that differs only by
  a relabelling of the camera axes it reports about 11° where the corrected form
  reports 0. It affects only the bounded gate `h_rot`, whose source-flight floor was
  computed the same way, but reward caches written by the two versions are not
  interchangeable: delete `reward_floors.json` before mixing them.
- **Sampling entry point.** `scripts/generate.py` samples through the adapted model's
  own sampler, which evaluates the network twice per step (conditional and
  unconditional) and combines them as `v_cond + guidance·(v_cond − v_uncond)`. Four
  steps therefore cost eight network evaluations, and `--guidance 1` is not a plain
  conditional prediction. This is the path the reported videos were generated with.

## Generation

```bash
torchrun --standalone --nproc_per_node=1 scripts/generate.py \
    --checkpoint checkpoints/worldslider_student.pt \
    --flightgrid flights/arc_turn_flightgrid.mp4 \
    --caption "Aerial FPV view. A coastal town at golden hour, terracotta roofs and a harbour." \
    --first-frame worlds/coastal_town.png \
    --steps 4 --guidance 1 --out out/arc_turn_coastal_town.mp4
```

A prescribed trajectory (from a recording, a simulator, or written by hand) is
first rendered to FlightGrid; the same FlightGrid can then be paired with any
caption or first frame to generate parallel worlds of that flight.
