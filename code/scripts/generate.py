"""Generate a parallel world of a flight.

Inputs: a FlightGrid control video (rendered from the prescribed trajectory), an
environment caption and, optionally, a first frame of the target world.
Output: an RGB video that follows the prescribed flight inside the described
world.

The model is instantiated from the stage-1 experiment config and loaded from a
consolidated `.pt` checkpoint; the 4-step distilled student (stage 2/3) uses
the same architecture and loads the same way. Run from a cosmos-transfer2.5
checkout with the WorldSlider files installed (see code/README.md):

    torchrun --standalone --nproc_per_node=1 scripts/generate.py \
        --checkpoint checkpoints/worldslider_student.pt \
        --flightgrid flights/arc_turn_flightgrid.mp4 \
        --caption "Aerial FPV view. A coastal town at golden hour, ..." \
        --first-frame worlds/coastal_town.png \
        --steps 4 --guidance 1 --out out/arc_turn_coastal_town.mp4

Use --steps 35 --guidance 7 for the multi-step stage-1 model.

Sampling cost. This entry point uses the sampler of the adapted model, which evaluates
the network twice per step, conditional and unconditional, and combines them as
`v = v_cond + guidance * (v_cond - v_uncond)`. A four-step run therefore performs eight
network evaluations, and `--guidance 1` does not disable the combination: it is the
setting the reported results were produced with, not a plain conditional prediction.
Pass `--guidance 0` for the conditional prediction alone (still two evaluations per
step). The script prints the step and evaluation counts it used.
"""
import argparse
import importlib
import os

import cv2
import numpy as np
import torch

from cosmos_oss.init import init_environment
from cosmos_transfer2._src.imaginaire.utils import misc
from cosmos_transfer2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_transfer2._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp

CONFIG = "cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py"
EXPERIMENT = "worldslider_control_adaptation"


def read_video(path, size):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames), fps


def read_image(path, size):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def build_batch(args):
    ctrl, fps = read_video(args.flightgrid, args.size)           # T,H,W,3 uint8
    T = ctrl.shape[0]
    if args.first_frame:
        first = read_image(args.first_frame, args.size)
        video = np.repeat(first[None], T, axis=0)                # only frame 0 conditions the model
        n_cond = 1
    else:
        video = np.zeros_like(ctrl)
        n_cond = 0
    to_tensor = lambda a: torch.from_numpy(a).permute(3, 0, 1, 2)[None].contiguous()  # 1,3,T,H,W uint8
    return {
        "video": to_tensor(video),
        "control_input_flightgrid": to_tensor(ctrl),
        "ai_caption": [args.caption],
        "fps": torch.tensor([float(fps)]),
        "image_size": torch.tensor([[args.size] * 4]),
        "padding_mask": torch.ones(1, 1, args.size, args.size),
        "num_frames": torch.tensor([T]),
        "chunk_index": torch.tensor([0]),
        "__key__": [os.path.basename(args.flightgrid)],
        NUM_CONDITIONAL_FRAMES_KEY: n_cond,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="consolidated .pt checkpoint")
    ap.add_argument("--flightgrid", required=True, help="FlightGrid control video (.mp4)")
    ap.add_argument("--caption", required=True, help="environment caption; describe the world, not the motion")
    ap.add_argument("--first-frame", default=None, help="optional first frame of the target world")
    ap.add_argument("--out", required=True, help="output .mp4 (extension added if missing)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    init_environment()
    config = importlib.import_module(get_config_module(CONFIG)).make_config()
    config = override(config, [f"experiment={EXPERIMENT}", f"checkpoint.load_path={args.checkpoint}",
                               "job.name=worldslider_generate"])
    config.validate()
    config.freeze()
    model = create_model_from_consolidated_checkpoint_with_fsdp(config)
    model.eval()

    torch.manual_seed(args.seed)
    print(f"sampling: {args.steps} steps, guidance {args.guidance}, "
          f"{2 * args.steps} network evaluations, seed {args.seed}", flush=True)
    batch = misc.to(build_batch(args), device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
            emb = model.text_encoder.compute_text_embeddings_online(batch, model.input_caption_key)
            batch["t5_text_embeddings"] = emb
            batch["t5_text_mask"] = torch.ones(emb.shape[0], emb.shape[1], device="cuda")
        _, x0, _ = model.get_data_and_condition(batch)
        sample = model.generate_samples_from_batch(
            batch, guidance=args.guidance, seed=args.seed, state_shape=x0.shape[1:],
            n_sample=x0.shape[0], num_steps=args.steps, is_negative_prompt=False)
        video = model.decode(sample).float().cpu()                 # 1,3,T,H,W in [-1,1]

    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        out = args.out[:-4] if args.out.endswith(".mp4") else args.out
        save_img_or_video((video[0] + 1.0) / 2.0, out, fps=int(batch["fps"][0]))
        print(f"wrote {out}.mp4")


if __name__ == "__main__":
    main()
