"""Cut flight recordings into training windows and render their FlightGrids.

Input: a directory with one sub-directory per trajectory, each holding
    rgb.mp4        the recorded flight (square frames)
    poses.txt      one 'tx ty tz qx qy qz qw' row per frame, body-to-world in the
                   x-forward / y-right / z-down convention of
                   worldslider/flightgrid/render.py. No conversion is applied here,
                   so convert optical-frame poses before running this script.
    caption.txt    the environment caption for this trajectory (no camera-motion
                   words). It is copied to every window of the trajectory; the
                   per-clip captions used in the paper come from a vision-language
                   model run over each window, which is not part of this release.

Output (the layout read by the training configs):
    <out>/videos/<traj>_s<start>.mp4       93-frame RGB window
    <out>/flightgrid/<traj>_s<start>.mp4   FlightGrid rendered from the window's poses only
    <out>/poses/<traj>_s<start>.txt        the window's poses (used by the geometry reward)
    <out>/captions/<traj>_s<start>.json    {"caption": ...}

    python scripts/prepare_dataset.py raw/ datasets/worldslider --fov 90 --fps 10 --stride 70
"""
import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from worldslider.flightgrid.render import render_clip, write_mp4

WINDOW = 93


def cut_rgb(src, dst, start, fps, size):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", f"select=gte(n\\,{start})*lt(n\\,{start + WINDOW}),setpts=N/({fps}*TB),scale={size}:{size}",
                    "-r", str(fps), "-frames:v", str(WINDOW),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--stride", type=int, default=70)
    ap.add_argument("--caption", default="Aerial FPV view.", help="fallback caption")
    args = ap.parse_args()

    out = Path(args.out)
    for d in ("videos", "flightgrid", "poses", "captions"):
        (out / d).mkdir(parents=True, exist_ok=True)

    for traj in sorted(p for p in Path(args.raw).iterdir() if (p / "poses.txt").exists()):
        P = np.loadtxt(traj / "poses.txt")
        cap_file = traj / "caption.txt"
        caption = cap_file.read_text().strip() if cap_file.exists() else args.caption
        for start in range(0, len(P) - WINDOW + 1, args.stride):
            name = f"{traj.name}_s{start:05d}"
            if (out / "flightgrid" / f"{name}.mp4").exists():
                continue
            win = P[start:start + WINDOW]
            np.savetxt(out / "poses" / f"{name}.txt", win, fmt="%.6f")
            cut_rgb(traj / "rgb.mp4", out / "videos" / f"{name}.mp4", start, args.fps, args.size)
            write_mp4(render_clip(win[:, :3], win[:, 3:7], args.fps, args.size, args.fov),
                      out / "flightgrid" / f"{name}.mp4", args.size, args.fps)
            (out / "captions" / f"{name}.json").write_text(json.dumps({"caption": caption}))
            print(name)


if __name__ == "__main__":
    main()
