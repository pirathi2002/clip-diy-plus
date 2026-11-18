"""Segment one image with any text prompts.

    python scripts/demo.py third_party/clip-diy/images/diy-5090797_640.jpg scrabble pencil "rubber stamp" "paper clip"
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clipdiy.config import load_seg_config  # noqa: E402
from clipdiy.data import read_rgb  # noqa: E402
from clipdiy.pipeline import PALETTE, CLIPDIYPipeline, overlay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("classes", nargs="+")
    ap.add_argument("--config", default="configs/clipdiy_plus.yaml")
    ap.add_argument("--compare", default="configs/original.yaml", help="also run this config side by side")
    ap.add_argument("--out", default="results/demo.png")
    args = ap.parse_args()

    img = read_rgb(args.image)
    cfgs = [("CLIP-DIY", load_seg_config(args.compare))] if args.compare else []
    cfgs.append(("CLIP-DIY+ (ours)", load_seg_config(args.config)))
    pipe = CLIPDIYPipeline(cfgs[0][1])
    panels = [("input", img)]
    for name, cfg in cfgs:
        pipe.cfg = cfg
        labels, sal = pipe(img, args.classes)
        panels.append((name, overlay(img, labels)))
    panels.insert(1, ("FOUND objectness", sal))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5 * img.shape[0] / img.shape[1] + 1))
    for ax, (title, im) in zip(axes, panels):
        ax.imshow(im, cmap="magma" if im.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    handles = [mpatches.Patch(color=PALETTE[i + 1] / 255, label=c) for i, c in enumerate(args.classes)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
