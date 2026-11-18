"""Qualitative comparison on VOC val: image | objectness | GT | CLIP-DIY | CLIP-DIY+.

Rows are the images where CLIP-DIY+ gains the most and loses the most per-image mIoU, to show both
improvements and remaining failure modes. Also writes the ablation bar chart if the CSV exists.

    python scripts/make_figures.py --cache cache/voc_val_b32 --root data
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clipdiy.clip_dense import DenseCLIP  # noqa: E402
from clipdiy.config import load_seg_config  # noqa: E402
from clipdiy.data import load_dataset  # noqa: E402
from clipdiy.evaluation import BankCache, load_cache, predict  # noqa: E402
from clipdiy.metrics import ConfusionMatrix  # noqa: E402
from clipdiy.prompts import SYNONYMS  # noqa: E402

VOC_COLORS = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
                       [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                       [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
                       [128, 192, 0], [0, 64, 128]], np.uint8)


def colorize(labels):
    out = VOC_COLORS[np.clip(labels, 0, 20)]
    out[labels == 255] = 255
    return out


def image_miou(pred, gt, n):
    cm = ConfusionMatrix(n)
    cm.update(pred, gt)
    present = (cm.mat.sum(0) + cm.mat.sum(1)) > 0
    return cm.iou()[present].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--plus", default="configs/clipdiy_plus.yaml")
    ap.add_argument("--n-best", type=int, default=5)
    ap.add_argument("--n-worst", type=int, default=2)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta, ids, items, gts = load_cache(args.cache)
    n = len(meta["classes"])
    cfgs = {"CLIP-DIY": load_seg_config("configs/original.yaml"), "CLIP-DIY+ (ours)": load_seg_config(args.plus)}
    banks = BankCache(DenseCLIP(meta["clip_model"]), meta["classes"], SYNONYMS.get(meta["dataset"]))

    preds = {k: [predict(it, banks.get(c), c, gt.shape).cpu().numpy() for it, gt in zip(items, gts)]
             for k, c in cfgs.items()}
    gains = np.array([image_miou(b, gt, n) - image_miou(a, gt, n)
                      for a, b, gt in zip(*preds.values(), gts)])
    order = np.argsort(-gains)
    picks = list(order[:args.n_best]) + list(order[::-1][:args.n_worst])

    ds = load_dataset(meta["dataset"], args.root, meta["split"])
    index = {str(x): i for i, x in enumerate(ds.ids)}
    fig, axes = plt.subplots(len(picks), 5, figsize=(17, 3.3 * len(picks)))
    for r, k in enumerate(picks):
        gt = gts[k]
        img = cv2.resize(ds.image(index[ids[k]]), gt.shape[::-1], interpolation=cv2.INTER_LINEAR)
        sal = F.interpolate(items[k]["sal"].float()[None, None], size=gt.shape, mode="bilinear")[0, 0].numpy()
        panels = [(ids[k], img), ("FOUND objectness", sal), ("ground truth", colorize(gt))]
        panels += [(f"{name}  mIoU {100 * image_miou(p[k], gt, n):.0f}", colorize(p[k])) for name, p in preds.items()]
        for c, (title, im) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im, cmap="magma" if im.ndim == 2 else None)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        present = sorted((set(np.unique(gt)) | set(np.unique(preds["CLIP-DIY"][k])) |
                          set(np.unique(preds["CLIP-DIY+ (ours)"][k]))) - {0, 255})
        axes[r, 4].text(1.02, 0.5, "\n".join(meta["classes"][i] for i in present), transform=axes[r, 4].transAxes,
                        va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "qualitative_voc.png"), dpi=80)
    print(f"saved {args.out}/qualitative_voc.png  (gain rows: {[ids[k] for k in picks]})")

    csv = os.path.join("results", f"ablations_{os.path.basename(os.path.normpath(args.cache))}.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        df = df[df.group.isin(["reproduction", "cumulative"])]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(df.experiment[::-1], df.mIoU[::-1], color=["#2b7bba"] * (len(df) - 1) + ["#888"])
        for y, v in enumerate(df.mIoU[::-1]):
            ax.text(v + 0.2, y, f"{v:.1f}", va="center", fontsize=9)
        ax.set_xlim(df.mIoU.min() - 5, df.mIoU.max() + 3)
        ax.set_xlabel("PASCAL VOC val mIoU (%)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ablation_voc.png"), dpi=110)
        print(f"saved {args.out}/ablation_voc.png")


if __name__ == "__main__":
    main()
