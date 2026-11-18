"""Run the image encoders once per image and cache the results.

    python scripts/extract_features.py --dataset voc --root data --split val --out cache/voc_val_b32
"""
import argparse
import json
import os
import pickle
import random
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clipdiy.clip_dense import DenseCLIP  # noqa: E402
from clipdiy.data import load_dataset  # noqa: E402
from clipdiy.features import ExtractConfig, extract  # noqa: E402
from clipdiy.found import FOUND  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="voc", choices=["voc", "coco"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--subset", type=int, default=0, help="random subset size (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip-model", default="laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--windows", type=int, nargs="+", default=[256, 128, 64])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.root, args.split)
    order = list(range(len(ds)))
    if args.subset:
        order = sorted(random.Random(args.seed).sample(order, args.subset))
    ids = [str(ds.ids[i]) for i in order]

    os.makedirs(os.path.join(args.out, "feats"), exist_ok=True)
    meta = {"dataset": args.dataset, "split": args.split, "clip_model": args.clip_model, "size": args.size,
            "windows": args.windows, "classes": ds.classes, "ids": ids}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)

    gt_path = os.path.join(args.out, "gt.pkl")
    gts = {}
    if os.path.exists(gt_path):
        with open(gt_path, "rb") as f:
            gts = pickle.load(f)

    clip = DenseCLIP(args.clip_model)
    found = FOUND()
    cfg = ExtractConfig(size=args.size, windows=tuple(args.windows))
    t0, n_done = time.time(), 0
    for i, img_id in zip(order, tqdm(ids)):
        path = os.path.join(args.out, "feats", f"{img_id}.pt")
        if img_id not in gts:
            gts[img_id] = ds.gt(i)
        if os.path.exists(path):
            continue
        torch.save(extract(ds.image(i), clip, found, cfg), path)
        n_done += 1
    with open(gt_path, "wb") as f:
        pickle.dump(gts, f)
    if n_done:
        print(f"extracted {n_done} images, {(time.time() - t0) / n_done:.3f} s/img")


if __name__ == "__main__":
    main()
