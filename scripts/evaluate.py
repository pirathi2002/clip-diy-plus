"""Evaluate one configuration on cached features.

    python scripts/evaluate.py --cache cache/voc_val_b32 --config configs/clipdiy_plus.yaml
    python scripts/evaluate.py --cache cache/voc_val_b32 --config configs/original.yaml --set saliency=false
"""
import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clipdiy.clip_dense import DenseCLIP  # noqa: E402
from clipdiy.config import load_seg_config  # noqa: E402
from clipdiy.evaluation import evaluate, load_cache  # noqa: E402
from clipdiy.prompts import SYNONYMS  # noqa: E402
from clipdiy.segment import build_text_bank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. fusion=prob refine=true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write metrics json here")
    args = ap.parse_args()

    cfg = load_seg_config(args.config, args.set)
    meta, ids, items, gts = load_cache(args.cache, args.limit)
    clip = DenseCLIP(meta["clip_model"], device=args.device)
    bank = build_text_bank(clip, meta["classes"], cfg, SYNONYMS.get(meta["dataset"]))
    t0 = time.time()
    cm = evaluate(items, gts, bank, cfg, progress=True)
    dt = (time.time() - t0) / len(items)

    print(f"{cfg}\n\nmIoU {100 * cm.miou():.2f} | pixel acc {100 * cm.pixel_acc():.2f} | {1000 * dt:.1f} ms/img (cached)")
    for name, iou in zip(meta["classes"], cm.iou()):
        print(f"  {name:>14s} {100 * iou:5.1f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": dataclasses.asdict(cfg), "mIoU": cm.miou(), "pixel_acc": cm.pixel_acc(),
                       "per_class": dict(zip(meta["classes"], cm.iou().tolist()))}, f, indent=2)


if __name__ == "__main__":
    main()
