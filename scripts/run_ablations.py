"""Reproduction, paper ablations, and our step-by-step improvements, all on the same cached features.

    python scripts/run_ablations.py --cache cache/voc_val_b32 --plus configs/clipdiy_plus.yaml
"""
import argparse
import dataclasses
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clipdiy.clip_dense import DenseCLIP  # noqa: E402
from clipdiy.config import load_seg_config  # noqa: E402
from clipdiy.evaluation import BankCache, evaluate, load_cache  # noqa: E402
from clipdiy.prompts import SYNONYMS  # noqa: E402
from clipdiy.segment import SegConfig  # noqa: E402


def experiments(plus: SegConfig):
    base = SegConfig()
    yield "reproduction", "CLIP-DIY (official settings)", base
    yield "paper ablation", "w/o objectness (FOUND)", dataclasses.replace(base, saliency=False)
    for s in (256, 128, 64):
        yield "paper ablation", f"single scale {s}px", dataclasses.replace(base, scales=(s,))

    # steps whose tuned value equals the baseline are skipped here and reported under "rejected"
    steps = [
        ("+ fix prompt article ('an aeroplane')", dict(prompts="fixed")),
        ("+ 80-template prompt ensemble", dict(prompts=plus.prompts)),
        ("+ natural class names / synonyms", dict(synonyms=plus.synonyms)),
        ("+ probabilistic objectness fusion", dict(fusion="prob", temperature=plus.temperature,
                                                   bg_weight=plus.bg_weight)),
        ("+ background-concept prompts", dict(bg_prompts=plus.bg_prompts)),
        ("+ flip TTA", dict(flip=plus.flip)),
        ("+ image-level class filtering", dict(class_filter=plus.class_filter, filter_source=plus.filter_source)),
        ("+ DINO refinement of class probabilities", dict(refine=plus.refine, refine_knn=plus.refine_knn,
                                                          refine_tau=plus.refine_tau, refine_iters=plus.refine_iters)),
        ("+ DINO propagation of objectness", dict(sal_refine=plus.sal_refine)),
        ("+ bilinear score upsampling", dict(upsample=plus.upsample)),
    ]
    cfg = base
    for name, change in steps:
        new = dataclasses.replace(cfg, **change)
        if new != cfg:
            cfg = new
            yield "cumulative", name, cfg
    assert cfg == plus, "cumulative steps must end at the tuned config"

    yield "final", "CLIP-DIY+ (full)", plus
    removals = {
        "probabilistic fusion": dict(fusion="original", bg_prompts="background", refine=False, sal_refine=0.0),
        "background concepts": dict(bg_prompts="background"),
        "DINO class refinement": dict(refine=False),
        "DINO objectness propagation": dict(sal_refine=0.0),
        "objectness (FOUND)": dict(saliency=False),
    }
    for name, change in removals.items():
        cfg = dataclasses.replace(plus, **change)
        if cfg != plus:
            yield "leave-one-out", f"CLIP-DIY+ w/o {name}", cfg

    # ideas rejected during tuning on the train subset, shown added on top of the final model
    rejected = {
        "80-template prompt ensemble": dict(prompts="ensemble"),
        "class-name synonyms": dict(synonyms=True),
        "flip TTA": dict(flip=True),
        "class filter (whole image, p>=0.05)": dict(class_filter=0.05, filter_source="global"),
        "class filter (max over windows, p>=0.5)": dict(class_filter=0.5, filter_source="windows"),
    }
    for name, change in rejected.items():
        cfg = dataclasses.replace(plus, **change)
        if cfg != plus:
            yield "rejected", f"CLIP-DIY+ + {name}", cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--plus", default="configs/clipdiy_plus.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    plus = load_seg_config(args.plus)
    meta, _, items, gts = load_cache(args.cache)
    banks = BankCache(DenseCLIP(meta["clip_model"]), meta["classes"], SYNONYMS.get(meta["dataset"]))
    rows, ref = [], None
    for group, name, cfg in experiments(plus):
        cm = evaluate(items, gts, banks.get(cfg), cfg)
        miou = 100 * cm.miou()
        ref = miou if ref is None else ref
        rows.append({"group": group, "experiment": name, "mIoU": round(miou, 2), "delta": round(miou - ref, 2),
                     "pixel_acc": round(100 * cm.pixel_acc(), 2),
                     **{f"iou_{c}": round(100 * v, 1) for c, v in zip(meta["classes"], cm.iou())}})
        print(f"[{group:>14s}] {name:<42s} mIoU {miou:6.2f}  ({miou - ref:+.2f})", flush=True)

    tag = os.path.basename(os.path.normpath(args.cache))
    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"ablations_{tag}.csv"), index=False)
    with open(os.path.join(args.out, f"ablations_{tag}.md"), "w") as f:
        f.write(df[["group", "experiment", "mIoU", "delta", "pixel_acc"]].to_markdown(index=False))
    print(f"saved to {args.out}/ablations_{tag}.*")


if __name__ == "__main__":
    main()
