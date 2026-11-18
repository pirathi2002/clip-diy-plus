"""Choose CLIP-DIY+ hyper-parameters by coordinate descent on a held-out subset of the VOC *train* split.

The validation set is never used here, so the val numbers reported afterwards are not tuned on test data.

    python scripts/tune.py --cache cache/voc_train300_b32 --out configs/clipdiy_plus.yaml
"""
import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clipdiy.clip_dense import DenseCLIP  # noqa: E402
from clipdiy.config import save_seg_config  # noqa: E402
from clipdiy.evaluation import BankCache, evaluate, load_cache  # noqa: E402
from clipdiy.prompts import SYNONYMS  # noqa: E402
from clipdiy.segment import SegConfig  # noqa: E402

START = SegConfig(prompts="ensemble", synonyms=True, bg_prompts="background", flip=True, fusion="prob",
                  temperature=100.0, bg_weight=1.0, class_filter=0.0, refine=False, upsample="score")

GRID = {
    "prompts": ["fixed", "ensemble"],
    "synonyms": [False, True],
    "bg_prompts": ["background", "concepts", "none"],
    "temperature": [25.0, 50.0, 100.0, 200.0],
    "bg_weight": [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5],
    "class_filter": [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
    "filter_source": ["global", "windows"],
    "refine": [False, True],
    "sal_refine": [0.0, 0.5, 1.0, 2.0, 3.0],
    "refine_knn": [16, 32, 64, 128, 256],
    "refine_tau": [0.1, 0.2, 0.5, 1.0, 2.0],
    "refine_iters": [1, 2, 3, 5, 8, 12],
    "flip": [False, True],
    "upsample": ["label", "score"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="configs/clipdiy_plus.yaml")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    meta, _, items, gts = load_cache(args.cache)
    banks = BankCache(DenseCLIP(meta["clip_model"]), meta["classes"], SYNONYMS.get(meta["dataset"]))
    seen = {}

    def score(cfg):
        key = tuple(sorted(dataclasses.asdict(cfg).items()))
        if key not in seen:
            seen[key] = evaluate(items, gts, banks.get(cfg), cfg).miou()
        return seen[key]

    best = START
    best_score = score(best)
    print(f"start: {100 * best_score:.2f}")
    for r in range(args.rounds):
        for key, values in GRID.items():
            if key.startswith("refine_") and not (best.refine or best.sal_refine > 0):
                continue
            results = {v: score(dataclasses.replace(best, **{key: v})) for v in values}
            v_best = max(results, key=results.get)
            print(f"round {r} {key:>13s}: " + "  ".join(f"{v}={100 * s:.2f}" for v, s in results.items()))
            if results[v_best] > best_score + 1e-6:
                best, best_score = dataclasses.replace(best, **{key: v_best}), results[v_best]
    print(f"\nbest train-subset mIoU {100 * best_score:.2f}\n{best}")
    save_seg_config(best, args.out)


if __name__ == "__main__":
    main()
