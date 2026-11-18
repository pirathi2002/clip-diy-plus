import glob
import json
import os
import pickle
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import ConfusionMatrix
from .segment import SegConfig, TextBank, build_text_bank, segment


def load_cache(cache_dir: str, limit: Optional[int] = None):
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(cache_dir, "gt.pkl"), "rb") as f:
        gts = pickle.load(f)
    ids = meta["ids"][:limit] if limit else meta["ids"]
    items = [torch.load(os.path.join(cache_dir, "feats", f"{i}.pt")) for i in tqdm(ids, desc="load cache", leave=False)]
    return meta, ids, items, [gts[i] for i in ids]


@torch.no_grad()
def predict(item: dict, bank: TextBank, cfg: SegConfig, out_hw) -> torch.Tensor:
    """Label map at resolution out_hw."""
    scores = segment(item, bank, cfg)
    if cfg.upsample == "label":
        lab = scores.argmax(0)[None, None].float()
        return F.interpolate(lab, size=out_hw, mode="nearest")[0, 0].long()
    scores = torch.nan_to_num(scores, neginf=-1e4)
    return F.interpolate(scores[None], size=out_hw, mode="bilinear", align_corners=False)[0].argmax(0)


class BankCache:
    """Text banks keyed by the vocabulary-related config fields (encoding 80 templates is the slow part)."""

    def __init__(self, clip, classes, synonyms=None):
        self.clip, self.classes, self.synonyms, self._banks = clip, classes, synonyms, {}

    def get(self, cfg: SegConfig) -> TextBank:
        key = (cfg.prompts, cfg.synonyms, cfg.bg_prompts)
        if key not in self._banks:
            self._banks[key] = build_text_bank(self.clip, self.classes, cfg, self.synonyms)
        return self._banks[key]


def evaluate(items: List[dict], gts: List[np.ndarray], bank: TextBank, cfg: SegConfig, progress=False) -> ConfusionMatrix:
    cm = ConfusionMatrix(bank.n_classes)
    for item, gt in tqdm(list(zip(items, gts)), disable=not progress, leave=False):
        cm.update(predict(item, bank, cfg, gt.shape).cpu().numpy(), gt)
    return cm
