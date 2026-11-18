import dataclasses
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clipdiy.metrics import ConfusionMatrix  # noqa: E402
from clipdiy.prompts import fixed_prompt, original_prompt  # noqa: E402
from clipdiy.segment import SegConfig, TextBank, _per_class, segment  # noqa: E402

D, SIZE = 8, 32


def make_item(seed=0, sal=None):
    g = torch.Generator().manual_seed(seed)

    def rnd(*s):
        return F.normalize(torch.randn(*s, D, generator=g), dim=-1)
    views = {v: {32: rnd(1, 1), 16: rnd(3, 3), 8: rnd(7, 7)} for v in ("orig", "flip")}
    return {"clip": views, "global": {v: rnd(1)[0] for v in views},
            "sal": torch.rand(SIZE, SIZE, generator=g) if sal is None else sal,
            "dino": torch.randn(16, 4, 4, generator=g)}


def make_bank(n_classes=4, seed=1, names_per_class=None):
    g = torch.Generator().manual_seed(seed)
    cls = names_per_class or list(range(n_classes))
    return TextBank(F.normalize(torch.randn(len(cls), D, generator=g), dim=-1), torch.tensor(cls), n_classes)


CFG = SegConfig(scales=(32, 16, 8))


def test_original_fusion_matches_official_formula():
    item, bank = make_item(), make_bank()
    scores = segment(item, bank, CFG)
    maps = []
    for p in CFG.scales:  # official ClipConv.forward + objectness fusion in eval_clipconv.py
        cos = (item["clip"]["orig"][p] @ bank.emb.T).permute(2, 0, 1)[None]
        maps.append(F.interpolate(cos, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0])
    out = torch.stack(maps).mean(0)
    s = item["sal"][None]
    official = (out * torch.cat([1 - s, s.expand(bank.n_classes - 1, -1, -1)])).softmax(0)
    assert torch.equal(scores.argmax(0), official.argmax(0))


def test_synonyms_use_max():
    bank = make_bank(n_classes=3, names_per_class=[0, 1, 1, 2])
    cos = torch.tensor([[0.1, 0.2, 0.5, 0.3]])
    assert torch.allclose(_per_class(cos, bank), torch.tensor([[0.1, 0.5, 0.3]]))


def test_prob_fusion_is_background_where_not_salient():
    item = make_item(sal=torch.zeros(SIZE, SIZE))
    scores = segment(item, make_bank(), dataclasses.replace(CFG, fusion="prob"))
    assert (scores.argmax(0) == 0).all()


def test_prob_fusion_without_background_text_never_predicts_bg_when_fully_salient():
    item = make_item(sal=torch.ones(SIZE, SIZE))
    bank = make_bank(n_classes=4, names_per_class=[1, 2, 3])
    scores = segment(item, bank, dataclasses.replace(CFG, fusion="prob"))
    assert (scores.argmax(0) != 0).all()
    assert torch.allclose(scores.sum(0), torch.ones(SIZE, SIZE), atol=1e-5)


def test_swapping_views_mirrors_the_prediction():
    item = make_item()
    swapped = dict(item, clip={"orig": item["clip"]["flip"], "flip": item["clip"]["orig"]})
    cfg = dataclasses.replace(CFG, flip=True, saliency=False)
    a, b = segment(item, make_bank(), cfg), segment(swapped, make_bank(), cfg)
    assert torch.allclose(a, b.flip(-1), atol=1e-6)


@pytest.mark.parametrize("thr", [0.0, 0.5, 0.99])
def test_class_filter_keeps_at_least_top1(thr):
    cfg = dataclasses.replace(CFG, fusion="prob", class_filter=thr)
    bank = make_bank(n_classes=4, names_per_class=[1, 2, 3])
    labels = segment(make_item(sal=torch.ones(SIZE, SIZE)), bank, cfg).argmax(0)
    assert labels.unique().numel() >= 1 and (labels > 0).all()
    if thr == 0.99:
        assert labels.unique().numel() == 1


def test_refinement_outputs_distribution():
    cfg = dataclasses.replace(CFG, fusion="prob", refine=True, refine_knn=4, saliency=False)
    p = segment(make_item(), make_bank(), cfg)
    assert torch.allclose(p.sum(0), torch.ones(SIZE, SIZE), atol=1e-5) and (p >= 0).all()


def test_confusion_matrix():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 1, 1, 2, 2]), np.array([0, 1, 2, 2, 255]))
    assert np.allclose(cm.iou(), [1.0, 0.5, 0.5])
    assert cm.pixel_acc() == 0.75


def test_prompts():
    assert original_prompt("aeroplane") == "a photo of a aeroplane"  # upstream bug, kept for reproduction
    assert fixed_prompt("aeroplane") == "a photo of an aeroplane."
    assert fixed_prompt("dog") == "a photo of a dog."
