"""Text-side scoring, objectness fusion and refinement.

Everything that depends on the vocabulary or on fusion hyper-parameters lives here and runs on the
cached image features produced by `features.extract`, so one image costs a few milliseconds.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .prompts import BACKGROUND_CONCEPTS, TEMPLATES, fixed_prompt, original_prompt


@dataclass
class SegConfig:
    # vocabulary
    prompts: str = "original"        # original (official, article bug) | fixed | ensemble (80 templates)
    synonyms: bool = False           # several natural names per class, scored with max
    bg_prompts: str = "background"   # background (single "background" text) | concepts (sky, wall, ...) | none
    # dense CLIP
    scales: Sequence[int] = (256, 128, 64)
    flip: bool = False               # horizontal-flip test-time augmentation
    # objectness fusion
    saliency: bool = True
    fusion: str = "original"         # original: argmax(cos * [1-s, s, ..., s]) | prob: see `fuse`
    temperature: float = 100.0       # CLIP logit scale used for the class softmax (prob fusion / class filter)
    bg_weight: float = 1.0           # weight of (1 - saliency) for background (prob fusion)
    class_filter: float = 0.0        # drop classes whose image-level probability is below this (0 = off)
    filter_source: str = "global"    # global: whole-image CLIP | windows: max over all sliding windows
    # DINO-affinity refinement (prob fusion only); the graph parameters are shared by both refinements
    sal_refine: float = 0.0          # strength of objectness propagation on the DINO graph (0 = off)
    refine: bool = False             # propagate class probabilities on the DINO graph
    refine_knn: int = 32
    refine_tau: float = 0.1
    refine_iters: int = 2
    # evaluation
    upsample: str = "label"          # label: nearest on the argmax (official) | score: bilinear on scores


@dataclass
class TextBank:
    emb: torch.Tensor          # (N, D) one row per name
    cls: torch.Tensor          # (N,) class index of each name
    n_classes: int
    names: List[str] = field(default_factory=list)

    @property
    def valid(self) -> torch.Tensor:
        v = torch.zeros(self.n_classes, dtype=torch.bool, device=self.cls.device)
        v[self.cls] = True
        return v


def build_text_bank(clip, class_names: List[str], cfg: SegConfig,
                    synonyms: Optional[Dict[str, List[str]]] = None) -> TextBank:
    """class_names[0] must be the background class."""
    def prompts_for(name):
        if cfg.prompts == "original":
            return [original_prompt(name)]
        if cfg.prompts == "fixed":
            return [fixed_prompt(name)]
        if cfg.prompts == "ensemble":
            return [t.format(name) for t in TEMPLATES]
        raise ValueError(cfg.prompts)

    entries = []
    if cfg.bg_prompts == "background":
        entries.append((class_names[0], 0))
    elif cfg.bg_prompts == "concepts":
        entries += [(n, 0) for n in BACKGROUND_CONCEPTS]
    elif cfg.bg_prompts != "none":
        raise ValueError(cfg.bg_prompts)
    for c, name in enumerate(class_names[1:], 1):
        names = synonyms.get(name, [name]) if (cfg.synonyms and synonyms) else [name]
        entries += [(n, c) for n in names]

    embs = [F.normalize(clip.encode_text(prompts_for(n)).mean(0), dim=-1) for n, _ in entries]
    cls = torch.tensor([c for _, c in entries], device=clip.device)
    return TextBank(torch.stack(embs), cls, len(class_names), [n for n, _ in entries])


def _per_class(cos: torch.Tensor, bank: TextBank) -> torch.Tensor:
    """cos (..., N) over names -> (..., C) over classes, max over synonyms. Classes without names get 0
    and are masked out later with `bank.valid`."""
    out = cos.new_zeros(*cos.shape[:-1], bank.n_classes)
    idx = bank.cls.expand_as(cos)
    return out.scatter_reduce(-1, idx, cos, reduce="amax", include_self=False)


def clip_maps(item: dict, bank: TextBank, cfg: SegConfig, size: int) -> torch.Tensor:
    """Multi-scale patch-text cosine maps averaged over scales (and flips): (C, size, size)."""
    dev = bank.emb.device
    maps = []
    for view in (("orig", "flip") if cfg.flip else ("orig",)):
        for p in cfg.scales:
            cos = _per_class(item["clip"][view][p].to(dev).float() @ bank.emb.T, bank)  # nh, nw, C
            m = cos.permute(2, 0, 1)[None]
            if view == "flip":
                m = m.flip(-1)
            maps.append(F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)[0])
    return torch.stack(maps).mean(0)


def image_level_probs(item: dict, bank: TextBank, cfg: SegConfig) -> torch.Tensor:
    """Image-level probabilities of the foreground classes: (C-1,).

    global: softmax of the whole-image embedding. windows: per-class max of the window softmaxes over all
    scales, which keeps small objects that the whole-image view misses."""
    dev = bank.emb.device
    views = ("orig", "flip") if cfg.flip else ("orig",)
    if cfg.filter_source == "global":
        cos = torch.stack([_per_class(item["global"][v].to(dev).float() @ bank.emb.T, bank) for v in views]).mean(0)
        return (cfg.temperature * cos[1:]).softmax(-1)
    if cfg.filter_source != "windows":
        raise ValueError(cfg.filter_source)
    emb = torch.cat([item["clip"][v][p].to(dev).float().flatten(0, 1) for v in views for p in cfg.scales])
    probs = (cfg.temperature * _per_class(emb @ bank.emb.T, bank)[:, 1:]).softmax(-1)
    return probs.max(0).values


def dino_affinity(dino: torch.Tensor, cfg: SegConfig) -> torch.Tensor:
    """(N, N) k-NN graph on DINO keys (N = grid*grid), edge weight exp((cos - 1) / tau), not normalised."""
    f = F.normalize(dino.float().flatten(1), dim=0)  # D, N
    sim = f.T @ f
    val, idx = sim.topk(min(cfg.refine_knn, sim.shape[0]), dim=1)
    return torch.zeros_like(sim).scatter_(1, idx, torch.exp((val - 1) / cfg.refine_tau))


def _random_walk(x: torch.Tensor, A: torch.Tensor, iters: int) -> torch.Tensor:
    """x (C, N); A (N, N) row-normalised."""
    for _ in range(iters):
        x = x @ A.T
    return x


def refine_probs(probs: torch.Tensor, A: torch.Tensor, sal: torch.Tensor, cfg: SegConfig) -> torch.Tensor:
    """Random-walk propagation of class probabilities on the DINO graph.

    CLIP window scores are blurry and leak across neighbouring objects; DINO keys group pixels of the
    same object. Only salient nodes send class evidence, so background clutter does not dilute labels.
    """
    C, H, W = probs.shape
    g = int(A.shape[0] ** 0.5)
    s = F.adaptive_avg_pool2d(sal[None, None], g).flatten()
    A = A * (s[None, :] + 1e-3)
    A = A / A.sum(1, keepdim=True)
    p = _random_walk(F.adaptive_avg_pool2d(probs[None], g)[0].flatten(1), A, cfg.refine_iters)
    p = F.interpolate(p.view(1, C, g, g), size=(H, W), mode="bilinear", align_corners=False)[0]
    return p / p.sum(0, keepdim=True).clamp_min(1e-8)


def refine_saliency(sal: torch.Tensor, A: torch.Tensor, cfg: SegConfig) -> torch.Tensor:
    """Spread FOUND objectness to DINO-similar pixels, which recovers object parts FOUND misses.

    The low-resolution change is added as a residual, so FOUND's sharp full-resolution boundaries are kept."""
    H, W = sal.shape
    g = int(A.shape[0] ** 0.5)
    s0 = F.adaptive_avg_pool2d(sal[None, None], g).flatten(1)  # 1, N
    s1 = _random_walk(s0, A / A.sum(1, keepdim=True), cfg.refine_iters)
    delta = F.interpolate((s1 - s0).view(1, 1, g, g), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return (sal + cfg.sal_refine * delta).clamp(0, 1)


def segment(item: dict, bank: TextBank, cfg: SegConfig) -> torch.Tensor:
    """Per-class scores (C, size, size) at the working resolution; argmax gives the label map."""
    dev = bank.emb.device
    sal = item["sal"].to(dev).float()
    size = sal.shape[-1]
    cos = clip_maps(item, bank, cfg, size)
    invalid = ~bank.valid
    if cfg.class_filter > 0:
        g = image_level_probs(item, bank, cfg)
        keep = g >= cfg.class_filter
        keep[g.argmax()] = True
        invalid = invalid.clone()
        invalid[1:] |= ~keep

    if cfg.fusion == "original":
        if cfg.saliency:
            mask = torch.cat([1 - sal[None], sal[None].expand(bank.n_classes - 1, -1, -1)])
            cos = cos * mask
        return cos.masked_fill(invalid[:, None, None], float("-inf"))

    if cfg.fusion != "prob":
        raise ValueError(cfg.fusion)
    probs = (cfg.temperature * cos).masked_fill(invalid[:, None, None], float("-inf")).softmax(0)
    if cfg.refine or cfg.sal_refine > 0:
        A = dino_affinity(item["dino"].to(dev), cfg)
        if cfg.sal_refine > 0:
            sal = refine_saliency(sal, A, cfg)
        if cfg.refine:
            probs = refine_probs(probs, A, sal, cfg)
    if not cfg.saliency:
        return probs
    # P(bg) = (1-s) + s * P(bg-text | fg),  P(c) = s * P(c | fg)
    scores = sal[None] * probs
    scores[0] = scores[0] + cfg.bg_weight * (1 - sal)
    return scores
