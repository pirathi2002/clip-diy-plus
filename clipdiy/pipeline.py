"""End-to-end open-vocabulary segmentation of a single image with free-text class names."""
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .clip_dense import DenseCLIP
from .features import ExtractConfig, extract
from .found import FOUND
from .segment import SegConfig, build_text_bank, segment

PALETTE = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
                    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
                    [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]], np.uint8)


class CLIPDIYPipeline:
    def __init__(self, cfg: SegConfig, clip_model: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                 device: str = "cuda", extract_cfg: Optional[ExtractConfig] = None):
        self.cfg = cfg
        self.clip = DenseCLIP(clip_model, device)
        self.found = FOUND(device)
        self.extract_cfg = extract_cfg or ExtractConfig()

    @torch.no_grad()
    def __call__(self, img: np.ndarray, class_names: List[str]):
        """img: HxWx3 uint8 RGB. class_names: foreground names. Returns (labels HxW, saliency HxW); label 0 =
        background, label i = class_names[i-1]."""
        item = extract(img, self.clip, self.found, self.extract_cfg)
        bank = build_text_bank(self.clip, ["background"] + list(class_names), self.cfg)
        scores = torch.nan_to_num(segment(item, bank, self.cfg), neginf=-1e4)
        H, W = img.shape[:2]
        labels = F.interpolate(scores[None], size=(H, W), mode="bilinear", align_corners=False)[0].argmax(0)
        sal = F.interpolate(item["sal"].float()[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        return labels.cpu().numpy().astype(np.uint8), sal.numpy()


def overlay(img: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    color = PALETTE[labels % len(PALETTE)]
    out = img.astype(np.float32)
    fg = labels > 0
    out[fg] = (1 - alpha) * out[fg] + alpha * color[fg]
    return out.astype(np.uint8)
