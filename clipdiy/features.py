"""Image-side feature extraction. Everything here is independent of the text vocabulary, so it is
computed once per image and cached; every ablation then only re-runs the cheap text-side scoring."""
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .clip_dense import DenseCLIP
from .found import FOUND


@dataclass
class ExtractConfig:
    size: int = 256                          # working resolution (image is resized to size x size, as in the paper)
    windows: Sequence[int] = (256, 128, 64)  # sliding-window sizes ("scales")
    stride_div: int = 2                      # window step = window // stride_div
    dino_grid: int = 32                      # DINO keys are pooled to dino_grid x dino_grid for refinement


def to_tensor(img: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(img).to(device).permute(2, 0, 1).float().div_(255)


@torch.no_grad()
def extract(img: np.ndarray, clip: DenseCLIP, found: FOUND, cfg: ExtractConfig) -> dict:
    """img: HxWx3 uint8 RGB. Returns a dict of fp16 CPU tensors."""
    H, W = img.shape[:2]
    # cv2 INTER_LINEAR is what albumentations.Resize uses in the official pipeline
    small = to_tensor(cv2.resize(img, (cfg.size, cfg.size), interpolation=cv2.INTER_LINEAR), clip.device)
    out = {"hw": (H, W), "clip": {}, "global": {}}
    for view, x in (("orig", small), ("flip", small.flip(-1))):
        out["clip"][view] = {p: e.half().cpu() for p, e in clip.encode_windows(x, cfg.windows, cfg.stride_div).items()}
        out["global"][view] = clip.encode_image(x)[0].half().cpu()

    # FOUND runs on the full-resolution image (official extract_saliency.py), then is resized like the image
    sal, keys = found(to_tensor(img, found.device))
    sal = cv2.resize(sal.float().cpu().numpy(), (cfg.size, cfg.size), interpolation=cv2.INTER_LINEAR)
    out["sal"] = torch.from_numpy(sal).half()
    out["dino"] = F.adaptive_avg_pool2d(keys[None], cfg.dino_grid)[0].half().cpu()
    return out
