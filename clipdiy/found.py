"""FOUND unsupervised saliency (Simeoni et al., CVPR 2023), re-wrapped without its training code.

FOUND = frozen DINO ViT-S/8 + a 1x1 conv on the last-layer attention keys. We also return those keys,
which are reused (for free) as a pixel-affinity source by the refinement step.
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DINO_DIR = os.path.join(ROOT, "third_party", "dino")
FOUND_WEIGHTS = os.path.join(ROOT, "third_party", "FOUND", "data", "weights", "decoder_weights.pt")
DINO_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_dino_module():
    # dino's vision_transformer.py does `from utils import trunc_normal_`; make that resolve to dino/utils.py
    sys.path.insert(0, DINO_DIR)
    try:
        spec = importlib.util.spec_from_file_location("dino_vit", os.path.join(DINO_DIR, "vision_transformer.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(DINO_DIR)
    return module


class FOUND(nn.Module):
    patch_size = 8

    def __init__(self, device: str = "cuda", weights: str = FOUND_WEIGHTS):
        super().__init__()
        vits = _load_dino_module()
        self.vit = vits.vit_small(patch_size=self.patch_size, num_classes=0)
        self.vit.load_state_dict(torch.hub.load_state_dict_from_url(DINO_URL, map_location="cpu"), strict=True)
        self.decoder = nn.Conv2d(384, 1, 1)
        self.decoder.load_state_dict(torch.load(weights, map_location="cpu")["decoder"])
        self._qkv = None
        self.vit.blocks[-1].attn.qkv.register_forward_hook(lambda m, i, o: setattr(self, "_qkv", o))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.to(device).eval()
        self.device = device

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        """image: (3, H, W) float in [0, 1].

        Returns saliency (H, W) in [0, 1] and DINO last-layer keys (384, ceil(H/8), ceil(W/8)).
        """
        _, H, W = image.shape
        x = ((image[None].to(self.device) - self.mean) / self.std)
        pad_h, pad_w = (-H) % self.patch_size, (-W) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0)  # same zero-padding as FOUND's make_input_divisible
        hf, wf = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        self.vit.get_last_selfattention(x)
        n_tok = self._qkv.shape[1]
        heads = self.vit.blocks[-1].attn.num_heads
        k = self._qkv.reshape(1, n_tok, 3, heads, -1)[:, 1:, 1]  # drop CLS, take keys -> 1, hf*wf, heads, 64
        keys = k.reshape(1, hf, wf, -1).permute(0, 3, 1, 2)  # 1, 384, hf, wf
        logits = F.interpolate(self.decoder(keys), scale_factor=self.patch_size, mode="bilinear",
                               align_corners=False)[..., :H, :W]
        return torch.sigmoid(logits)[0, 0], keys[0]
