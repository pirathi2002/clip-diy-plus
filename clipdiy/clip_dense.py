"""Dense CLIP inference by classifying sliding windows (Sec. 3.1 of CLIP-DIY).

Unlike the official implementation, which converts every crop to a PIL image and runs the HuggingFace
image processor on CPU, all windows are cropped, resized and normalised on the GPU in one batch.
Image and text sides are kept separate (CLIP is a dual encoder), so window embeddings can be cached once
and re-scored against any vocabulary.
"""
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class DenseCLIP(torch.nn.Module):
    # fp16 is off by default: on GPUs without tensor cores (e.g. GTX 1650) it is ~4x slower than fp32
    def __init__(self, model_name: str, device: str = "cuda", fp16: bool = False, chunk: int = 128):
        super().__init__()
        dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
        self.model = CLIPModel.from_pretrained(model_name, dtype=dtype).to(device).eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.input_size = self.model.config.vision_config.image_size
        self.device, self.dtype, self.chunk = device, dtype, chunk
        self.register_buffer("mean", torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """(N, D) L2-normalised float32 text embeddings."""
        out = []
        for i in range(0, len(prompts), 256):
            tok = self.tokenizer(prompts[i:i + 256], padding=True, return_tensors="pt").to(self.device)
            out.append(self.model.get_text_features(**tok).float())
        return F.normalize(torch.cat(out), dim=-1)

    @torch.no_grad()
    def encode_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """crops: (N, 3, h, w) float in [0, 1] -> (N, D) L2-normalised float32 image embeddings."""
        out = []
        for i in range(0, crops.shape[0], self.chunk):
            x = crops[i:i + self.chunk]
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic",
                              align_corners=False, antialias=x.shape[-1] > self.input_size).clamp_(0, 1)
            x = ((x - self.mean) / self.std).to(self.dtype)
            out.append(self.model.get_image_features(pixel_values=x).float())
        return F.normalize(torch.cat(out), dim=-1)

    @torch.no_grad()
    def encode_windows(self, image: torch.Tensor, windows: Sequence[int], stride_div: int = 2) -> Dict[int, torch.Tensor]:
        """Embed every sliding window of every size.

        image: (3, H, W) float in [0, 1] on device.
        Returns {window: (nh, nw, D)}. A window of size p moves with step p // stride_div, as in the paper.
        """
        _, H, W = image.shape
        grids, crops = [], []
        for p in windows:
            if p > min(H, W):
                raise ValueError(f"window {p} larger than image {H}x{W}")
            step = max(p // stride_div, 1)
            patches = image.unfold(1, p, step).unfold(2, p, step)  # 3, nh, nw, p, p
            nh, nw = patches.shape[1:3]
            grids.append((p, nh, nw))
            crops.append(patches.permute(1, 2, 0, 3, 4).reshape(nh * nw, 3, p, p))
        # crops of different sizes are encoded separately but in one call per size
        out = {}
        for (p, nh, nw), c in zip(grids, crops):
            out[p] = self.encode_crops(c).view(nh, nw, -1)
        return out

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Whole-image embedding (1, D), used for image-level class filtering."""
        return self.encode_crops(image[None])
