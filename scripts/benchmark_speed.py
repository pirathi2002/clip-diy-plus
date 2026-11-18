"""Speed + agreement of the official ClipConv (PIL crops, CPU image processor, fp32) vs our DenseCLIP.

    python scripts/benchmark_speed.py --root data --n 20
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import cv2
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "clip-diy"))
from src.clip_conv import ClipConv  # noqa: E402  (official implementation)

from clipdiy.clip_dense import DenseCLIP  # noqa: E402
from clipdiy.data import VOCSegmentation  # noqa: E402
from clipdiy.features import to_tensor  # noqa: E402
from clipdiy.prompts import VOC_CLASSES, original_prompt  # noqa: E402

MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


def timed(fn, images):
    torch.cuda.synchronize()
    t0 = time.time()
    outs = [fn(x) for x in images]
    torch.cuda.synchronize()
    return outs, (time.time() - t0) / len(images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    ds = VOCSegmentation(args.root, "val")
    imgs = [cv2.resize(ds.image(i), (256, 256), interpolation=cv2.INTER_LINEAR) for i in range(args.n)]

    cfg = SimpleNamespace(clip_model=MODEL, patch_sizes=[256, 128, 64], strides=[2], interpolate_logits=True,
                          img_size=[256, 256], align_corners=False, patch_w1=1.0, patch_w2=1.0, patch_w3=1.0)
    official = ClipConv(cfg, VOC_CLASSES).cuda()
    off_maps, t_off = timed(lambda x: official(torch.from_numpy(x).permute(2, 0, 1)[None].cuda())[0], imgs)
    del official
    torch.cuda.empty_cache()

    results = {}
    for fp16 in (False, True):
        clip = DenseCLIP(MODEL, fp16=fp16)
        text = clip.encode_text([original_prompt(c) for c in VOC_CLASSES])

        def ours(x):
            maps = []
            for p, e in clip.encode_windows(to_tensor(x, "cuda"), (256, 128, 64)).items():
                m = (e @ text.T).permute(2, 0, 1)[None]
                maps.append(F.interpolate(m, size=(256, 256), mode="bilinear", align_corners=False)[0])
            return torch.stack(maps).mean(0)
        ours(imgs[0])  # warm-up
        maps, t = timed(ours, imgs)
        agree = torch.stack([(a.argmax(0) == b.argmax(0)).float().mean() for a, b in zip(off_maps, maps)]).mean()
        err = torch.stack([(a - b).abs().max() for a, b in zip(off_maps, maps)]).max()
        results["fp16" if fp16 else "fp32"] = (t, agree.item(), err.item())
        del clip
        torch.cuda.empty_cache()

    print(f"official ClipConv     : {1000 * t_off:7.1f} ms/img")
    for k, (t, agree, err) in results.items():
        print(f"ours DenseCLIP ({k}) : {1000 * t:7.1f} ms/img  speed-up x{t_off / t:4.1f} | "
              f"argmax agreement {100 * agree:.2f}% | max |cos diff| {err:.4f}")


if __name__ == "__main__":
    main()
