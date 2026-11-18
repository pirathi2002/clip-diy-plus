# Code review guide (for the examiners and for us)

## Walk-through order (≈5 min)

1. `clipdiy/clip_dense.py` — `encode_windows`: `unfold` cuts 256/128/64-px windows with step = size/2
   (1 + 9 + 49 = 59 windows per image). All windows are resized to 224 and encoded in one batch on the GPU.
2. `clipdiy/found.py` — FOUND = DINO ViT-S/8 + a 1×1 conv on the last-layer **keys**. We hook `qkv` of the last
   block, take the keys, and return them too (they are reused by the refinement).
3. `clipdiy/features.py` — everything that does not depend on the vocabulary is computed once and cached.
4. `clipdiy/segment.py` — `segment()` is the method:
   - `clip_maps`: window-text cosine → per-class max over synonyms → bilinear upsample → mean over scales/flips
   - `fusion="original"`: `argmax(cos · [1−s, s, …, s])` — exactly the official rule (unit-tested)
   - `fusion="prob"`: `P(bg)=λ(1−s)+s·P(bg-text|fg)`, `P(c)=s·P(c|fg)` with a temperature softmax
   - `dino_affinity` / `refine_saliency` / `refine_probs`: random walk on a k-NN graph of DINO keys
5. `scripts/tune.py` — coordinate descent on 300 **train** images; val is only used for the final numbers.
6. `tests/test_segment.py` — run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`.

## Likely questions

**Did you train anything?** No. CLIP, DINO and the FOUND decoder are frozen, released weights. The only
data-driven choice is a handful of scalar hyper-parameters picked on 300 VOC *train* images (no val leakage).
That keeps the "for-free / zero-shot" claim of the paper.

**How do you know your re-implementation is faithful?** (1) `test_original_fusion_matches_official_formula`
checks our fusion against the official formula; (2) `scripts/benchmark_speed.py` runs the official `ClipConv`
class and ours on the same images: 98% identical argmax (the rest is PIL-vs-torch bicubic resize);
(3) with the official settings we get 59.8 mIoU on VOC val vs 59.9 in the paper; the paper's ablations match too
(w/o objectness 24.4 vs 24.1; single scale 56.3 vs 56.0).

**Why cache features?** CLIP is a dual encoder: window image embeddings do not depend on the text. Caching
them (plus saliency and DINO keys) once turns every text/fusion ablation from ~30 min into ~20 s.

**Why is the official fusion "really a threshold"?** It takes `argmax_c m_c · cos_c` with
`m_0 = 1−s`, `m_c = s`. Cosines are all ≈0.2–0.3, so the background wins roughly when
`s < cos_bg / (cos_bg + cos_c) ≈ 0.5`. The softmax after the product does not change the argmax and has no
temperature. Our probabilistic fusion makes the trade-off explicit (λ) and lets CLIP's confidence matter.

**Why does refinement use DINO keys?** FOUND already computes them, so it is free. DINO keys cluster pixels
of the same object well (that is why FOUND works), while CLIP window scores are blurry (64-px windows).

**What did not work?** Prompt ensembling (−1.3 on VOC val when added to the final model) and whole-image class
filtering (−1.6); synonyms and flip TTA ≈0. They were rejected on the train subset, and we did not re-select on val.

**Is the gain just tuning?** The same config, tuned only on VOC train, also gives +3.0 on COCO-Object (80 classes
it never saw during tuning). The leave-one-out table shows each kept component contributes on both datasets.

**What is the largest single gain and why?** DINO objectness propagation (+1.8 VOC, +1.1 COCO). Error analysis
showed ~60% of wrong pixels are object pixels that FOUND calls background; propagating saliency to DINO-similar
pixels recovers missing object parts. It is added as a low-resolution residual so FOUND's sharp edges stay.

**Why fp32?** On the GTX 1650 (no tensor cores) fp16 matmul measured 342 GFLOPS vs 1461 GFLOPS for fp32.

**Limitations?** Objects that FOUND misses completely cannot be recovered (≈60% of remaining wrong pixels
are object pixels predicted as background). Images are squashed to 256×256 as in the paper.

## Use of AI coding assistance

> TODO (group): edit this to describe your actual process honestly, as the assignment asks for
> "judicious AI coding".

Suggested points: which parts were AI-assisted (e.g. scaffolding, tests, plotting), which decisions you made
or verified yourselves, and how every reported number can be re-run from the commands in the README.
