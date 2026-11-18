# Final presentation: slide outline (10 min + code review)

Figures are in `results/figures/`. Numbers are from `results/ablations_*.md`.
Suggested split: Pirathishanth slides 1–5, Dinuja slides 6–10 (swap as you like).

| # | Slide | Content | Time |
|---|-------|---------|------|
| 1 | Title | CLIP-DIY+ — reproducing and improving CLIP-DIY (WACV 2024), group, module | 0:20 |
| 2 | Recap in one picture | Sliding-window CLIP at 256/128/64 px → averaged maps → FOUND objectness → label map. "No training, any vocabulary." (reuse the pipeline slide from the first talk) | 0:50 |
| 3 | What we built | Clean re-implementation (`clipdiy/`, ~730 lines, 11 unit tests); features cached once so every experiment takes ~10 s; runs on a 4 GB laptop GPU | 0:50 |
| 4 | Reproduction | Table: VOC 59.8 (paper 59.9); w/o objectness 24.4 (24.1); single scale 56.3 (56.0); COCO subset 32.3 (31.0 full). Matches the official ClipConv on 98% of pixels | 1:00 |
| 5 | Bugs found in official code | prompt article bug, unused logit scale (fusion = fixed threshold), config `$` typo, undefined `cfg.results`, missing `errno`, COCO filter name mismatch | 0:50 |
| 6 | Error analysis → idea | On train images: ~60% of errors = object pixels FOUND calls background, ~25% false objects, ~12% wrong class → focus on objectness, not on prompts | 1:00 |
| 7 | Our improvements | (a) probabilistic fusion with CLIP temperature and λ; (b) background-concept prompts; (c) DINO objectness propagation (random walk of saliency on DINO-key graph, residual); (d) DINO class refinement; all training-free, DINO keys are free from FOUND | 1:30 |
| 8 | Results | `ablation_voc.png` bar chart. VOC 59.8 → **64.8 (+5.0)**, COCO 32.3 → **35.3 (+3.0)** with VOC-tuned settings. Tuned only on 300 train images | 1:10 |
| 9 | Qualitative + honest negatives | `qualitative_voc.png`: hallucinations removed; failure: bottle on grass / people in front of trees swallowed by background concepts. Rejected ideas: prompt ensemble (−1.3), class filter (−1.6), synonyms/flip ≈0 | 1:10 |
| 10 | Takeaways + future work | Objectness is the bottleneck; free DINO features fix part of it. Next: ViT-B/16 backbone, full COCO, adaptive background concepts. Live demo: `python app.py` | 1:00 |

## Live demo plan (30 s, optional)
1. Before the talk: `source .venv/bin/activate && python app.py`, open http://127.0.0.1:7860.
2. Click the DIY example → shows FOUND map, CLIP-DIY and CLIP-DIY+ side by side.
3. Upload a photo taken in the room and type 3 class names (e.g. "laptop, chair, bottle").
Fallback if the laptop struggles: show `results/figures/demo_diy.png`.

## Code review checklist (night before)
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest` passes.
- `python scripts/evaluate.py --cache cache/voc_val_b32 --config configs/clipdiy_plus.yaml` prints 64.80.
- Both of you can explain `segment()` in `clipdiy/segment.py` line by line (see `docs/code_review_guide.md`).
- Fill in the "Use of AI coding assistance" section in `docs/code_review_guide.md`.
