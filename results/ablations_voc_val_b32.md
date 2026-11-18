| group          | experiment                                          |   mIoU |   delta |   pixel_acc |
|:---------------|:----------------------------------------------------|-------:|--------:|------------:|
| reproduction   | CLIP-DIY (official settings)                        |  59.82 |    0    |       88.82 |
| paper ablation | w/o objectness (FOUND)                              |  24.4  |  -35.42 |       37.32 |
| paper ablation | single scale 256px                                  |  56.28 |   -3.53 |       87.61 |
| paper ablation | single scale 128px                                  |  57.23 |   -2.59 |       88.28 |
| paper ablation | single scale 64px                                   |  45.22 |  -14.6  |       85.49 |
| cumulative     | + fix prompt article ('an aeroplane')               |  60.05 |    0.24 |       88.85 |
| cumulative     | + probabilistic objectness fusion                   |  61.5  |    1.68 |       89.26 |
| cumulative     | + background-concept prompts                        |  62.35 |    2.54 |       89.75 |
| cumulative     | + DINO refinement of class probabilities            |  62.93 |    3.12 |       89.93 |
| cumulative     | + DINO propagation of objectness                    |  64.77 |    4.95 |       90.78 |
| cumulative     | + bilinear score upsampling                         |  64.8  |    4.98 |       90.79 |
| final          | CLIP-DIY+ (full)                                    |  64.8  |    4.98 |       90.79 |
| leave-one-out  | CLIP-DIY+ w/o probabilistic fusion                  |  60.1  |    0.28 |       88.87 |
| leave-one-out  | CLIP-DIY+ w/o background concepts                   |  64.26 |    4.45 |       90.45 |
| leave-one-out  | CLIP-DIY+ w/o DINO class refinement                 |  64.19 |    4.37 |       90.58 |
| leave-one-out  | CLIP-DIY+ w/o DINO objectness propagation           |  62.95 |    3.14 |       89.94 |
| leave-one-out  | CLIP-DIY+ w/o objectness (FOUND)                    |  26.15 |  -33.67 |       41.48 |
| rejected       | CLIP-DIY+ + 80-template prompt ensemble             |  63.49 |    3.67 |       90.4  |
| rejected       | CLIP-DIY+ + class-name synonyms                     |  64.91 |    5.09 |       90.76 |
| rejected       | CLIP-DIY+ + flip TTA                                |  64.97 |    5.16 |       90.85 |
| rejected       | CLIP-DIY+ + class filter (whole image, p>=0.05)     |  63.19 |    3.37 |       90.23 |
| rejected       | CLIP-DIY+ + class filter (max over windows, p>=0.5) |  64.78 |    4.97 |       90.78 |