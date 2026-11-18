| group          | experiment                                          |   mIoU |   delta |   pixel_acc |
|:---------------|:----------------------------------------------------|-------:|--------:|------------:|
| reproduction   | CLIP-DIY (official settings)                        |  32.31 |    0    |       77.61 |
| paper ablation | w/o objectness (FOUND)                              |  16.61 |  -15.7  |       28.9  |
| paper ablation | single scale 256px                                  |  27.84 |   -4.47 |       75.12 |
| paper ablation | single scale 128px                                  |  30.46 |   -1.85 |       77.16 |
| paper ablation | single scale 64px                                   |  23.13 |   -9.17 |       75.75 |
| cumulative     | + fix prompt article ('an aeroplane')               |  32.51 |    0.2  |       77.52 |
| cumulative     | + probabilistic objectness fusion                   |  33.08 |    0.77 |       77.18 |
| cumulative     | + background-concept prompts                        |  33.81 |    1.5  |       77.9  |
| cumulative     | + DINO refinement of class probabilities            |  34.19 |    1.88 |       78.16 |
| cumulative     | + DINO propagation of objectness                    |  35.27 |    2.96 |       79.07 |
| cumulative     | + bilinear score upsampling                         |  35.33 |    3.02 |       79.1  |
| final          | CLIP-DIY+ (full)                                    |  35.33 |    3.02 |       79.1  |
| leave-one-out  | CLIP-DIY+ w/o probabilistic fusion                  |  32.55 |    0.24 |       77.55 |
| leave-one-out  | CLIP-DIY+ w/o background concepts                   |  34.78 |    2.47 |       78.51 |
| leave-one-out  | CLIP-DIY+ w/o DINO class refinement                 |  34.81 |    2.5  |       78.74 |
| leave-one-out  | CLIP-DIY+ w/o DINO objectness propagation           |  34.22 |    1.92 |       78.16 |
| leave-one-out  | CLIP-DIY+ w/o objectness (FOUND)                    |  17.8  |  -14.5  |       34.67 |
| rejected       | CLIP-DIY+ + 80-template prompt ensemble             |  34.62 |    2.31 |       78.01 |
| rejected       | CLIP-DIY+ + class-name synonyms                     |  35.93 |    3.62 |       79.48 |
| rejected       | CLIP-DIY+ + flip TTA                                |  35.49 |    3.18 |       79.14 |
| rejected       | CLIP-DIY+ + class filter (whole image, p>=0.05)     |  32.84 |    0.54 |       77.5  |
| rejected       | CLIP-DIY+ + class filter (max over windows, p>=0.5) |  35.3  |    2.99 |       79.08 |