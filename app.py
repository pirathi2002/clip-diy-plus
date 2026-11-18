"""Interactive demo for the presentation: upload an image, type class names, compare CLIP-DIY vs CLIP-DIY+.

    python app.py   # then open http://127.0.0.1:7860
"""
import gradio as gr
import numpy as np

from clipdiy.config import load_seg_config
from clipdiy.pipeline import CLIPDIYPipeline, overlay

ORIGINAL = load_seg_config("configs/original.yaml")
PLUS = load_seg_config("configs/clipdiy_plus.yaml")
pipe = CLIPDIYPipeline(PLUS)


def run(image: np.ndarray, classes: str):
    names = [c.strip() for c in classes.split(",") if c.strip()]
    if image is None or not names:
        raise gr.Error("Upload an image and enter at least one class name.")
    outs = []
    for cfg in (ORIGINAL, PLUS):
        pipe.cfg = cfg
        labels, sal = pipe(image, names)
        outs.append(overlay(image, labels))
    legend = ", ".join(f"{i + 1}: {n}" for i, n in enumerate(names))
    return (sal * 255).astype(np.uint8), outs[0], outs[1], legend


demo = gr.Interface(
    fn=run,
    inputs=[gr.Image(type="numpy", label="image"), gr.Textbox(value="dog, cat, sofa", label="classes (comma separated)")],
    outputs=[gr.Image(label="FOUND objectness"), gr.Image(label="CLIP-DIY (official)"),
             gr.Image(label="CLIP-DIY+ (ours)"), gr.Textbox(label="labels")],
    examples=[["third_party/clip-diy/images/diy-5090797_640.jpg", "scrabble, pencil, rubber stamp, paper clip"]],
    title="CLIP-DIY+: training-free open-vocabulary segmentation",
)

if __name__ == "__main__":
    demo.launch()
