"""Class vocabularies and prompt templates.

`VOC_CLASSES` is the exact list used by the official CLIP-DIY code (needed to reproduce the paper).
`VOC_SYNONYMS` is our cleaned vocabulary: dataset-specific tokens such as "tv/monitor" or "diningtable"
are replaced by natural phrases, and several names can point to the same class.
"""

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor",
]

VOC_SYNONYMS = {
    "aeroplane": ["aeroplane", "airplane"],
    "bicycle": ["bicycle"],
    "bird": ["bird"],
    "boat": ["boat", "ship"],
    "bottle": ["bottle"],
    "bus": ["bus"],
    "car": ["car"],
    "cat": ["cat"],
    "chair": ["chair"],
    "cow": ["cow"],
    "diningtable": ["dining table", "table"],
    "dog": ["dog"],
    "horse": ["horse"],
    "motorbike": ["motorbike", "motorcycle"],
    "person": ["person", "man", "woman", "child"],
    "potted plant": ["potted plant", "houseplant"],
    "sheep": ["sheep"],
    "sofa": ["sofa", "couch"],
    "train": ["train"],
    "tv/monitor": ["television", "computer monitor", "tv screen"],
}

# Only ambiguous or unnatural COCO names are expanded; every other class keeps its own name.
COCO_SYNONYMS = {
    "person": ["person", "man", "woman", "child"],
    "tv": ["television", "tv screen", "computer monitor"],
    "couch": ["couch", "sofa"],
    "dining table": ["dining table", "table"],
    "mouse": ["computer mouse"],
    "remote": ["remote control"],
    "keyboard": ["computer keyboard"],
    "cell phone": ["cell phone", "mobile phone"],
    "sports ball": ["sports ball", "ball"],
    "hair drier": ["hair dryer"],
    "motorcycle": ["motorcycle", "motorbike"],
    "airplane": ["airplane", "aeroplane"],
    "boat": ["boat", "ship"],
}

SYNONYMS = {"voc": VOC_SYNONYMS, "coco": COCO_SYNONYMS}

# Concepts that are never a foreground class in VOC/COCO-Object; mapped to label 0 when enabled.
BACKGROUND_CONCEPTS = [
    "sky", "wall", "floor", "ground", "grass", "tree", "road", "building", "water", "field", "mountain",
    "ceiling", "window", "curtain",
]

# Subset of the 80 ImageNet templates from the CLIP paper that are commonly used for dense prediction.
TEMPLATES = [
    "a bad photo of a {}.", "a photo of many {}.", "a sculpture of a {}.", "a photo of the hard to see {}.",
    "a low resolution photo of the {}.", "a rendering of a {}.", "graffiti of a {}.", "a bad photo of the {}.",
    "a cropped photo of the {}.", "a tattoo of a {}.", "the embroidered {}.", "a photo of a hard to see {}.",
    "a bright photo of a {}.", "a photo of a clean {}.", "a photo of a dirty {}.", "a dark photo of the {}.",
    "a drawing of a {}.", "a photo of my {}.", "the plastic {}.", "a photo of the cool {}.",
    "a close-up photo of a {}.", "a black and white photo of the {}.", "a painting of the {}.",
    "a painting of a {}.", "a pixelated photo of the {}.", "a sculpture of the {}.", "a bright photo of the {}.",
    "a cropped photo of a {}.", "a plastic {}.", "a photo of the dirty {}.", "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.", "a photo of the {}.", "a good photo of the {}.", "a rendering of the {}.",
    "a {} in a video game.", "a photo of one {}.", "a doodle of a {}.", "a close-up photo of the {}.",
    "a photo of a {}.", "the origami {}.", "the {} in a video game.", "a sketch of a {}.", "a doodle of the {}.",
    "a origami {}.", "a low resolution photo of a {}.", "the toy {}.", "a rendition of the {}.",
    "a photo of the clean {}.", "a photo of a large {}.", "a rendition of a {}.", "a photo of a nice {}.",
    "a photo of a weird {}.", "a blurry photo of a {}.", "a cartoon {}.", "art of a {}.", "a sketch of the {}.",
    "a embroidered {}.", "a pixelated photo of a {}.", "itap of the {}.", "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.", "a plushie {}.", "a photo of the nice {}.", "a photo of the small {}.",
    "a photo of the weird {}.", "the cartoon {}.", "art of the {}.", "a drawing of the {}.",
    "a photo of the large {}.", "a black and white photo of a {}.", "the plushie {}.", "a dark photo of a {}.",
    "itap of a {}.", "graffiti of the {}.", "a toy {}.", "itap of my {}.", "a photo of a cool {}.",
    "a photo of a small {}.", "a tattoo of the {}.",
]


def original_prompt(name: str) -> str:
    """Prompt exactly as in the official code. `str.startswith('aeiou')` is never true for a class name,
    so every class gets the article "a" (e.g. "a photo of a aeroplane"). Kept only for reproduction."""
    return "a photo of an " + name if name.startswith("aeiou") else "a photo of a " + name


def fixed_prompt(name: str) -> str:
    article = "an" if name[0].lower() in "aeiou" else "a"
    return f"a photo of {article} {name}."
