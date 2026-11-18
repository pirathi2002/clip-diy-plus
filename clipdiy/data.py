"""PASCAL VOC 2012 and COCO-Object datasets (images + semantic ground truth)."""
import os
from typing import List

import cv2
import numpy as np
from PIL import Image

from .prompts import VOC_CLASSES

EVAL_SHORT_SIDE = 448  # official protocol: GT resized (nearest) so its shorter side is 448 px


def resize_short_side(mask: np.ndarray, short: int = EVAL_SHORT_SIDE) -> np.ndarray:
    """Same as torchvision `resize(mask, 448, NEAREST)` used by the official evaluation."""
    h, w = mask.shape
    if h <= w:
        nh, nw = short, int(short * w / h)
    else:
        nh, nw = int(short * h / w), short
    return np.array(Image.fromarray(mask).resize((nw, nh), Image.NEAREST))


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class VOCSegmentation:
    classes = VOC_CLASSES

    def __init__(self, root: str, split: str = "val"):
        voc = os.path.join(root, "VOCdevkit", "VOC2012")
        with open(os.path.join(voc, "ImageSets", "Segmentation", f"{split}.txt")) as f:
            self.ids: List[str] = [x.strip() for x in f if x.strip()]
        self.img_dir = os.path.join(voc, "JPEGImages")
        self.gt_dir = os.path.join(voc, "SegmentationClass")

    def __len__(self):
        return len(self.ids)

    def image(self, i: int) -> np.ndarray:
        return read_rgb(os.path.join(self.img_dir, self.ids[i] + ".jpg"))

    def gt(self, i: int) -> np.ndarray:
        """Label map with 255 = ignore, resized to the evaluation resolution."""
        return resize_short_side(np.array(Image.open(os.path.join(self.gt_dir, self.ids[i] + ".png"))))


class COCOObject:
    """COCO-Object: 80 thing classes + background, built from val2017 instance masks.

    Follows the official code (thirdparty/mscoco.py): the first instance covering a pixel wins, and only
    images with >= 1000 foreground pixels are kept (the official "keep all val images" branch never runs
    because it checks for "full_coco" while the config passes "coco_full")."""

    def __init__(self, root: str, split: str = "val", min_fg_pixels: int = 1000):
        from pycocotools.coco import COCO
        self.coco = COCO(os.path.join(root, "annotations", f"instances_{split}2017.json"))
        self.img_dir = os.path.join(root, f"{split}2017")
        cats = self.coco.loadCats(sorted(self.coco.getCatIds()))
        self.cat_to_label = {c["id"]: i + 1 for i, c in enumerate(cats)}
        self.classes = ["background"] + [c["name"] for c in cats]
        self.ids = [i for i in sorted(self.coco.getImgIds()) if (self._mask(i) > 0).sum() >= min_fg_pixels]

    def __len__(self):
        return len(self.ids)

    def _mask(self, img_id: int) -> np.ndarray:
        info = self.coco.loadImgs(img_id)[0]
        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            mask[(mask == 0) & (self.coco.annToMask(ann) > 0)] = self.cat_to_label[ann["category_id"]]
        return mask

    def image(self, i: int) -> np.ndarray:
        return read_rgb(os.path.join(self.img_dir, self.coco.loadImgs(self.ids[i])[0]["file_name"]))

    def gt(self, i: int) -> np.ndarray:
        return resize_short_side(self._mask(self.ids[i]))


def load_dataset(name: str, root: str, split: str):
    if name == "voc":
        return VOCSegmentation(root, split)
    if name == "coco":
        return COCOObject(root, split)
    raise ValueError(name)
