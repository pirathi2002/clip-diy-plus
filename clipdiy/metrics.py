import numpy as np


class ConfusionMatrix:
    """Dataset-level confusion matrix; mIoU = mean over classes of TP / (TP + FP + FN)."""

    def __init__(self, n_classes: int, ignore_index: int = 255):
        self.n = n_classes
        self.ignore = ignore_index
        self.mat = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        valid = gt != self.ignore
        idx = self.n * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
        self.mat += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self) -> np.ndarray:
        tp = np.diag(self.mat)
        denom = self.mat.sum(0) + self.mat.sum(1) - tp
        return tp / np.maximum(denom, 1)

    def miou(self) -> float:
        return float(self.iou().mean())

    def pixel_acc(self) -> float:
        return float(np.diag(self.mat).sum() / max(self.mat.sum(), 1))
