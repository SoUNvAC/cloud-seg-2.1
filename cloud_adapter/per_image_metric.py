"""Per-image IoU accounting for experiment 01.

Protocol §9 requires paired bootstrap confidence intervals obtained by
resampling *test images* 10,000 times. Aggregate metrics cannot support that, so
this metric additionally stores, for every image, the three per-class areas that
mmseg's own ``IoUMetric`` is built from:

* ``intersect[c]``  -- true positives of class ``c``
* ``pred_area[c]``  -- predicted pixels of class ``c``  (TP + FP)
* ``label_area[c]`` -- ground-truth pixels of class ``c`` (TP + FN)

From these, any resampled mIoU (and per-class IoU) is recoverable by summing the
selected images and computing ``intersect / (pred_area + label_area - intersect)``
-- the exact formula used by ``IoUMetric``, so the resampled numbers stay
consistent with the reported ones.

The arrays are written to a ``.npz`` next to the run so an analysis pass can
redo the statistics without rerunning inference.
"""

import os.path as osp

import numpy as np
from mmseg.evaluation.metrics.iou_metric import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class PerImageIoUMetric(IoUMetric):
    """``IoUMetric`` that also dumps per-image intersection/area arrays."""

    def __init__(self, per_image_path=None, **kwargs):
        super().__init__(**kwargs)
        self.per_image_path = per_image_path
        self.per_image_records = []

    def process(self, data_batch: dict, data_samples) -> None:
        if not self.format_only:
            num_classes = len(self.dataset_meta["classes"])
            for data_sample in data_samples:
                pred_label = data_sample["pred_sem_seg"]["data"].squeeze()
                label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)
                intersect, pred_area, label_area, _ = self.intersect_and_union(
                    pred_label, label, num_classes, self.ignore_index
                )
                self.per_image_records.append(
                    (
                        data_sample.get("img_path", ""),
                        np.asarray(intersect, dtype=np.int64),
                        np.asarray(pred_area, dtype=np.int64),
                        np.asarray(label_area, dtype=np.int64),
                    )
                )
        super().process(data_batch, data_samples)

    def compute_metrics(self, results: list) -> dict:
        metrics = super().compute_metrics(results)
        if self.per_image_path and self.per_image_records:
            directory = osp.dirname(osp.abspath(self.per_image_path))
            if directory:
                import os

                os.makedirs(directory, exist_ok=True)
            np.savez_compressed(
                self.per_image_path,
                img_paths=np.array(
                    [record[0] for record in self.per_image_records]
                ),
                intersect=np.stack([record[1] for record in self.per_image_records]),
                pred_area=np.stack([record[2] for record in self.per_image_records]),
                label_area=np.stack([record[3] for record in self.per_image_records]),
                classes=np.array(list(self.dataset_meta["classes"])),
            )
        return metrics
