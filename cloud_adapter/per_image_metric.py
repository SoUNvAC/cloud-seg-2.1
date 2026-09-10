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

    def process(self, data_batch: dict, data_samples) -> None:
        start = len(self.results)
        super().process(data_batch, data_samples)
        if self.format_only:
            return

        # Store the path inside BaseMetric.results. That list is gathered from
        # every rank before compute_metrics(), unlike an instance-local side
        # list, so distributed evaluation produces a complete paired sample.
        batch_results = self.results[start:]
        if len(batch_results) != len(data_samples):
            raise RuntimeError("IoUMetric produced an unexpected result count")
        for offset, (areas, data_sample) in enumerate(
            zip(batch_results, data_samples)
        ):
            self.results[start + offset] = (*areas, data_sample.get("img_path", ""))

    def compute_metrics(self, results: list) -> dict:
        if self.format_only:
            return super().compute_metrics(results)

        area_results = [record[:4] for record in results]
        metrics = super().compute_metrics(area_results)

        # IoUMetric prints per-class values but only returns aggregate metrics.
        # Return them explicitly so the experiment acceptance checks can use
        # actual class IoUs rather than silently recording missing values.
        total_intersect = sum(record[0] for record in area_results)
        total_union = sum(record[1] for record in area_results)
        class_iou = (total_intersect / total_union).cpu().numpy() * 100.0
        for class_name, value in zip(self.dataset_meta["classes"], class_iou):
            metrics[f"IoU.{class_name}"] = float(np.round(value, 2))

        if self.per_image_path and results:
            directory = osp.dirname(osp.abspath(self.per_image_path))
            if directory:
                import os

                os.makedirs(directory, exist_ok=True)
            np.savez_compressed(
                self.per_image_path,
                img_paths=np.array([record[4] for record in results]),
                intersect=np.stack(
                    [record[0].cpu().numpy().astype(np.int64) for record in results]
                ),
                pred_area=np.stack(
                    [record[2].cpu().numpy().astype(np.int64) for record in results]
                ),
                label_area=np.stack(
                    [record[3].cpu().numpy().astype(np.int64) for record in results]
                ),
                classes=np.array(list(self.dataset_meta["classes"])),
            )
        return metrics
