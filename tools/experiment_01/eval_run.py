"""Evaluate one trained run on the validation or test split.

Wraps ``mmseg``'s runner so the metrics dict comes back in-process instead of
having to be scraped out of logs and timestamped work directories. Also injects
the pretrained backbone through ``LoadBackboneHook``, which is required because
checkpoints only contain the adapter and decode head (see the README: "the saved
weights include only the adapter and head components").

Writes:
    --metrics-out   the evaluator's metrics dict as JSON
    per-image npz   via ``PerImageIoUMetric.per_image_path``
"""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..", "..")))

from mmengine.config import Config, DictAction  # noqa: E402
from mmengine.runner import Runner  # noqa: E402

import cloud_adapter  # noqa: E402,F401  (registers metrics and hooks)
import cloud_adapter.datasets  # noqa: E402,F401
import cloud_adapter.models  # noqa: E402,F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--backbone", default="checkpoints/dinov2_converted_512x512.pth")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--per-image-out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint
    cfg.merge_from_dict({"randomness.seed": args.seed})

    # The pretrained VFM weights are not part of the checkpoint.
    custom_hooks = list(cfg.get("custom_hooks", []))
    custom_hooks.append(
        dict(type="LoadBackboneHook", checkpoint_path=args.backbone)
    )
    cfg.custom_hooks = custom_hooks

    evaluator_key = "val_evaluator" if args.split == "val" else "test_evaluator"
    if args.per_image_out:
        os.makedirs(osp.dirname(osp.abspath(args.per_image_out)), exist_ok=True)
        cfg.merge_from_dict({f"{evaluator_key}.per_image_path": args.per_image_out})
    else:
        cfg.merge_from_dict({f"{evaluator_key}.per_image_path": None})

    runner = Runner.from_cfg(cfg)
    metrics = runner.val() if args.split == "val" else runner.test()

    os.makedirs(osp.dirname(osp.abspath(args.metrics_out)), exist_ok=True)
    from mmengine.fileio import dump

    dump(dict(metrics), args.metrics_out)
    print(f"[eval_run] {args.split} metrics -> {args.metrics_out}")
    print(metrics)


if __name__ == "__main__":
    main()
