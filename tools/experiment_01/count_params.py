"""Report trainable/total parameter counts for a run's config (protocol §2, §10.5).

Builds the segmentor straight from the config, so the counts describe the model
that will actually be trained rather than a hand-maintained estimate.

    python tools/experiment_01/count_params.py <config> [--cfg-options ...]
"""

import argparse
import os.path as osp
import sys

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..", "..")))

from mmengine.config import Config, DictAction  # noqa: E402
from mmseg.registry import MODELS  # noqa: E402

import cloud_adapter  # noqa: E402,F401
import cloud_adapter.datasets  # noqa: E402,F401


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    parser.add_argument("--out", default=None, help="optional JSON output path")
    return parser.parse_args()


def summarise(model):
    # nn.Module.train() recurses into children, so this runs the backbone's
    # overridden train() and therefore set_requires_grad(["cloud_adapter"]).
    model.train()
    total = trainable = 0
    alpha_params = 0
    alpha_names = []
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            if name.endswith(".alpha"):
                alpha_params += param.numel()
                alpha_names.append(name)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio_percent": round(trainable * 100.0 / total, 4) if total else 0.0,
        "alpha_params": alpha_params,
        "num_alpha_tensors": len(alpha_names),
        "alpha_names": alpha_names,
    }


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # Building the model would otherwise try to load the converted VFM weights,
    # which are irrelevant to a parameter count.
    if cfg.model.get("backbone", {}).get("init_cfg"):
        cfg.model.backbone.init_cfg = None

    model = MODELS.build(cfg.model)
    summary = summarise(model)

    print(f"config:            {args.config}")
    print(f"total params:      {summary['total_params']:,}")
    print(f"trainable params:  {summary['trainable_params']:,} "
          f"({summary['trainable_ratio_percent']}% of total)")
    print(f"alpha tensors:     {summary['num_alpha_tensors']} "
          f"({summary['alpha_params']:,} parameters)")

    if args.out:
        from common import write_json

        write_json(args.out, summary)


if __name__ == "__main__":
    main()
