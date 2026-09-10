"""Checkpoint / freezing verification (protocol §10.7).

Protocol §10 condition 7 requires two things of every LS* run:

1. DINOv2 stays **completely frozen** — no parameter of the pretrained vision
   backbone may be trainable, and the interaction layers' ``alpha_l`` must be
   the only thing the layer-scale change adds there;
2. the checkpoint can **independently restore all 24 ``alpha_l``** — each layer
   keeps its own parameter (they are not tied), the values survive a
   save/reload round trip bit-exactly, and the count is exactly 24.

The round trip is done against a *freshly built* model so a shared in-memory
reference cannot mask a serialization bug.

Usage::

    python tools/experiment_01/verify_checkpoint.py \\
        configs/experiment_01/main_ls_star.py \\
        work_dirs/experiment_01/LSstar_scalar_init0p1_seed42/best_*.pth \\
        --out work_dirs/experiment_01/checkpoint_verify.json
"""

import argparse
import os
import os.path as osp
import sys
import tempfile

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..", "..")))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from common import (  # noqa: E402
    BACKBONE_CHECKPOINT,
    EXP_DIR,
    ensure_dir,
    write_json,
)

# Parameters of the adapter live under this prefix inside the segmentor.
ADAPTER_MARKER = "cloud_adapter"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--backbone", default=BACKBONE_CHECKPOINT)
    parser.add_argument("--expected-num-layers", type=int, default=24)
    parser.add_argument("--out", default=osp.join(EXP_DIR, "checkpoint_verify.json"))
    return parser.parse_args()


def build(config, backbone, load_run_checkpoint=True, checkpoint=None):
    """Build the segmentor, optionally with the converted VFM weights loaded."""
    from mmengine.config import Config
    from mmengine.runner.checkpoint import _load_checkpoint
    from mmseg.registry import MODELS

    import cloud_adapter  # noqa: F401
    import cloud_adapter.datasets  # noqa: F401
    import cloud_adapter.models  # noqa: F401

    cfg = Config.fromfile(config)
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)

    if backbone:
        converted = _load_checkpoint(backbone, map_location="cpu")
        if "state_dict" in converted:
            converted = converted["state_dict"]
        model.load_state_dict(
            {f"backbone.{key}": value for key, value in converted.items()},
            strict=False,
        )
    if load_run_checkpoint and checkpoint:
        payload = _load_checkpoint(checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("state_dict", payload), strict=False)
    return model


def collect_alpha(model):
    """Every parameter whose name ends in ``.alpha`` inside the adapter."""
    found = {}
    for name, param in model.named_parameters():
        if ADAPTER_MARKER in name and name.endswith(".alpha"):
            found[name] = param
    return found


def check(results):
    conditions = {}

    alpha_names = results["alpha_names"]
    conditions["all_alpha_present"] = {
        "passed": len(alpha_names) == results["expected_num_layers"],
        "found": len(alpha_names),
        "expected": results["expected_num_layers"],
    }

    conditions["alpha_are_independent"] = {
        "passed": results["num_distinct_alpha_objects"] == len(alpha_names),
        "distinct_objects": results["num_distinct_alpha_objects"],
        "num_alpha": len(alpha_names),
        "detail": "each layer owns its own parameter (no weight tying)",
    }

    conditions["alpha_are_trainable"] = {
        "passed": results["num_trainable_alpha"] == len(alpha_names),
        "trainable": results["num_trainable_alpha"],
        "num_alpha": len(alpha_names),
    }

    conditions["alpha_in_checkpoint"] = {
        "passed": results["alpha_missing_from_checkpoint"] == [],
        "missing": results["alpha_missing_from_checkpoint"],
    }

    conditions["alpha_round_trip_exact"] = {
        "passed": results["num_alpha_mismatched_after_reload"] == 0,
        "mismatched": results["num_alpha_mismatched_after_reload"],
    }

    conditions["vfm_frozen"] = {
        "passed": results["num_trainable_vfm_params"] == 0,
        "trainable_vfm_params": results["num_trainable_vfm_params"],
        "num_vfm_params": results["num_vfm_params"],
        "detail": "no DINOv2 parameter outside the cloud_adapter is trainable",
    }

    passed = all(entry["passed"] for entry in conditions.values())
    return {"passed": passed, "conditions": conditions}


def main():
    args = parse_args()

    import torch

    model = build(args.config, args.backbone, checkpoint=args.checkpoint)

    # Exercise the training-time configuration: CloudAdapterDinoVisionTransformer
    # .train() is what applies set_requires_grad(..., ["cloud_adapter"]).
    model.train()

    alpha = collect_alpha(model)
    alpha_names = sorted(alpha)

    vfm_params = [
        (name, param)
        for name, param in model.named_parameters()
        if name.startswith("backbone.") and ADAPTER_MARKER not in name
    ]
    trainable_vfm = [name for name, param in vfm_params if param.requires_grad]

    checkpoint_keys = set()
    payload = None
    try:
        from mmengine.runner.checkpoint import _load_checkpoint

        payload = _load_checkpoint(args.checkpoint, map_location="cpu")
        checkpoint_keys = set(payload.get("state_dict", payload).keys())
    except Exception as exc:  # noqa: BLE001
        print(f"[verify_checkpoint] could not read checkpoint keys: {exc}")

    missing_from_ckpt = [name for name in alpha_names if name not in checkpoint_keys]

    # --- save / reload round trip against a fresh model ------------------
    mismatched = []
    with tempfile.TemporaryDirectory() as tmp:
        dump_path = osp.join(tmp, "round_trip.pth")
        saved = {name: param.detach().clone() for name, param in alpha.items()}
        torch.save({"state_dict": model.state_dict()}, dump_path)

        fresh = build(args.config, backbone=None, checkpoint=dump_path)
        fresh_alpha = collect_alpha(fresh)
        for name in alpha_names:
            if name not in fresh_alpha:
                mismatched.append({"name": name, "reason": "absent after reload"})
            elif not torch.equal(fresh_alpha[name].detach().cpu(),
                                 saved[name].cpu()):
                mismatched.append({"name": name, "reason": "value differs"})
        extra = sorted(set(fresh_alpha) - set(alpha_names))

    results = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "backbone_checkpoint": args.backbone,
        "expected_num_layers": args.expected_num_layers,
        "alpha_names": alpha_names,
        "alpha_shapes": {name: list(param.shape) for name, param in alpha.items()},
        "alpha_values": {
            name: (
                float(param.detach().cpu())
                if param.ndim == 0
                else [round(float(v), 6) for v in param.detach().cpu().flatten()[:8]]
            )
            for name, param in alpha.items()
        },
        "num_distinct_alpha_objects": len({id(param) for param in alpha.values()}),
        "num_trainable_alpha": sum(
            1 for param in alpha.values() if param.requires_grad
        ),
        "alpha_missing_from_checkpoint": missing_from_ckpt,
        "alpha_extra_after_reload": extra,
        "num_alpha_mismatched_after_reload": len(mismatched),
        "alpha_mismatches": mismatched,
        "num_vfm_params": len(vfm_params),
        "num_trainable_vfm_params": len(trainable_vfm),
        "trainable_vfm_param_names": trainable_vfm[:20],
    }
    results["verification"] = check(results)
    results["passed"] = results["verification"]["passed"]

    ensure_dir(osp.dirname(osp.abspath(args.out)))
    write_json(args.out, results)

    print(f"\n[verify_checkpoint] {args.checkpoint}")
    for name, entry in results["verification"]["conditions"].items():
        print(f"  {'PASS' if entry['passed'] else 'FAIL'}  {name}")
    print(f"  -> {'PASSED' if results['passed'] else 'FAILED'}")
    print(f"[verify_checkpoint] -> {osp.abspath(args.out)}")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
