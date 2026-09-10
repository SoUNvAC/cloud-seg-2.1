"""Collapse one run's artifacts into a single ``metrics.json``.

Reads what training and evaluation left in the run directory (run metadata,
evaluation metrics, the layer-scale JSONL, parameter counts) and normalises it
into the record the rest of the analysis consumes. Missing pieces are recorded
as ``None`` rather than guessed, so an incomplete run is visibly incomplete.
"""

import argparse
import json
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from common import (  # noqa: E402
    CLASSES,
    discover_checkpoints,
    discover_file,
    git_commit,
    read_json,
    run_dir,
    write_json,
)

# Steady-state window for the residual-ratio summary: the mean over the last
# 10 % of logged steps, alongside the mean over the whole run.
FINAL_FRACTION = 0.10


def load_stats(run_id):
    path = discover_file(run_id, "layer_scale_stats.jsonl")
    if path is None:
        return [], None
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records, path


def _column(records, key):
    """List-per-record column -> list-per-layer-of-lists."""
    return [record.get(key) or [] for record in records]


def summarise_alpha(records):
    if not records:
        return None
    final = records[-1].get("alpha") or []
    return {
        "final": final,
        "mean": records[-1].get("alpha_mean"),
        "std": records[-1].get("alpha_std"),
        "max_abs": records[-1].get("alpha_max_abs"),
        "num_layers": records[-1].get("num_layers"),
        "grad_norm_final": records[-1].get("alpha_grad_norm"),
    }


def summarise_residual_ratio(records):
    if not records:
        return None
    window = max(1, int(len(records) * FINAL_FRACTION))
    tail = records[-window:]

    num_layers = max((len(r.get("residual_ratio") or []) for r in records), default=0)
    overall, final = [], []
    for layer in range(num_layers):
        column = [
            r["residual_ratio"][layer]
            for r in records
            if len(r.get("residual_ratio") or []) > layer
        ]
        if column:
            overall.append(sum(column) / len(column))
        tail_column = [
            r["residual_ratio"][layer]
            for r in tail
            if len(r.get("residual_ratio") or []) > layer
        ]
        if tail_column:
            final.append(sum(tail_column) / len(tail_column))

    flat = [value for record in records for value in (record.get("residual_ratio") or [])]
    return {
        "per_layer_overall": overall,
        "per_layer_final": final,
        "mean_overall": (sum(flat) / len(flat)) if flat else None,
        "max_overall": max(flat) if flat else None,
        "num_layers": num_layers,
    }


def _eval_block(run_id, split, which):
    payload = read_json(osp.join(run_dir(run_id), f"eval_{split}_{which}.json"))
    if payload is None:
        return None
    per_class = {}
    for index, name in enumerate(CLASSES):
        value = payload.get(f"IoU.{name}")
        if value is not None:
            per_class[name] = value
    return {
        "mIoU": payload.get("mIoU"),
        "aAcc": payload.get("aAcc"),
        "mAcc": payload.get("mAcc"),
        "mDice": payload.get("mDice"),
        "mFscore": payload.get("mFscore"),
        "mPrecision": payload.get("mPrecision"),
        "mRecall": payload.get("mRecall"),
        "per_class_iou": per_class,
        "raw": payload,
    }


def build_record(run_id, stage, variant=None, seed=None, cfg_options=None,
                 notes=""):
    meta = read_json(osp.join(run_dir(run_id), "run_meta.json"), default={}) or {}
    records, stats_path = load_stats(run_id)
    params = read_json(osp.join(run_dir(run_id), "params.json"))
    aborted = read_json(osp.join(run_dir(run_id), "ABORTED.json"))
    checkpoints = discover_checkpoints(run_id)

    peak_mem = [
        record["peak_mem_mb"]
        for record in records
        if isinstance(record.get("peak_mem_mb"), (int, float))
        and record["peak_mem_mb"] == record["peak_mem_mb"]  # drop NaN
    ]

    evaluation = {
        split: {
            which: _eval_block(run_id, split, which)
            for which in ("best", "last")
        }
        for split in ("val", "test")
    }

    val_best = evaluation["val"]["best"] or {}
    test_best = evaluation["test"]["best"] or {}
    test_last = evaluation["test"]["last"] or {}

    status = "complete"
    if aborted:
        status = "aborted"
    elif not records or not (test_best or val_best):
        status = "incomplete"

    return {
        "run_id": run_id,
        "stage": stage or meta.get("stage"),
        "variant": variant if variant is not None else meta.get("variant"),
        "seed": seed if seed is not None else meta.get("seed"),
        "notes": notes or meta.get("notes", ""),
        "config": meta.get("config"),
        "cfg_options": cfg_options if cfg_options is not None else meta.get("cfg_options", []),
        "git_commit": git_commit(),
        "status": status,
        "aborted": bool(aborted),
        "abort_reasons": (aborted or {}).get("reasons", []),
        "layer_scale": {
            "enabled": bool(params and params.get("num_alpha_tensors")),
            "type": meta.get("layer_scale_type"),
            "init": meta.get("layer_scale_init"),
            "num_alpha_tensors": (params or {}).get("num_alpha_tensors"),
        },
        "params": params,
        "train": {
            "wall_sec": meta.get("train_wall_sec"),
            "iters": records[-1]["iter"] if records else None,
            "peak_mem_mb": max(peak_mem) if peak_mem else None,
            "log_interval": meta.get("log_interval"),
        },
        "alpha": summarise_alpha(records),
        "residual_ratio": summarise_residual_ratio(records),
        "val": {
            "mIoU": val_best.get("mIoU"),
            "per_class_iou": val_best.get("per_class_iou"),
            "best": val_best,
            "last": evaluation["val"]["last"],
        },
        "test": {
            "mIoU": test_best.get("mIoU"),
            "aAcc": test_best.get("aAcc"),
            "mAcc": test_best.get("mAcc"),
            "mDice": test_best.get("mDice"),
            "per_class_iou": test_best.get("per_class_iou"),
            "best": test_best,
            "last": test_last,
        },
        "checkpoints": {
            "best": checkpoints.get("best"),
            "last": checkpoints.get("last"),
            "last_iter": checkpoints.get("last_iter"),
        },
        "artifacts": {
            "stats_jsonl": stats_path,
            # Named per (split, checkpoint) so a paired bootstrap always reads
            # the arrays of the checkpoint the reported metric belongs to.
            "per_image_val": discover_file(run_id, "per_image_val_best.npz"),
            "per_image_val_last": discover_file(run_id, "per_image_val_last.npz"),
            "per_image_test": discover_file(run_id, "per_image_test_best.npz"),
            "per_image_test_last": discover_file(run_id, "per_image_test_last.npz"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    record = build_record(args.run_id, args.stage, seed=args.seed)
    out = osp.join(run_dir(args.run_id), "metrics.json")
    write_json(out, record)
    print(
        f"[collect_run] {args.run_id}: status={record['status']} "
        f"val_mIoU={record['val']['mIoU']} test_mIoU={record['test']['mIoU']} "
        f"-> {out}"
    )


if __name__ == "__main__":
    main()
