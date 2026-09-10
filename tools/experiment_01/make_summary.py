"""Build ``work_dirs/experiment_01/summary.csv`` from the runs' metrics.json.

One row per run. Columns follow protocol §8 (test mIoU, the four per-class IoUs,
parameters, peak memory, latency, alpha statistics, residual ratio) with the
val/test distinction kept explicit, because screening runs are decided on
validation and must not be read as test results.
"""

import argparse
import csv
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from common import (  # noqa: E402
    BASELINE_SEEDS,
    CLASSES,
    EXP_DIR,
    MAIN_SEEDS,
    SCREEN_VARIANTS,
    baseline_run_id,
    ensure_dir,
    main_run_id,
    read_json,
    run_dir,
    screen_run_id,
)

COLUMNS = [
    "run_id",
    "stage",
    "variant",
    "model",
    "seed",
    "scale_type",
    "init",
    "status",
    "val_mIoU",
    "test_mIoU",
    "test_mIoU_final_ckpt",
    "aAcc",
    "mAcc",
    "mDice",
    "mIoU_CRS",
    "mIoU_TKC",
    "mIoU_TNC",
    "mIoU_CDS",
    "total_params",
    "trainable_params",
    "alpha_params",
    "alpha_tensors",
    "peak_mem_mb",
    "train_wall_sec",
    "latency_median_ms",
    "alpha_mean",
    "alpha_std",
    "alpha_max_abs",
    "residual_ratio_mean",
    "residual_ratio_max",
    "aborted",
    "notes",
]

DEVELOPMENT = {"S0": (None, None)}
VARIANT_META = {v[0]: (v[2], v[3]) for v in SCREEN_VARIANTS}


def _per_class(block, name):
    if not block:
        return None
    return (block.get("per_class_iou") or {}).get(name)


def load_latency():
    return read_json(osp.join(EXP_DIR, "latency.json"), default={}) or {}


def latency_entry(latency, run_id):
    """Per-run latency block, under either the flat or the namespaced layout."""
    entry = latency.get(run_id)
    if entry is None:
        entry = (latency.get("models") or {}).get(run_id)
    return entry or {}


def load_run(run_id, latency):
    record = read_json(osp.join(run_dir(run_id), "metrics.json"))
    if record is None:
        return None

    params = record.get("params") or {}
    alpha = record.get("alpha") or {}
    ratio = record.get("residual_ratio") or {}
    test_best = (record.get("test") or {}).get("best") or {}
    test_last = (record.get("test") or {}).get("last") or {}
    variant = record.get("variant") or ""
    layer_scale = record.get("layer_scale") or {}
    scale_type, init = VARIANT_META.get(variant, (None, None))
    if layer_scale.get("type") is not None:
        scale_type = layer_scale["type"]
    if layer_scale.get("init") is not None:
        init = layer_scale["init"]

    bench = latency_entry(latency, run_id)

    return {
        "run_id": run_id,
        "stage": record.get("stage"),
        "variant": variant,
        "model": (
            "Cloud-Adapter (no layer scale)"
            if record.get("stage") == "baseline"
            else f"Cloud-Adapter + layer scale ({scale_type} {init})"
        ),
        "seed": record.get("seed"),
        "scale_type": scale_type,
        "init": init,
        "status": record.get("status"),
        "val_mIoU": (record.get("val") or {}).get("mIoU"),
        "test_mIoU": test_best.get("mIoU"),
        "test_mIoU_final_ckpt": test_last.get("mIoU"),
        "aAcc": test_best.get("aAcc"),
        "mAcc": test_best.get("mAcc"),
        "mDice": test_best.get("mDice"),
        "mIoU_CRS": _per_class(test_best, CLASSES[0]),
        "mIoU_TKC": _per_class(test_best, CLASSES[1]),
        "mIoU_TNC": _per_class(test_best, CLASSES[2]),
        "mIoU_CDS": _per_class(test_best, CLASSES[3]),
        "total_params": params.get("total_params"),
        "trainable_params": params.get("trainable_params"),
        "alpha_params": params.get("alpha_params"),
        "alpha_tensors": params.get("num_alpha_tensors"),
        "peak_mem_mb": (record.get("train") or {}).get("peak_mem_mb"),
        "train_wall_sec": (record.get("train") or {}).get("wall_sec"),
        "latency_median_ms": bench.get("median_ms"),
        "alpha_mean": alpha.get("mean"),
        "alpha_std": alpha.get("std"),
        "alpha_max_abs": alpha.get("max_abs"),
        "residual_ratio_mean": ratio.get("mean_overall"),
        "residual_ratio_max": ratio.get("max_overall"),
        "aborted": record.get("aborted"),
        "notes": record.get("notes"),
    }


def collect_run_ids():
    run_ids = [baseline_run_id(seed) for seed in BASELINE_SEEDS]
    run_ids += [screen_run_id(variant) for variant, *_ in SCREEN_VARIANTS]
    selection = read_json(osp.join(EXP_DIR, "screening.json")) or {}
    if selection.get("winner"):
        run_ids += [
            main_run_id(selection["layer_scale_type"], selection["layer_scale_init"], seed)
            for seed in MAIN_SEEDS
        ]
    return run_ids


def build_rows():
    latency = load_latency()
    rows = []
    for run_id in collect_run_ids():
        row = load_run(run_id, latency)
        if row is not None:
            rows.append(row)
    return rows


def write_csv(path, rows):
    ensure_dir(osp.dirname(osp.abspath(path)))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in COLUMNS})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=osp.join(EXP_DIR, "summary.csv"))
    args = parser.parse_args()

    rows = build_rows()
    if not rows:
        print("[make_summary] no runs found under work_dirs/experiment_01")
        return
    path = write_csv(args.out, rows)
    print(f"[make_summary] {len(rows)} runs -> {path}")


if __name__ == "__main__":
    main()
