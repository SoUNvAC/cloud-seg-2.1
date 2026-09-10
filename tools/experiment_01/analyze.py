"""Statistics, plots and the protocol §10 acceptance verdict.

Produces, under ``work_dirs/experiment_01/``:

* ``summary.csv``      — one row per run (delegated to make_summary)
* ``bootstrap.json``   — paired per-seed differences and 95 % bootstrap CIs
* ``acceptance.json``  — machine-readable verdict for every §10 condition
* ``acceptance.md``    — the same checklist in prose, for the report
* ``alpha_layers.png`` — 24-layer alpha curves (protocol §8)
* ``residual_ratio.png`` — 24-layer residual-ratio curves (protocol §8)

The bootstrap resamples *test images* (10,000 paired draws by default), which is
why it consumes the per-image arrays written by ``PerImageIoUMetric``. Image
order is verified to match between the two runs being compared; a mismatch
aborts rather than producing a meaningless interval.

Plots are skipped with a clear message when matplotlib is unavailable — the
numbers are unaffected.
"""

import argparse
import json
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from common import (  # noqa: E402
    ACCEPT_MAX_CLASS_REGRESSION,
    ACCEPT_MAX_EXTRA_PARAMS_M,
    ACCEPT_MAX_LATENCY_INCREASE,
    ACCEPT_MAX_MEMORY_INCREASE,
    ACCEPT_MAX_SEED_REGRESSION,
    ACCEPT_MEAN_GAIN,
    ACCEPT_SEED_WINS,
    BASELINE_SEEDS,
    CLASSES,
    EXP_DIR,
    MAIN_SEEDS,
    SCREEN_SEED,
    baseline_run_id,
    discover_file,
    ensure_dir,
    main_run_id,
    read_json,
    run_dir,
    screen_run_id,
    write_json,
)

import make_summary  # noqa: E402

N_BOOTSTRAP = 10000
BOOTSTRAP_CHUNK = 500


# ---------------------------------------------------------------------------
# per-image accounting
# ---------------------------------------------------------------------------
def load_per_image(path):
    import numpy as np

    data = np.load(path, allow_pickle=False)
    return {
        "img_paths": np.asarray(data["img_paths"]),
        "intersect": data["intersect"].astype(np.int64),
        "pred_area": data["pred_area"].astype(np.int64),
        "label_area": data["label_area"].astype(np.int64),
        "classes": [str(name) for name in data["classes"]],
    }


def miou_from_areas(intersect, pred_area, label_area):
    """mIoU (percent) from summed areas, matching mmseg's own formula."""
    import numpy as np

    union = pred_area + label_area - intersect
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, intersect / union, np.nan)
    return float(np.nanmean(iou) * 100.0)


def per_class_iou_from_areas(intersect, pred_area, label_area):
    import numpy as np

    union = pred_area + label_area - intersect
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, intersect / union, np.nan)
    return (iou * 100.0).tolist()


def paired_bootstrap(seed_pairs, n_boot=N_BOOTSTRAP, rng_seed=0):
    """95 % CI for mean_seeds(mIoU_A - mIoU_B), resampling test images.

    ``seed_pairs`` is a list of ``(A, B)`` per-image dicts; the same sampled
    image indices are used for both models (paired) and across seeds, so the
    interval describes the mean difference over the three seeds.
    """
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    num_images = seed_pairs[0][0]["intersect"].shape[0]
    deltas = np.empty(n_boot, dtype=np.float64)

    def _miou(areas, idx):
        intersect = areas["intersect"][idx].sum(axis=1)
        pred_area = areas["pred_area"][idx].sum(axis=1)
        label_area = areas["label_area"][idx].sum(axis=1)
        union = pred_area + label_area - intersect
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(union > 0, intersect / union, np.nan)
        return np.nanmean(iou, axis=-1) * 100.0

    done = 0
    while done < n_boot:
        size = min(BOOTSTRAP_CHUNK, n_boot - done)
        idx = rng.integers(0, num_images, size=(size, num_images))
        per_seed = [_miou(a, idx) - _miou(b, idx) for a, b in seed_pairs]
        deltas[done : done + size] = np.mean(per_seed, axis=0)
        done += size

    return {
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas, ddof=1)),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "n_bootstrap": int(n_boot),
        "n_images": int(num_images),
    }


# ---------------------------------------------------------------------------
# run record access
# ---------------------------------------------------------------------------
def load_record(run_id):
    return read_json(osp.join(run_dir(run_id), "metrics.json"))


def test_mIoU(run_id, which="best"):
    record = load_record(run_id)
    if not record:
        return None
    return ((record.get("test") or {}).get(which) or {}).get("mIoU")


def per_class(run_id, which="best"):
    record = load_record(run_id)
    if not record:
        return {}
    block = ((record.get("test") or {}).get(which) or {})
    return block.get("per_class_iou") or {}


def load_checkpoint_verifications():
    """Merge the per-run §10.7 verification reports into one verdict.

    Condition 7 is a property of every LS* run, not of one of them, so a single
    failing run fails the condition. A hand-run ``verify_checkpoint.py`` writing
    straight to ``EXP_DIR/checkpoint_verify.json`` is accepted too.
    """
    selection = read_json(osp.join(EXP_DIR, "screening.json")) or {}
    reports = {}

    if selection.get("winner"):
        for seed in MAIN_SEEDS:
            run_id = main_run_id(selection["layer_scale_type"],
                                 selection["layer_scale_init"], seed)
            payload = read_json(osp.join(run_dir(run_id), "checkpoint_verify.json"))
            if payload is not None:
                reports[run_id] = payload

    standalone = read_json(osp.join(EXP_DIR, "checkpoint_verify.json"))
    if not reports and standalone:
        return standalone

    if not reports:
        return {
            "passed": False,
            "detail": "no checkpoint verification report found",
            "runs": {},
        }

    return {
        "passed": all(report.get("passed") for report in reports.values()),
        "num_runs": len(reports),
        "expected_runs": len(MAIN_SEEDS),
        "all_runs_verified": len(reports) == len(MAIN_SEEDS),
        "per_run": {
            run_id: {
                "passed": report.get("passed"),
                "conditions": {
                    name: entry.get("passed")
                    for name, entry in (
                        (report.get("verification") or {}).get("conditions") or {}
                    ).items()
                },
                "num_alpha": len(report.get("alpha_names") or []),
                "num_trainable_vfm_params": report.get("num_trainable_vfm_params"),
            }
            for run_id, report in reports.items()
        },
    }


def per_image_for(run_id):
    record = load_record(run_id)
    if not record:
        return None
    path = (record.get("artifacts") or {}).get("per_image_test")
    if not path or not osp.exists(path):
        # Always the best-validation checkpoint's arrays: the paired bootstrap
        # must describe the same checkpoint the reported mIoU comes from.
        path = discover_file(run_id, "per_image_test_best.npz")
    if not path or not osp.exists(path):
        return None
    return load_per_image(path)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def build_bootstrap():
    selection = read_json(osp.join(EXP_DIR, "screening.json")) or {}
    if not selection.get("winner"):
        return {"available": False, "reason": "no LS* selected — screening stage has not run"}

    scale_type = selection["layer_scale_type"]
    init = selection["layer_scale_init"]

    per_seed = []
    skipped = []
    for seed in MAIN_SEEDS:
        ls_path = per_image_for(main_run_id(scale_type, init, seed))
        b0_path = per_image_for(baseline_run_id(seed))
        if ls_path is None or b0_path is None:
            skipped.append(
                {
                    "seed": seed,
                    "reason": "missing per-image arrays for "
                              f"{'LS*' if ls_path is None else ''}"
                              f"{' and ' if ls_path is None and b0_path is None else ''}"
                              f"{'B0' if b0_path is None else ''}",
                }
            )
            continue
        ls_names, b0_names = ls_path["img_paths"], b0_path["img_paths"]
        same_order = (
            ls_names.shape == b0_names.shape
            and bool((ls_names == b0_names).all())
        )
        if not same_order:
            raise SystemExit(
                f"[analyze] test image order differs between LS* seed {seed} "
                f"({ls_names.shape[0]} images) and B0 seed {seed} "
                f"({b0_names.shape[0]} images); the paired comparison would be "
                "invalid, so no interval is reported."
            )
        per_seed.append((seed, ls_path, b0_path))

    if not per_seed:
        return {"available": False, "reason": "no paired per-image data", "skipped": skipped}

    payload = {"available": True, "skipped": skipped, "per_seed": {}}
    seed_deltas = []
    for seed, ls_areas, b0_areas in per_seed:
        ls_miou = miou_from_areas(
            ls_areas["intersect"].sum(0),
            ls_areas["pred_area"].sum(0),
            ls_areas["label_area"].sum(0),
        )
        b0_miou = miou_from_areas(
            b0_areas["intersect"].sum(0),
            b0_areas["pred_area"].sum(0),
            b0_areas["label_area"].sum(0),
        )
        payload["per_seed"][str(seed)] = {
            "mIoU_LS": round(ls_miou, 4),
            "mIoU_B0": round(b0_miou, 4),
            "delta": round(ls_miou - b0_miou, 4),
        }
        seed_deltas.append(ls_miou - b0_miou)

    payload["paired_mean_delta"] = round(sum(seed_deltas) / len(seed_deltas), 4)
    payload["paired_seed_deltas"] = [round(value, 4) for value in seed_deltas]

    if len(per_seed) == len(MAIN_SEEDS):
        payload["bootstrap"] = paired_bootstrap(
            [(ls_areas, b0_areas) for _s, ls_areas, b0_areas in per_seed]
        )
    else:
        payload["bootstrap"] = None
        payload["bootstrap_note"] = (
            "bootstrap requires all three paired seeds; only "
            f"{len(per_seed)} available"
        )
    return payload


def per_class_deltas():
    """Three-seed mean per-class IoU for LS* and B0, and their difference."""
    selection = read_json(osp.join(EXP_DIR, "screening.json")) or {}
    if not selection.get("winner"):
        return None
    scale_type = selection["layer_scale_type"]
    init = selection["layer_scale_init"]

    result = {}
    for label, ids in (
        ("LS", [main_run_id(scale_type, init, seed) for seed in MAIN_SEEDS]),
        ("B0", [baseline_run_id(seed) for seed in BASELINE_SEEDS]),
    ):
        per_class_means = {}
        for class_name in CLASSES:
            values = []
            for run_id in ids:
                value = per_class(run_id).get(class_name)
                if value is not None:
                    values.append(value)
            per_class_means[class_name] = (
                round(sum(values) / len(values), 4) if values else None
            )
        result[label] = per_class_means

    result["delta"] = {
        class_name: (
            round(result["LS"][class_name] - result["B0"][class_name], 4)
            if result["LS"].get(class_name) is not None
            and result["B0"].get(class_name) is not None
            else None
        )
        for class_name in CLASSES
    }
    return result


# ---------------------------------------------------------------------------
# acceptance
# ---------------------------------------------------------------------------
def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def evaluate_acceptance(bootstrap, class_deltas, latency, checkpoint_verify):
    selection = read_json(osp.join(EXP_DIR, "screening.json")) or {}
    conditions = {}

    ls_test = [test_mIoU(main_run_id(selection["layer_scale_type"],
                                     selection["layer_scale_init"], seed))
               for seed in MAIN_SEEDS] if selection.get("winner") else []
    b0_test = [test_mIoU(baseline_run_id(seed)) for seed in BASELINE_SEEDS]

    mean_ls = _mean(ls_test)
    mean_b0 = _mean(b0_test)

    # 1. mean(mIoU_LS*) >= B0 + 0.25
    if mean_ls is not None and mean_b0 is not None:
        conditions["1_mean_gain_at_least_0.25"] = {
            "passed": mean_ls >= mean_b0 + ACCEPT_MEAN_GAIN,
            "mean_mIoU_LS": round(mean_ls, 4),
            "mean_mIoU_B0": round(mean_b0, 4),
            "gain": round(mean_ls - mean_b0, 4),
            "threshold": ACCEPT_MEAN_GAIN,
        }
    else:
        conditions["1_mean_gain_at_least_0.25"] = {
            "passed": False,
            "detail": "missing test metrics",
        }

    # 2. at least 2/3 seeds above their paired baseline
    wins = 0
    comparable = 0
    paired = {}
    for index, seed in enumerate(MAIN_SEEDS):
        ls_value = ls_test[index] if index < len(ls_test) else None
        b0_value = b0_test[BASELINE_SEEDS.index(seed)] if seed in BASELINE_SEEDS else None
        if ls_value is None or b0_value is None:
            continue
        comparable += 1
        wins += 1 if ls_value > b0_value else 0
        paired[str(seed)] = {"LS": ls_value, "B0": b0_value,
                             "delta": round(ls_value - b0_value, 4)}
    conditions["2_at_least_2_of_3_seeds_improve"] = {
        "passed": comparable == 3 and wins >= ACCEPT_SEED_WINS,
        "wins": wins,
        "comparable_seeds": comparable,
        "required": ACCEPT_SEED_WINS,
        "per_seed": paired,
    }

    # 3. no seed regresses by more than 0.15
    worst = None
    for seed, entry in paired.items():
        if worst is None or entry["delta"] < worst[1]:
            worst = (seed, entry["delta"])
    conditions["3_no_seed_regression_over_0.15"] = {
        "passed": bool(worst) and worst[1] >= -ACCEPT_MAX_SEED_REGRESSION,
        "worst_seed": worst[0] if worst else None,
        "worst_delta": worst[1] if worst else None,
        "threshold": -ACCEPT_MAX_SEED_REGRESSION,
    }

    # 4. no class IoU regresses by more than 0.40 (three-seed mean)
    if class_deltas:
        worst_class = min(
            ((name, value) for name, value in class_deltas["delta"].items()
             if value is not None),
            key=lambda item: item[1],
            default=None,
        )
        conditions["4_no_class_regression_over_0.40"] = {
            "passed": bool(worst_class) and worst_class[1] >= -ACCEPT_MAX_CLASS_REGRESSION,
            "worst_class": worst_class[0] if worst_class else None,
            "worst_delta": worst_class[1] if worst_class else None,
            "threshold": -ACCEPT_MAX_CLASS_REGRESSION,
            "per_class_delta": class_deltas["delta"],
        }
    else:
        conditions["4_no_class_regression_over_0.40"] = {
            "passed": False, "detail": "missing per-class metrics"
        }

    # 5. at most 0.001M extra trainable parameters
    extra_params = None
    if selection.get("winner"):
        ls_params = [
            (load_record(main_run_id(selection["layer_scale_type"],
                                     selection["layer_scale_init"], seed)) or {})
            .get("params", {}).get("trainable_params")
            for seed in MAIN_SEEDS
        ]
        b0_params = [
            (load_record(baseline_run_id(seed)) or {}).get("params", {}).get("trainable_params")
            for seed in BASELINE_SEEDS
        ]
        ls_value, b0_value = _mean(ls_params), _mean(b0_params)
        if ls_value is not None and b0_value is not None:
            extra_params = ls_value - b0_value
    conditions["5_extra_params_within_0.001M"] = {
        "passed": extra_params is not None and extra_params <= ACCEPT_MAX_EXTRA_PARAMS_M * 1e6,
        "extra_params": extra_params,
        "threshold": ACCEPT_MAX_EXTRA_PARAMS_M * 1e6,
    }

    # 6. latency and peak memory within 1 %
    latency_check = (latency or {}).get("comparison") or {}
    conditions["6a_latency_within_1_percent"] = {
        "passed": bool(latency_check.get("latency_within_tolerance")),
        "detail": latency_check.get("latency"),
        "threshold": ACCEPT_MAX_LATENCY_INCREASE,
    }
    conditions["6b_peak_memory_within_1_percent"] = {
        "passed": bool(latency_check.get("memory_within_tolerance")),
        "detail": latency_check.get("memory"),
        "threshold": ACCEPT_MAX_MEMORY_INCREASE,
    }

    # 7. frozen VFM + checkpoint restores every alpha_l
    verify = checkpoint_verify or {}
    conditions["7_vfm_frozen_and_alpha_restorable"] = {
        "passed": bool(verify.get("passed")),
        "detail": verify,
    }

    passed = all(entry.get("passed") for entry in conditions.values())
    return {
        "passed": passed,
        "conditions": conditions,
        "verdict": (
            "SUCCESS — every protocol §10 condition is satisfied"
            if passed
            else "FAILED — at least one protocol §10 condition is not satisfied; "
                 "a positive trend is not a substitute for success"
        ),
    }


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def load_stats_records(run_id):
    path = discover_file(run_id, "layer_scale_stats.jsonl")
    if path is None:
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _final_alpha_per_layer(records):
    import numpy as np

    if not records:
        return None
    final = records[-1].get("alpha") or []
    if not final:
        return None
    num_layers = records[-1].get("num_layers") or 0
    if num_layers and len(final) == num_layers:
        return np.asarray(final, dtype=float)
    # channel-wise: alpha is flattened as layer-major, so reshape per layer
    if num_layers and len(final) % num_layers == 0:
        width = len(final) // num_layers
        return np.asarray(final, dtype=float).reshape(num_layers, width).mean(axis=1)
    return np.asarray(final, dtype=float)


def make_plots(plot_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[analyze] matplotlib unavailable ({exc}); skipping plots")
        return []

    selection = read_json(osp.join(EXP_DIR, "screening.json")) or {}
    if not selection.get("winner"):
        print("[analyze] no LS* selected; skipping plots")
        return []

    scale_type = selection["layer_scale_type"]
    init = selection["layer_scale_init"]
    written = []

    # --- alpha_layers.png -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for seed in MAIN_SEEDS:
        records = load_stats_records(main_run_id(scale_type, init, seed))
        final = _final_alpha_per_layer(records)
        if final is None:
            continue
        axes[0].plot(range(1, len(final) + 1), final, marker="o",
                     markersize=3, label=f"LS* seed {seed}")
    axes[0].set_xlabel("interaction layer index")
    axes[0].set_ylabel(r"final $\alpha_l$")
    axes[0].set_title(r"Final $\alpha_l$ per layer")
    axes[0].axhline(1.0, color="grey", linestyle="--", linewidth=1,
                    label=r"$\alpha = 1$ (no scaling)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    records = load_stats_records(main_run_id(scale_type, init, SCREEN_SEED))
    if records:
        iterations = [record["iter"] for record in records]
        num_layers = records[-1].get("num_layers") or 0
        for layer in range(num_layers):
            if scale_type == "channel":
                continue  # one line per channel would be unreadable
            series = [
                (record.get("alpha") or [None] * num_layers)[layer]
                for record in records
            ]
            axes[1].plot(iterations, series, linewidth=0.9, alpha=0.75)
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel(r"$\alpha_l$")
        axes[1].set_title(
            rf"$\alpha_l$ trajectory, LS* seed {SCREEN_SEED} "
            f"({scale_type})"
        )
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    alpha_path = osp.join(plot_dir, "alpha_layers.png")
    fig.savefig(alpha_path, dpi=150)
    plt.close(fig)
    written.append(alpha_path)

    # --- residual_ratio.png ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for seed in MAIN_SEEDS:
        records = load_stats_records(main_run_id(scale_type, init, seed))
        if not records:
            continue
        num_layers = records[-1].get("num_layers") or 0
        means = []
        for layer in range(num_layers):
            column = [
                record["residual_ratio"][layer]
                for record in records
                if len(record.get("residual_ratio") or []) > layer
            ]
            means.append(sum(column) / len(column) if column else float("nan"))
        axes[0].plot(range(1, len(means) + 1), means, marker="o",
                     markersize=3, label=f"LS* seed {seed}")

    for seed in BASELINE_SEEDS:
        records = load_stats_records(baseline_run_id(seed))
        if not records:
            continue
        num_layers = records[-1].get("num_layers") or 0
        means = []
        for layer in range(num_layers):
            column = [
                record["residual_ratio"][layer]
                for record in records
                if len(record.get("residual_ratio") or []) > layer
            ]
            means.append(sum(column) / len(column) if column else float("nan"))
        axes[0].plot(range(1, len(means) + 1), means, linestyle="--",
                     linewidth=1, color="grey", alpha=0.6,
                     label=f"B0 seed {seed}" if seed == BASELINE_SEEDS[0] else None)
    axes[0].set_xlabel("interaction layer index")
    axes[0].set_ylabel(r"mean $\|\alpha_l \delta_l\| / \|x_l\|$")
    axes[0].set_title("Residual ratio per layer (mean over training)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    records = load_stats_records(main_run_id(scale_type, init, SCREEN_SEED))
    if records:
        iterations = [record["iter"] for record in records]
        num_layers = records[-1].get("num_layers") or 0
        for layer in range(num_layers):
            series = [
                (record.get("residual_ratio") or [None] * num_layers)[layer]
                for record in records
            ]
            axes[1].plot(iterations, series, linewidth=0.9, alpha=0.75)
        axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel(r"$\|\alpha_l \delta_l\| / \|x_l\|$")
        axes[1].set_title(f"Residual ratio trajectory, LS* seed {SCREEN_SEED}")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    ratio_path = osp.join(plot_dir, "residual_ratio.png")
    fig.savefig(ratio_path, dpi=150)
    plt.close(fig)
    written.append(ratio_path)

    return written


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_acceptance_md(acceptance, bootstrap, class_deltas, gate, latency,
                         checkpoint_verify):
    lines = ["# Experiment 01 — acceptance checklist", ""]
    lines.append(f"**Verdict: {'PASS' if acceptance['passed'] else 'FAIL'}**")
    lines.append("")
    lines.append("| §10 | condition | result | detail |")
    lines.append("|---|---|---|---|")
    for name, entry in acceptance["conditions"].items():
        # Condition keys are prefixed with their protocol §10 clause number.
        clause, _, label = name.partition("_")
        detail = entry.get("detail")
        if isinstance(detail, dict):
            detail = ", ".join(f"{k}={v}" for k, v in detail.items() if k != "per_seed")
        elif detail is None:
            detail = ", ".join(
                f"{k}={v}" for k, v in entry.items() if k not in ("passed", "threshold")
            )
        lines.append(
            f"| {clause} | {label.replace('_', ' ')} | "
            f"{'PASS' if entry.get('passed') else 'FAIL'} | {detail} |"
        )
    lines.append("")

    if bootstrap and bootstrap.get("available"):
        lines.append("## Paired per-seed differences (mIoU_LS* - mIoU_B0)")
        lines.append("")
        lines.append("| seed | mIoU LS* | mIoU B0 | delta |")
        lines.append("|---:|---:|---:|---:|")
        for seed, entry in sorted(bootstrap["per_seed"].items()):
            lines.append(
                f"| {seed} | {entry['mIoU_LS']} | {entry['mIoU_B0']} | {entry['delta']} |"
            )
        lines.append("")
        if bootstrap.get("bootstrap"):
            boot = bootstrap["bootstrap"]
            lines.append(
                f"Mean paired difference **{bootstrap['paired_mean_delta']}** mIoU; "
                f"95 % bootstrap CI over {boot['n_images']} test images "
                f"({boot['n_bootstrap']:,} paired resamples): "
                f"[{boot['ci95_low']}, {boot['ci95_high']}]."
            )
            lines.append("")
    else:
        lines.append("Paired bootstrap: not available "
                     f"({(bootstrap or {}).get('reason', 'not run')}).")
        lines.append("")

    if class_deltas:
        lines.append("## Per-class IoU (three-seed mean)")
        lines.append("")
        header = "| model | " + " | ".join(CLASSES) + " |"
        lines.append(header)
        lines.append("|---" * (len(CLASSES) + 1) + "|")
        for label in ("LS", "B0", "delta"):
            values = " | ".join(str(class_deltas[label].get(name)) for name in CLASSES)
            lines.append(f"| {label} | {values} |")
        lines.append("")

    if latency:
        lines.append("## Latency and memory (protocol §7)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(latency.get("comparison", latency), indent=2))
        lines.append("```")
        lines.append("")

    if checkpoint_verify:
        lines.append("## Checkpoint verification (protocol §10.7)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(checkpoint_verify, indent=2))
        lines.append("```")
        lines.append("")

    if gate:
        lines.append("## Baseline gate (protocol §3)")
        lines.append("")
        lines.append(f"passed: **{gate.get('passed')}** — {gate.get('conclusion')}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dir(EXP_DIR)

    rows = make_summary.build_rows()
    if rows:
        path = make_summary.write_csv(osp.join(EXP_DIR, "summary.csv"), rows)
        print(f"[analyze] summary.csv: {len(rows)} runs -> {path}")
    else:
        print("[analyze] no runs with metrics.json; run the collect stage first")

    bootstrap = build_bootstrap()
    write_json(osp.join(EXP_DIR, "bootstrap.json"), bootstrap)

    class_deltas = per_class_deltas()
    latency = read_json(osp.join(EXP_DIR, "latency.json"))
    checkpoint_verify = load_checkpoint_verifications()
    write_json(osp.join(EXP_DIR, "checkpoint_verify.json"), checkpoint_verify)
    gate = read_json(osp.join(EXP_DIR, "baseline_gate.json"))

    acceptance = evaluate_acceptance(bootstrap, class_deltas, latency, checkpoint_verify)
    write_json(osp.join(EXP_DIR, "acceptance.json"), acceptance)

    md = render_acceptance_md(acceptance, bootstrap, class_deltas, gate, latency,
                              checkpoint_verify)
    with open(osp.join(EXP_DIR, "acceptance.md"), "w", encoding="utf-8") as handle:
        handle.write(md)

    plots = make_plots(EXP_DIR)
    print(f"[analyze] verdict: {'PASS' if acceptance['passed'] else 'FAIL'}")
    for path in plots:
        print(f"[analyze] plot -> {path}")


if __name__ == "__main__":
    main()
