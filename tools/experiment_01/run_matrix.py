"""Drive the full experiment-01 run matrix.

Stages
------
``env``       collect ``env.txt`` and the dataset SHA256 manifest
``baseline``  B0-13 / B0-42 / B0-3407 — unmodified Cloud-Adapter
``gate``      protocol §3 baseline reproduction gate
``screen``    S0..S5 at seed 42, evaluated on validation only, picks LS*
``main``      LS* at seeds 13 / 42 / 3407
``bench``     protocol §7 latency / peak memory for B0-42 and LS*-42
``verify``    protocol §10.7 frozen-VFM and alpha round-trip check
``collect``   fold every run's artifacts into ``metrics.json``
``analyze``   summary.csv, bootstrap CIs, plots, acceptance report
``all``       env -> baseline -> gate -> screen -> main -> bench -> verify
              -> collect -> analyze

Each stage is skipped when its outputs already exist, so ``all`` is safe to
re-run after an interruption. Use ``--force`` to redo, ``--only <run_id>`` to
redo a single run, and ``--dry-run`` to print the commands without running them.

The baseline gate (protocol §3) stops the pipeline: if the unmodified model does
not reproduce the published number within tolerance, the conclusion is
"environment/reproduction failure" and no structural claim may be made.
"""

import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import time

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from common import (  # noqa: E402
    BACKBONE_CHECKPOINT,
    BASELINE_GATE_ABS,
    BASELINE_GATE_RANGE,
    BASELINE_SEEDS,
    EXP_DIR,
    MAIN_SEEDS,
    PAPER_MIOU_L1C,
    REPO_ROOT,
    SCREEN_SEED,
    SCREEN_VARIANTS,
    baseline_run_id,
    baseline_specs,
    discover_checkpoints,
    ensure_dir,
    main_run_id,
    main_specs,
    python_executable,
    read_json,
    run_dir,
    screen_run_id,
    screen_specs,
    write_json,
)

TRAIN = osp.join("tools", "train.py")
EVAL = osp.join("tools", "experiment_01", "eval_run.py")
ANALYZE = osp.join("tools", "experiment_01", "analyze.py")
VERIFY = osp.join("tools", "experiment_01", "verify_checkpoint.py")
BENCH = osp.join("tools", "experiment_01", "bench_latency.py")


# ---------------------------------------------------------------------------
# process helpers
# ---------------------------------------------------------------------------
def _display(cmd):
    return " ".join(cmd)


def run_command(cmd, log_path, dry_run=False):
    print(f"[run_matrix] $ {_display(cmd)}")
    if dry_run:
        return 0
    ensure_dir(osp.dirname(osp.abspath(log_path)))
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log.write(_display(cmd) + "\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        process.wait()
        log.write(f"===== exit code {process.returncode} =====\n")
    if process.returncode != 0:
        print(f"[run_matrix] command failed with code {process.returncode}")
    return process.returncode


def _train_cfg_options(spec):
    # Per-image arrays are produced by the explicit post-training evaluations
    # below, one file per (split, checkpoint), so the in-training validation
    # must not write them: it would leave a file whose checkpoint is ambiguous.
    options = [
        f"randomness.seed={spec.seed}",
        "val_evaluator.per_image_path=None",
        "test_evaluator.per_image_path=None",
    ]
    options.extend(spec.cfg_options)
    return options


# ---------------------------------------------------------------------------
# train / evaluate one run
# ---------------------------------------------------------------------------
def train_run(spec, dry_run=False, force=False):
    directory = run_dir(spec.run_id)
    metrics_path = osp.join(directory, "metrics.json")
    record = read_json(metrics_path, default={}) or {}
    checkpoints = discover_checkpoints(spec.run_id)
    aborted = read_json(osp.join(directory, "ABORTED.json"))
    if record.get("status") == "complete" and not force:
        print(f"[run_matrix] {spec.run_id}: completed metrics exist, skipping training")
        return True
    if aborted and not force:
        print(
            f"[run_matrix] {spec.run_id}: run is marked ABORTED; "
            "use --force only after fixing the cause"
        )
        return False
    if checkpoints.get("last_iter", -1) >= 40000 and not force:
        print(f"[run_matrix] {spec.run_id}: final checkpoint exists, skipping training")
        return True

    if not dry_run:
        ensure_dir(directory)
    cmd = [
        python_executable(),
        TRAIN,
        spec.config_path,
        "--work-dir",
        directory,
        "--cfg-options",
        *_train_cfg_options(spec),
    ]
    if checkpoints.get("last") and not force:
        cmd.extend(["--resume", checkpoints["last"]])
        print(
            f"[run_matrix] {spec.run_id}: resuming from "
            f"iteration {checkpoints.get('last_iter')}"
        )
    started = time.time()
    code = run_command(cmd, osp.join(directory, "train.log"), dry_run=dry_run)
    elapsed = time.time() - started

    if not dry_run:
        meta = spec.as_dict()
        meta.update(
            {
                "train_wall_sec": round(elapsed, 2),
                "train_exit_code": code,
                "log_interval": 500,
            }
        )
        write_json(osp.join(directory, "run_meta.json"), meta)
    return code == 0


def evaluate_run(spec, checkpoint, split, which, dry_run=False, force=False):
    directory = run_dir(spec.run_id)
    metrics_out = osp.join(directory, f"eval_{split}_{which}.json")
    if osp.exists(metrics_out) and not force:
        print(f"[run_matrix] {spec.run_id}: {split}/{which} already evaluated")
        return True

    cmd = [
        python_executable(),
        EVAL,
        spec.config_path,
        checkpoint,
        "--split",
        split,
        "--backbone",
        BACKBONE_CHECKPOINT,
        "--work-dir",
        osp.join(directory, f"eval_{split}_{which}"),
        "--metrics-out",
        metrics_out,
        "--seed",
        str(spec.seed),
        # One file per (split, checkpoint): the best and the final checkpoint
        # must not overwrite each other's per-image arrays.
        "--per-image-out",
        osp.join(directory, f"per_image_{split}_{which}.npz"),
    ]
    if spec.cfg_options:
        cmd.extend(["--cfg-options", *spec.cfg_options])

    log_path = osp.join(directory, f"eval_{split}_{which}.log")
    return run_command(cmd, log_path, dry_run=dry_run) == 0


def summarise_params(spec, dry_run=False):
    directory = run_dir(spec.run_id)
    out = osp.join(directory, "params.json")
    if osp.exists(out) or dry_run:
        return
    cmd = [
        python_executable(),
        osp.join("tools", "experiment_01", "count_params.py"),
        spec.config_path,
        "--out",
        out,
    ]
    if spec.cfg_options:
        cmd.extend(["--cfg-options", *spec.cfg_options])
    run_command(cmd, osp.join(directory, "params.log"), dry_run=dry_run)


def execute_run(spec, dry_run=False, force=False):
    """Train, then evaluate on the checkpoints the protocol asks for."""
    print(f"\n{'=' * 72}\n[run_matrix] run {spec.run_id} ({spec.stage})\n{'=' * 72}")
    ok = train_run(spec, dry_run=dry_run, force=force)
    if not ok and not dry_run:
        print(f"[run_matrix] {spec.run_id}: training failed; skipping evaluation")
        return False

    summarise_params(spec, dry_run=dry_run)

    if dry_run:
        # Print the evaluation commands that would follow.
        for split, which in _evaluation_plan(spec):
            evaluate_run(spec, "<checkpoint>", split, which, dry_run=True)
        return True

    checkpoints = discover_checkpoints(spec.run_id)
    if checkpoints["best"] is None and checkpoints["last"] is None:
        print(f"[run_matrix] {spec.run_id}: no checkpoint found")
        return False

    aborted = read_json(osp.join(run_dir(spec.run_id), "ABORTED.json"))
    evaluations_ok = True
    for split, which in _evaluation_plan(spec):
        checkpoint = checkpoints.get(which)
        if checkpoint is None:
            print(f"[run_matrix] {spec.run_id}: no {which} checkpoint, skipping {split}")
            evaluations_ok = False
            continue
        if aborted and split == "test":
            # An aborted run must not inform anything about the test set.
            print(f"[run_matrix] {spec.run_id}: aborted, skipping test evaluation")
            continue
        evaluations_ok = (
            evaluate_run(spec, checkpoint, split, which, force=force)
            and evaluations_ok
        )

    return evaluations_ok


def _evaluation_plan(spec):
    if spec.stage == "screen":
        # Screening is decided on validation only.
        return [("val", "best"), ("val", "last")]
    return [("val", "best"), ("test", "best"), ("test", "last")]


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _mIoU(run_id, split="test", which="best"):
    record = read_json(osp.join(run_dir(run_id), "metrics.json"))
    if not record:
        return None
    block = (record.get(split) or {}).get(which) or {}
    return block.get("mIoU")


def check_baseline_gate(dry_run=False):
    """Protocol §3: the baseline must reproduce before anything else counts."""
    results = {}
    for seed in BASELINE_SEEDS:
        run_id = baseline_run_id(seed)
        record = read_json(osp.join(run_dir(run_id), "metrics.json"))
        results[run_id] = {
            "test_mIoU_best": (record or {}).get("test", {}).get("mIoU"),
            "test_mIoU_last": (
                ((record or {}).get("test", {}).get("last") or {}).get("mIoU")
            ),
            "status": (record or {}).get("status"),
        }

    best_values = [
        entry["test_mIoU_best"]
        for entry in results.values()
        if entry["test_mIoU_best"] is not None
    ]

    conditions = {}
    details = {}
    if len(best_values) == 3:
        b0 = sum(best_values) / 3.0
        spread = max(best_values) - min(best_values)
        delta = abs(results[baseline_run_id(42)]["test_mIoU_best"] - PAPER_MIOU_L1C)
        conditions["reproduces_paper_within_0.30"] = delta <= BASELINE_GATE_ABS
        conditions["seed_spread_within_0.80"] = spread <= BASELINE_GATE_RANGE
        conditions["no_failed_runs"] = all(
            entry["status"] == "complete" for entry in results.values()
        )
        details = {
            "B0": round(b0, 4),
            "delta_vs_paper": round(delta, 4),
            "seed_spread": round(spread, 4),
            "per_seed": results,
        }
    else:
        conditions["all_three_baselines_evaluated"] = False
        details = {"per_seed": results,
                   "missing": [k for k, v in results.items()
                               if v["test_mIoU_best"] is None]}

    passed = bool(conditions) and all(conditions.values())
    payload = {
        "passed": passed,
        "conditions": conditions,
        "details": details,
        "paper_mIoU": PAPER_MIOU_L1C,
        "tolerance": {"abs_mIoU": BASELINE_GATE_ABS, "seed_range": BASELINE_GATE_RANGE},
        "conclusion": (
            "baseline reproduced" if passed
            else "environment/reproduction failure — the structural experiment "
                 "must not be reported as success or failure"
        ),
    }
    if not dry_run:
        write_json(osp.join(EXP_DIR, "baseline_gate.json"), payload)

    print("\n[run_matrix] baseline gate (protocol §3)")
    for name, value in conditions.items():
        print(f"  {'PASS' if value else 'FAIL'}  {name}")
    print(f"  -> {'PASSED' if passed else 'FAILED'}: {payload['conclusion']}")
    return payload


def select_ls_star(dry_run=False):
    """Protocol §5: pick the best scalar initialisation on validation."""
    table = {}
    for variant, _config, scale_type, init in SCREEN_VARIANTS:
        run_id = screen_run_id(variant)
        record = read_json(osp.join(run_dir(run_id), "metrics.json"))
        table[variant] = {
            "run_id": run_id,
            "scale_type": scale_type,
            "init": init,
            "val_mIoU_best": ((record or {}).get("val") or {}).get("mIoU"),
            "val_mIoU_last": (
                (((record or {}).get("val") or {}).get("last") or {}).get("mIoU")
            ),
            "test_mIoU_best": ((record or {}).get("test") or {}).get("mIoU"),
            "status": (record or {}).get("status"),
        }

    # LS* is chosen among the scalar initialisations only (protocol §5).
    scalar_variants = [
        variant for variant, _c, scale_type, _i in SCREEN_VARIANTS
        if scale_type == "scalar"
    ]
    scored = [
        (variant, table[variant]["val_mIoU_best"])
        for variant in scalar_variants
        if table[variant]["val_mIoU_best"] is not None
    ]

    payload = {
        "selection_metric": "val_mIoU_best",
        "table": table,
        "scalar_candidates": scalar_variants,
    }
    if not scored:
        payload["winner"] = None
        payload["note"] = "no screening run produced a validation metric"
    else:
        winner, winner_value = max(scored, key=lambda item: item[1])
        entry = table[winner]
        payload["winner"] = winner
        payload["layer_scale_type"] = entry["scale_type"]
        payload["layer_scale_init"] = entry["init"]
        payload["winner_val_mIoU"] = winner_value
        # Diagnostics: does simply not scaling (S0) or going channel-wise (S5)
        # already beat the best scalar initialisation?
        reference = table.get("S0", {}).get("val_mIoU_best")
        payload["s0_val_mIoU"] = reference
        payload["s0_beats_winner"] = (
            reference is not None and reference > winner_value
        )

    if not dry_run:
        write_json(osp.join(EXP_DIR, "screening.json"), payload)

    print("\n[run_matrix] screening on validation (protocol §5)")
    for variant in table:
        entry = table[variant]
        print(
            f"  {variant}: scale={entry['scale_type']} init={entry['init']} "
            f"val mIoU={entry['val_mIoU_best']} (last {entry['val_mIoU_last']})"
        )
    print(f"  -> LS* = {payload.get('winner')} {payload.get('layer_scale_type')} "
          f"init {payload.get('layer_scale_init')}")
    return payload


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_env(args):
    run_command(
        [python_executable(), osp.join("tools", "experiment_01", "collect_env.py")]
        + (["--force"] if args.force else []),
        osp.join(EXP_DIR, "collect_env.log"),
        dry_run=args.dry_run,
    )


def stage_baseline(args):
    specs = baseline_specs()
    for spec in specs:
        execute_run(spec, dry_run=args.dry_run, force=args.force)
    collect_runs([spec.run_id for spec in specs], args)


def stage_screen(args):
    for spec in screen_specs():
        execute_run(spec, dry_run=args.dry_run, force=args.force)
    collect_runs([spec.run_id for spec in screen_specs()], args)
    select_ls_star(dry_run=args.dry_run)


def stage_main(args):
    selection = read_json(osp.join(EXP_DIR, "screening.json"))
    if (not selection or selection.get("winner") is None) and args.dry_run:
        selection = {
            "winner": "<selected-scalar>",
            "layer_scale_type": "scalar",
            "layer_scale_init": 0.1,
        }
        print(
            "[run_matrix] dry-run: using scalar/init=0.1 as a placeholder "
            "for the validation-selected LS*"
        )
    if not selection or selection.get("winner") is None:
        print("[run_matrix] no screening winner; run the `screen` stage first")
        return
    specs = main_specs(selection["layer_scale_type"], selection["layer_scale_init"])
    for spec in specs:
        execute_run(spec, dry_run=args.dry_run, force=args.force)
    collect_runs([spec.run_id for spec in specs], args)


def collect_runs(run_ids, args):
    for run_id in run_ids:
        run_command(
            [python_executable(), osp.join("tools", "experiment_01", "collect_run.py"),
             run_id],
            osp.join(run_dir(run_id), "collect_run.log"),
            dry_run=args.dry_run,
        )


def stage_collect(args):
    run_ids = [baseline_run_id(seed) for seed in BASELINE_SEEDS]
    run_ids += [screen_run_id(variant) for variant, *_ in SCREEN_VARIANTS]
    selection = read_json(osp.join(EXP_DIR, "screening.json"))
    if selection and selection.get("winner"):
        run_ids += [
            main_run_id(selection["layer_scale_type"], selection["layer_scale_init"], seed)
            for seed in MAIN_SEEDS
        ]
    collect_runs(run_ids, args)


def stage_verify(args):
    """Protocol §10.7 for every LS* run (and the baseline, as a control)."""
    selection = read_json(osp.join(EXP_DIR, "screening.json"))
    if (not selection or selection.get("winner") is None) and args.dry_run:
        selection = {
            "winner": "<selected-scalar>",
            "layer_scale_type": "scalar",
            "layer_scale_init": 0.1,
        }
    if not selection or selection.get("winner") is None:
        print("[run_matrix] no screening winner; run the `screen` stage first")
        return
    specs = main_specs(selection["layer_scale_type"], selection["layer_scale_init"])
    for spec in specs:
        checkpoints = discover_checkpoints(spec.run_id)
        checkpoint = checkpoints.get("best") or checkpoints.get("last")
        if checkpoint is None and args.dry_run:
            checkpoint = "<checkpoint>"
        elif checkpoint is None:
            print(f"[run_matrix] {spec.run_id}: no checkpoint, skipping verification")
            continue
        out = osp.join(run_dir(spec.run_id), "checkpoint_verify.json")
        if osp.exists(out) and not args.force:
            print(f"[run_matrix] {spec.run_id}: checkpoint verification exists")
            continue
        run_command(
            [python_executable(), VERIFY, spec.config_path, checkpoint,
             "--backbone", BACKBONE_CHECKPOINT, "--out", out],
            osp.join(run_dir(spec.run_id), "verify.log"),
            dry_run=args.dry_run,
        )


def stage_bench(args):
    """Protocol §7: latency and peak memory for B0 and LS*."""
    if osp.exists(osp.join(EXP_DIR, "latency.json")) and not args.force:
        print("[run_matrix] latency.json exists, skipping benchmark")
        return

    selection = read_json(osp.join(EXP_DIR, "screening.json")) or {}
    if not selection.get("winner") and args.dry_run:
        selection = {
            "winner": "<selected-scalar>",
            "layer_scale_type": "scalar",
            "layer_scale_init": 0.1,
        }
    specs = [baseline_specs()[1]]  # B0 seed 42, the seed used for screening
    if selection.get("winner"):
        specs += [spec for spec in main_specs(selection["layer_scale_type"],
                                              selection["layer_scale_init"])
                  if spec.seed == SCREEN_SEED]

    cmd = [python_executable(), BENCH, "--reference", specs[0].run_id]
    for spec in specs:
        checkpoint = discover_checkpoints(spec.run_id).get("best")
        if checkpoint is None and args.dry_run:
            checkpoint = "<checkpoint>"
        elif checkpoint is None:
            print(f"[run_matrix] {spec.run_id}: no checkpoint, skipping benchmark")
            return
        cmd += ["--model", f"{spec.run_id}={spec.config_path}|{checkpoint}"]
    # LS* is selected at train time via --cfg-options; the benchmark must build
    # the same architecture. The keys are inert for the baseline config, which
    # does not enable layer scaling.
    for spec in specs:
        if spec.cfg_options:
            cmd += ["--cfg-options", *spec.cfg_options]
            break
    run_command(cmd, osp.join(EXP_DIR, "bench_latency.log"), dry_run=args.dry_run)


def stage_analyze(args):
    run_command(
        [python_executable(), ANALYZE] + (["--force"] if args.force else []),
        osp.join(EXP_DIR, "analyze.log"),
        dry_run=args.dry_run,
    )


def stage_gate(args):
    # The gate reads metrics.json, so make sure the baselines have been folded
    # (this is a no-op when the baseline stage already collected them).
    collect_runs([spec.run_id for spec in baseline_specs()], args)
    payload = check_baseline_gate(dry_run=args.dry_run)
    return payload.get("passed", False)


STAGES = {
    "env": stage_env,
    "baseline": stage_baseline,
    "gate": stage_gate,
    "screen": stage_screen,
    "main": stage_main,
    "verify": stage_verify,
    "bench": stage_bench,
    "collect": stage_collect,
    "analyze": stage_analyze,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=list(STAGES) + ["all"])
    parser.add_argument("--force", action="store_true",
                        help="redo steps whose outputs already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without executing them")
    parser.add_argument("--only", default=None,
                        help="run a single run_id (baseline/screen/main stages)")
    parser.add_argument("--ignore-gate", action="store_true",
                        help="continue past a failed baseline gate (records the "
                             "override in the report)")
    args = parser.parse_args()

    if not args.dry_run:
        ensure_dir(EXP_DIR)

    if args.only:
        for spec in baseline_specs() + screen_specs():
            if spec.run_id == args.only:
                execute_run(spec, dry_run=args.dry_run, force=True)
                return
        selection = read_json(osp.join(EXP_DIR, "screening.json")) or {}
        if selection.get("winner"):
            for spec in main_specs(selection["layer_scale_type"],
                                   selection["layer_scale_init"]):
                if spec.run_id == args.only:
                    execute_run(spec, dry_run=args.dry_run, force=True)
                    return
        print(f"[run_matrix] unknown run id {args.only!r}")
        return

    if args.stage == "all":
        stage_env(args)
        stage_baseline(args)
        gate_passed = stage_gate(args)
        if not gate_passed and not args.dry_run and not args.ignore_gate:
            print(
                "\n[run_matrix] baseline gate FAILED — stopping.\n"
                "  Protocol §3: the experiment is recorded as an environment /\n"
                "  reproduction failure; no structural claim may be made.\n"
                "  Re-run with --ignore-gate to continue anyway (the override is\n"
                "  recorded in the report)."
            )
            write_json(
                osp.join(EXP_DIR, "gate_override.json"),
                {"ignored": False, "reason": "gate failed, pipeline stopped"},
            )
            return
        if not gate_passed and not args.dry_run:
            write_json(
                osp.join(EXP_DIR, "gate_override.json"),
                {"ignored": True, "reason": "user passed --ignore-gate"},
            )
        stage_screen(args)
        stage_main(args)
        stage_bench(args)
        stage_verify(args)
        stage_collect(args)
        stage_analyze(args)
        return

    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
