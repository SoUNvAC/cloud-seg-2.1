"""Inference latency and peak-memory benchmark (protocol §7).

Protocol requirements implemented here:

* batch size 1, fixed input size (512x512 by default);
* 100 warm-up iterations, then 500 timed iterations per round;
* **three alternating rounds** per model (B0, LS*, B0, LS*, ...) so that any
  thermal or clock drift affects both models equally;
* identical GPU, precision and cudnn configuration for every model in a run;
* the reported latency is the **median**; peak memory is taken from
  ``torch.cuda.max_memory_allocated``.

Both models are loaded exactly the way evaluation loads them: the converted
DINOv2 weights are injected into ``backbone.*`` and the run checkpoint is
applied with ``strict=False`` — the same merge ``LoadBackboneHook`` performs.

Each model is built through ``MODELS.build`` with ``backbone.init_cfg`` cleared,
so the benchmark never depends on a checkpoint path baked into a config.

Usage::

    python tools/experiment_01/bench_latency.py            # auto-discover B0-42 / LS*-42
    python tools/experiment_01/bench_latency.py \\
        --model "B0_seed42=configs/experiment_01/s0_baseline_no_scale.py|work_dirs/.../best.pth" \\
        --model "LSstar=configs/experiment_01/main_ls_star.py|work_dirs/.../best.pth"
"""

import argparse
import json
import os.path as osp
import statistics
import sys
import time

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..", "..")))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from common import (  # noqa: E402
    ACCEPT_MAX_LATENCY_INCREASE,
    ACCEPT_MAX_MEMORY_INCREASE,
    BACKBONE_CHECKPOINT,
    EXP_DIR,
    REPO_ROOT,
    SCREEN_SEED,
    baseline_run_id,
    discover_checkpoints,
    ensure_dir,
    main_run_id,
    read_json,
    write_json,
)

DEFAULT_MODELS = [
    (baseline_run_id(SCREEN_SEED), "s0_baseline_no_scale.py"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="RUN_ID=CONFIG|CHECKPOINT",
        help="model to benchmark; repeatable. Defaults to the B0-42 baseline "
             "and the selected LS* run.",
    )
    parser.add_argument("--reference", default=None,
                        help="run_id used as the comparison baseline "
                             "(default: the first model given)")
    parser.add_argument("--backbone", default=BACKBONE_CHECKPOINT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SCREEN_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true",
                        help="benchmark under autocast (off by default: the "
                             "protocol locks a single precision for all models)")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=None,
        help="config overrides applied to every model, in the same form "
             "tools/train.py accepts (needed so LS* builds the same architecture "
             "it was trained with)",
    )
    parser.add_argument("--out", default=osp.join(EXP_DIR, "latency.json"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def build_model(config, checkpoint, backbone, seed, device, cfg_options=None):
    import torch
    from mmengine.config import Config
    from mmengine.runner.checkpoint import _load_checkpoint
    from mmseg.registry import MODELS

    import cloud_adapter  # noqa: F401  (registry side effects)
    import cloud_adapter.datasets  # noqa: F401
    import cloud_adapter.models  # noqa: F401

    cfg = Config.fromfile(config)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    # The benchmark loads weights explicitly below; leaving the Pretrained
    # init_cfg in place would make the result depend on a config path.
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)

    converted = _load_checkpoint(backbone, map_location="cpu")
    if "state_dict" in converted:
        converted = converted["state_dict"]
    backbone_state = {f"backbone.{key}": value for key, value in converted.items()}

    payload = _load_checkpoint(checkpoint, map_location="cpu")
    run_state = payload.get("state_dict", payload)

    # Mirror mmengine's load_checkpoint(): backbone first, then the run
    # checkpoint on top, both non-strict because each half is partial.
    model.load_state_dict(backbone_state, strict=False)
    missing, unexpected = model.load_state_dict(run_state, strict=False)

    model = model.to(device).eval()
    return model, {
        "checkpoint": osp.relpath(osp.abspath(checkpoint), REPO_ROOT).replace("\\", "/"),
        "num_missing_keys": len(missing),
        "num_unexpected_keys": len(unexpected),
    }


def make_inputs(batch_size, size, device):
    import torch
    from mmseg.structures import SegDataSample

    images = torch.rand(batch_size, 3, size, size, device=device)
    samples = []
    for _ in range(batch_size):
        sample = SegDataSample()
        sample.set_metainfo(
            dict(
                img_shape=(size, size),
                ori_shape=(size, size),
                pad_shape=(size, size),
                scale_factor=(1.0, 1.0),
                flip=False,
            )
        )
        samples.append(sample)
    return images, samples


def measure(model, images, samples, warmup, iters, device, use_amp):
    import torch

    def step():
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _forward(model, images, samples)
            else:
                _forward(model, images, samples)

    # Resolve the callable once, outside the timed region.
    mode = _resolve_mode(model, images, samples)

    def _forward(model_, images_, samples_):
        return _call(model_, images_, samples_, mode)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return times, peak_bytes / (1024 ** 2), mode


def _call(model, images, samples, mode):
    if mode == "preprocessed_predict":
        data = dict(inputs=images, data_samples=samples)
        batch_inputs, batch_samples = model.data_preprocessor(data, training=False)
        return model(batch_inputs, batch_samples, mode="predict")
    if mode == "predict":
        return model(images, samples, mode="predict")
    return model(images, samples, mode="tensor")


def _resolve_mode(model, images, samples):
    """Pick the deepest forward path this model actually supports.

    ``mode='predict'`` (preprocessing + decode head included) is what real
    inference runs; some mmseg versions expect the caller to preprocess. The
    fallback is ``mode='tensor'`` (backbone + neck only), which is recorded in
    the output so a comparison is never silently made across two modes — all
    models in one invocation are forced to the same mode.
    """
    import torch

    for mode in ("preprocessed_predict", "predict", "tensor"):
        try:
            with torch.no_grad():
                _call(model, images, samples, mode)
            return mode
        except Exception:  # noqa: BLE001 - any failure means "try the next path"
            continue
    raise RuntimeError("none of the forward modes worked for this model")


def parse_cfg_options(pairs):
    """``key=value`` strings -> dict, with values given Python literal typing."""
    import ast

    if not pairs:
        return None
    options = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # plain strings such as `scalar` stay as they are
        options[key] = value
    return options


def summarise(times):
    return {
        "median_ms": round(statistics.median(times), 4),
        "mean_ms": round(statistics.fmean(times), 4),
        "p10_ms": round(sorted(times)[int(0.10 * len(times))], 4),
        "p90_ms": round(sorted(times)[int(0.90 * len(times))], 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "std_ms": round(statistics.pstdev(times), 4),
        "n_samples": len(times),
    }


# ---------------------------------------------------------------------------
# model list
# ---------------------------------------------------------------------------
def resolve_models(args):
    """[(run_id, config_path, checkpoint_path)] from --model or auto-discovery."""
    if args.model:
        resolved = []
        for entry in args.model:
            run_id, _, payload = entry.partition("=")
            config, _, checkpoint = payload.partition("|")
            if not (run_id and config and checkpoint):
                raise SystemExit(
                    f"[bench_latency] bad --model {entry!r}; expected "
                    "RUN_ID=CONFIG|CHECKPOINT"
                )
            resolved.append((run_id, config, checkpoint))
        return resolved

    from common import CONFIG_DIR, screen_run_id

    resolved = [
        (baseline_run_id(SCREEN_SEED),
         osp.join(CONFIG_DIR, "s0_baseline_no_scale.py"),
         (discover_checkpoints(baseline_run_id(SCREEN_SEED)) or {}).get("best"))
    ]

    selection = read_json(osp.join(EXP_DIR, "screening.json")) or {}
    if selection.get("winner"):
        ls_id = main_run_id(selection["layer_scale_type"],
                            selection["layer_scale_init"], SCREEN_SEED)
        resolved.append(
            (ls_id,
             osp.join(CONFIG_DIR, "main_ls_star.py"),
             (discover_checkpoints(ls_id) or {}).get("best"))
        )
    else:
        print("[bench_latency] no LS* selected; benchmarking the baseline only")
    return [entry for entry in resolved if entry[2]]


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise SystemExit("[bench_latency] CUDA is not available on this machine")

    models = resolve_models(args)
    if not models:
        raise SystemExit(
            "[bench_latency] no trained checkpoints found — run the training "
            "stages first, or pass --model explicitly"
        )

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    print(f"[bench_latency] device={args.device} precision="
          f"{'amp-fp16' if args.amp else 'fp32'} batch={args.batch_size} "
          f"size={args.size}x{args.size} warmup={args.warmup} iters={args.iters} "
          f"rounds={args.rounds}")

    cfg_options = parse_cfg_options(args.cfg_options)
    built = []
    for run_id, config, checkpoint in models:
        print(f"[bench_latency] building {run_id} ...")
        model, info = build_model(config, checkpoint, args.backbone, args.seed,
                                  args.device, cfg_options=cfg_options)
        info["cfg_options"] = args.cfg_options or []
        images, samples = make_inputs(args.batch_size, args.size, args.device)
        built.append((run_id, model, images, samples, info))
        info["config"] = osp.relpath(osp.abspath(config), REPO_ROOT).replace("\\", "/")

    results = {run_id: {"rounds_ms": [], "round_peak_mem_mb": []}
               for run_id, *_ in built}
    mode = None
    for round_index in range(args.rounds):
        for run_id, model, images, samples, info in built:
            times, peak_mb, resolved = measure(
                model, images, samples, args.warmup, args.iters, args.device, args.amp
            )
            mode = mode or resolved
            results[run_id]["rounds_ms"].append(round(times, 4))
            results[run_id]["round_peak_mem_mb"].append(round(peak_mb, 2))
            print(f"[bench_latency] round {round_index + 1}/{args.rounds} "
                  f"{run_id}: median {statistics.median(times):.3f} ms, "
                  f"peak {peak_mb:.1f} MiB")

    for run_id, _model, _images, _samples, info in built:
        pooled = [value for rnd in results[run_id]["rounds_ms"] for value in rnd]
        round_medians = [statistics.median(rnd) for rnd in results[run_id]["rounds_ms"]]
        summary = summarise(pooled)
        summary.update(
            {
                "round_medians_ms": [round(value, 4) for value in round_medians],
                "median_of_round_medians_ms": round(statistics.median(round_medians), 4),
                "peak_mem_mb": max(results[run_id]["round_peak_mem_mb"]),
                "peak_mem_rounds_mb": results[run_id]["round_peak_mem_mb"],
                "forward_mode": mode,
                **info,
            }
        )
        results[run_id] = summary

    reference_id = args.reference or built[0][0]
    reference = results.get(reference_id)
    comparison = {"reference": reference_id}
    candidate_ids = [run_id for run_id, *_ in built if run_id != reference_id]
    for candidate_id in candidate_ids:
        candidate = results[candidate_id]
        if not reference:
            continue
        latency_ratio = candidate["median_ms"] / reference["median_ms"]
        memory_ratio = (candidate["peak_mem_mb"] / reference["peak_mem_mb"]
                        if reference["peak_mem_mb"] else float("nan"))
        comparison = {
            "reference": reference_id,
            "candidate": candidate_id,
            "latency": {
                "reference_median_ms": reference["median_ms"],
                "candidate_median_ms": candidate["median_ms"],
                "increase_fraction": round(latency_ratio - 1.0, 6),
                "increase_percent": round((latency_ratio - 1.0) * 100, 4),
            },
            "memory": {
                "reference_peak_mem_mb": reference["peak_mem_mb"],
                "candidate_peak_mem_mb": candidate["peak_mem_mb"],
                "increase_fraction": (
                    round(memory_ratio - 1.0, 6)
                    if memory_ratio == memory_ratio else None
                ),
                "increase_percent": (
                    round((memory_ratio - 1.0) * 100, 4)
                    if memory_ratio == memory_ratio else None
                ),
            },
            "latency_within_tolerance": (
                latency_ratio - 1.0 <= ACCEPT_MAX_LATENCY_INCREASE
            ),
            "memory_within_tolerance": (
                memory_ratio == memory_ratio
                and memory_ratio - 1.0 <= ACCEPT_MAX_MEMORY_INCREASE
            ),
        }

    payload = {
        "protocol": {
            "batch_size": args.batch_size,
            "input_size": [args.size, args.size],
            "warmup_iters": args.warmup,
            "timed_iters": args.iters,
            "rounds": args.rounds,
            "alternative": "models are benchmarked in alternating order each round",
            "precision": "amp-fp16" if args.amp else "fp32",
            "cudnn_benchmark": True,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "forward_mode": mode,
        },
        "models": results,
        "comparison": comparison,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    ensure_dir(osp.dirname(osp.abspath(args.out)))
    write_json(args.out, payload)

    print("\n[bench_latency] results (median ms / peak MiB)")
    for run_id, entry in results.items():
        print(f"  {run_id}: {entry['median_ms']} ms / {entry['peak_mem_mb']} MiB")
    if comparison.get("candidate"):
        print(f"  latency  : {comparison['latency']['increase_percent']:+.4f} % "
              f"(limit {ACCEPT_MAX_LATENCY_INCREASE * 100:.2f} %) -> "
              f"{'PASS' if comparison['latency_within_tolerance'] else 'FAIL'}")
        print(f"  memory   : {comparison['memory']['increase_percent']} % "
              f"(limit {ACCEPT_MAX_MEMORY_INCREASE * 100:.2f} %) -> "
              f"{'PASS' if comparison['memory_within_tolerance'] else 'FAIL'}")
    print(f"\n[bench_latency] -> {osp.abspath(args.out)}")


if __name__ == "__main__":
    main()
