"""Shared paths, run definitions and helpers for experiment 01.

Everything the other scripts need to agree on lives here: where runs are
written, how a run is named, how checkpoints are discovered, and how the
protocol's run matrix is laid out.
"""

import hashlib
import json
import os
import os.path as osp
import subprocess
import sys

# tools/experiment_01/common.py -> repository root
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
CONFIG_DIR = osp.join(REPO_ROOT, "configs", "experiment_01")
EXP_DIR = osp.join(REPO_ROOT, "work_dirs", "experiment_01")

DATASET_ROOT = "data/cloudsen12_high_l1c"
BACKBONE_CHECKPOINT = "checkpoints/dinov2_converted_512x512.pth"
PAPER_MIOU_L1C = 74.18

# protocol §5 / §2
SCREEN_SEED = 42
MAIN_SEEDS = [13, 42, 3407]
BASELINE_SEEDS = [13, 42, 3407]

# protocol §3 gates
BASELINE_GATE_ABS = 0.30  # |mIoU(B0-42) - 74.18|
BASELINE_GATE_RANGE = 0.80  # max - min over the three seeds

# protocol §5 screening variants: (variant, config file, scale type, init)
SCREEN_VARIANTS = [
    ("S0", "s0_baseline_no_scale.py", None, None),
    ("S1", "s1_scalar_init0.py", "scalar", 0.0),
    ("S2", "s2_scalar_init0p01.py", "scalar", 0.01),
    ("S3", "s3_scalar_init0p1.py", "scalar", 0.1),
    ("S4", "s4_scalar_init1p0.py", "scalar", 1.0),
    ("S5", "s5_channel_init0p1.py", "channel", 0.1),
]
SCREEN_BY_VARIANT = {v[0]: v for v in SCREEN_VARIANTS}

MAIN_CONFIG = "main_ls_star.py"

# protocol §10
ACCEPT_MEAN_GAIN = 0.25
ACCEPT_SEED_WINS = 2  # of 3
ACCEPT_MAX_SEED_REGRESSION = 0.15
ACCEPT_MAX_CLASS_REGRESSION = 0.40
ACCEPT_MAX_EXTRA_PARAMS_M = 0.001
ACCEPT_MAX_LATENCY_INCREASE = 0.01  # 1 %
ACCEPT_MAX_MEMORY_INCREASE = 0.01  # 1 %

CLASSES = ["clear", "thick cloud", "thin cloud", "cloud shadow"]
# Short tags used in the results table of protocol §8.
CLASS_TAGS = ["CRS", "TKC", "TNC", "CDS"]


# ---------------------------------------------------------------------------
# run naming
# ---------------------------------------------------------------------------
def baseline_run_id(seed):
    return f"B0_seed{seed}"


def screen_run_id(variant, seed=SCREEN_SEED):
    return f"{variant}_seed{seed}"


def main_run_id(layer_scale_type, layer_scale_init, seed):
    return f"LSstar_{layer_scale_type}_init{_fmt_init(layer_scale_init)}_seed{seed}"


def _fmt_init(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def run_dir(run_id):
    return osp.join(EXP_DIR, run_id)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, payload):
    ensure_dir(osp.dirname(osp.abspath(path)))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_jsonable)
    return path


def read_json(path, default=None):
    if not osp.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value):
    """Fallback encoder for numpy scalars/arrays."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty():
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def python_executable():
    return sys.executable or "python"


# ---------------------------------------------------------------------------
# work_dir discovery
#
# mmengine's layout is not fully stable across versions: depending on the
# version, logs and checkpoints may sit directly in the work_dir or in a
# timestamped subdirectory beneath it. Every discovery helper therefore walks
# the run directory recursively rather than assuming a layout.
# ---------------------------------------------------------------------------
def _walk(run_id):
    root = run_dir(run_id)
    for current, _dirs, files in os.walk(root):
        for name in files:
            yield osp.join(current, name)


def discover_checkpoints(run_id):
    """Return ``{"best": path, "last": path}`` for a completed run.

    ``best_*.pth`` is the best-validation checkpoint written by
    ``CheckpointHook(save_best=['mIoU'])``; ``last`` is the highest-numbered
    plain ``iter_*.pth``. Missing entries are ``None``.
    """
    best, last = None, None
    last_iter = -1
    for path in _walk(run_id):
        name = osp.basename(path)
        if not name.endswith(".pth"):
            continue
        if name.startswith("best_"):
            # keep the most recently written best checkpoint
            if best is None or osp.getmtime(path) > osp.getmtime(best):
                best = path
        elif name.startswith("iter_"):
            try:
                iteration = int(name[len("iter_") : -len(".pth")])
            except ValueError:
                continue
            if iteration > last_iter:
                last_iter, last = iteration, path
    return {"best": best, "last": last, "last_iter": last_iter}


def discover_file(run_id, filename):
    """Newest file with this basename anywhere under the run directory."""
    found = [p for p in _walk(run_id) if osp.basename(p) == filename]
    if not found:
        return None
    return max(found, key=osp.getmtime)


def discover_log(run_id):
    logs = [p for p in _walk(run_id) if p.endswith(".log")]
    if not logs:
        return None
    return max(logs, key=osp.getmtime)


# ---------------------------------------------------------------------------
# variant metadata
# ---------------------------------------------------------------------------
def describe_variant(variant):
    """(scale_type, init) for an S-variant id."""
    entry = SCREEN_BY_VARIANT.get(variant)
    if entry is None:
        raise KeyError(f"unknown variant {variant!r}")
    return entry[2], entry[3]


def load_screening():
    return read_json(osp.join(EXP_DIR, "screening.json"))


def screening_winner():
    """The LS* selection produced by the screening stage, or ``None``."""
    payload = load_screening()
    if not payload or payload.get("winner") is None:
        return None
    return payload["winner"]


class RunSpec:
    """One training-plus-evaluation unit of the experiment."""

    def __init__(self, run_id, stage, config, seed, cfg_options=None, notes="",
                 variant=None, layer_scale_type=None, layer_scale_init=None):
        self.run_id = run_id
        self.stage = stage
        self.config = config
        self.seed = seed
        self.cfg_options = list(cfg_options or [])
        self.notes = notes
        self.variant = variant
        self.layer_scale_type = layer_scale_type
        self.layer_scale_init = layer_scale_init

    @property
    def config_path(self):
        return osp.join(CONFIG_DIR, self.config)

    def as_dict(self):
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "config": self.config,
            "seed": self.seed,
            "cfg_options": self.cfg_options,
            "notes": self.notes,
            "variant": self.variant,
            "layer_scale_type": self.layer_scale_type,
            "layer_scale_init": self.layer_scale_init,
        }


def baseline_specs():
    return [
        RunSpec(
            baseline_run_id(seed),
            "baseline",
            "s0_baseline_no_scale.py",
            seed,
            notes="unmodified Cloud-Adapter",
        )
        for seed in BASELINE_SEEDS
    ]


def screen_specs():
    return [
        RunSpec(
            screen_run_id(variant),
            "screen",
            config,
            SCREEN_SEED,
            notes=f"scale_type={scale_type} init={init}",
            variant=variant,
            layer_scale_type=scale_type,
            layer_scale_init=init,
        )
        for variant, config, scale_type, init in SCREEN_VARIANTS
    ]


def main_specs(layer_scale_type, layer_scale_init):
    cfg_options = [
        f"model.backbone.cloud_adapter_config.layer_scale_type={layer_scale_type}",
        f"model.backbone.cloud_adapter_config.layer_scale_init={layer_scale_init}",
    ]
    return [
        RunSpec(
            main_run_id(layer_scale_type, layer_scale_init, seed),
            "main",
            MAIN_CONFIG,
            seed,
            cfg_options=cfg_options,
            notes=f"LS* = {layer_scale_type} init {layer_scale_init}",
            variant="LS*",
            layer_scale_type=layer_scale_type,
            layer_scale_init=layer_scale_init,
        )
        for seed in MAIN_SEEDS
    ]
