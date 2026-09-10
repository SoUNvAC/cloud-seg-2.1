"""Write the environment record required by protocol §2.

Produces ``work_dirs/experiment_01/env.txt`` plus a cached
``dataset_manifest.json`` holding the SHA256 of every dataset file. Hashing
20,000 files is slow enough to be worth caching, so the manifest is reused
unless ``--force`` is given.

Runs on a machine without torch/GPU too: every unavailable field is reported as
``unavailable`` rather than being guessed at.
"""

import argparse
import os
import os.path as osp
import platform
import sys
from datetime import datetime

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from common import (  # noqa: E402
    BACKBONE_CHECKPOINT,
    DATASET_ROOT,
    EXP_DIR,
    PAPER_MIOU_L1C,
    REPO_ROOT,
    ensure_dir,
    git_commit,
    git_dirty,
    read_json,
    sha256_file,
    write_json,
)

SPLITS = ["train", "val", "test"]


def _module_version(name):
    try:
        module = __import__(name)
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unavailable ({type(exc).__name__})"
    return getattr(module, "__version__", "unknown")


def _gpu_info():
    try:
        import torch
    except Exception:
        return ["unavailable (torch not importable)"]
    if not torch.cuda.is_available():
        return ["unavailable (no CUDA device visible)"]
    lines = [f"device_count: {torch.cuda.device_count()}"]
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        lines.append(
            f"gpu{index}: {torch.cuda.get_device_name(index)} | "
            f"{props.total_memory / 1024**3:.1f} GiB | "
            f"compute capability {props.major}.{props.minor}"
        )
    return lines


def _torch_info():
    try:
        import torch
    except Exception:
        return ["unavailable (torch not importable)"]
    lines = [
        f"torch: {torch.__version__}",
        f"torch.version.cuda: {torch.version.cuda}",
        f"torch.backends.cudnn.version: {torch.backends.cudnn.version()}",
    ]
    try:
        lines.append(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    except Exception:
        pass
    return lines


def _dataset_split_counts():
    counts = {}
    for split in SPLITS:
        img_dir = osp.join(REPO_ROOT, DATASET_ROOT, "img_dir", split)
        ann_dir = osp.join(REPO_ROOT, DATASET_ROOT, "ann_dir", split)
        counts[split] = {
            "img": len(os.listdir(img_dir)) if osp.isdir(img_dir) else None,
            "ann": len(os.listdir(ann_dir)) if osp.isdir(ann_dir) else None,
        }
    return counts


def build_manifest(force=False):
    """SHA256 of every dataset file, cached under work_dirs/experiment_01."""
    cache_path = osp.join(EXP_DIR, "dataset_manifest.json")
    cached = read_json(cache_path)
    if cached and not force:
        return cached, cache_path, True

    root = osp.join(REPO_ROOT, DATASET_ROOT)
    files = []
    if osp.isdir(root):
        for current, _dirs, names in os.walk(root):
            for name in sorted(names):
                full = osp.join(current, name)
                files.append(
                    {
                        "path": osp.relpath(full, REPO_ROOT).replace("\\", "/"),
                        "size": osp.getsize(full),
                        "sha256": sha256_file(full),
                    }
                )
    manifest = {
        "dataset_root": DATASET_ROOT,
        "num_files": len(files),
        "splits": _dataset_split_counts(),
        "files": files,
    }
    write_json(cache_path, manifest)
    return manifest, cache_path, False


def render(manifest, manifest_path, manifest_cached):
    lines = []
    add = lines.append
    add("# Experiment 01 environment record")
    add(f"generated: {datetime.now().isoformat(timespec='seconds')}")
    add("")
    add("## Code")
    add(f"repo_root: {REPO_ROOT}")
    add(f"git_commit: {git_commit()}")
    add(f"git_dirty: {git_dirty()}")
    add("")
    add("## Python")
    add(f"python: {platform.python_version()} ({sys.executable})")
    add(f"platform: {platform.platform()}")
    for name in ("torch", "mmengine", "mmcv", "mmseg", "mmdet", "timm", "einops",
                 "numpy", "scipy", "matplotlib"):
        add(f"{name}: {_module_version(name)}")
    add("")
    add("## CUDA / GPU")
    for line in _torch_info():
        add(line)
    for line in _gpu_info():
        add(line)
    add("")
    add("## Dataset")
    add(f"dataset_root: {DATASET_ROOT}")
    add(f"num_files: {manifest['num_files']}")
    for split in SPLITS:
        counts = manifest["splits"].get(split, {})
        add(f"  {split}: img={counts.get('img')} ann={counts.get('ann')}")
    add(f"manifest: {osp.relpath(manifest_path, REPO_ROOT).replace(chr(92), '/')}"
        f"{' (reused from cache)' if manifest_cached else ' (recomputed)'}")
    add("")
    add("## Reference")
    add(f"paper reference mIoU (DINOv2-L Cloud-Adapter, L1C): {PAPER_MIOU_L1C}")
    backbone_path = osp.join(REPO_ROOT, BACKBONE_CHECKPOINT)
    if osp.exists(backbone_path):
        add(f"backbone: {BACKBONE_CHECKPOINT} sha256={sha256_file(backbone_path)}")
    else:
        add(f"backbone: {BACKBONE_CHECKPOINT} MISSING")
    add("")
    add("## Seeds")
    add("baseline: 13, 42, 3407")
    add("screening: 42")
    add("main (LS*): 13, 42, 3407")
    add("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="recompute the dataset SHA256 manifest")
    args = parser.parse_args()

    ensure_dir(EXP_DIR)
    manifest, manifest_path, cached = build_manifest(force=args.force)
    text = render(manifest, manifest_path, cached)
    out_path = osp.join(EXP_DIR, "env.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    print(f"[collect_env] wrote {out_path}")


if __name__ == "__main__":
    main()
