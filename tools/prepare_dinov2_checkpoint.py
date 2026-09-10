#!/usr/bin/env python3
"""Download and convert the DINOv2-L backbone required by experiment 01."""

import argparse
import os
import os.path as osp
import sys
import urllib.request

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))


OFFICIAL_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
    "dinov2_vitl14_pretrain.pth"
)
DEFAULT_RAW = "checkpoints/dinov2_vitl14_pretrain.pth"
DEFAULT_OUTPUT = "checkpoints/dinov2_converted_512x512.pth"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=OFFICIAL_URL)
    parser.add_argument("--raw", default=DEFAULT_RAW,
                        help="downloaded/original ViT-L/14 checkpoint")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="converted checkpoint used by experiment 01")
    return parser.parse_args()


def load_vitl14(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} does not contain a state dict")

    patch = checkpoint.get("patch_embed.proj.weight")
    pos = checkpoint.get("pos_embed")
    has_last_block = any(key.startswith("blocks.23.") for key in checkpoint)
    if (
        patch is None
        or pos is None
        or patch.ndim != 4
        or patch.shape[0] != 1024
        or pos.shape[-1] != 1024
        or not has_last_block
    ):
        patch_shape = tuple(patch.shape) if patch is not None else None
        pos_shape = tuple(pos.shape) if pos is not None else None
        raise ValueError(
            f"{path} is not the required DINOv2-L/14 checkpoint "
            f"(expected embed_dim=1024 and 24 blocks; "
            f"patch_embed={patch_shape}, pos_embed={pos_shape}). "
            "DINOv2-S/B weights are not shape-compatible."
        )
    return checkpoint


def download(url, destination):
    os.makedirs(osp.dirname(osp.abspath(destination)), exist_ok=True)
    partial = destination + ".part"
    print(f"[prepare_checkpoint] downloading {url}")
    try:
        urllib.request.urlretrieve(url, partial, _report_progress)
        print()
        os.replace(partial, destination)
    except Exception:
        if osp.exists(partial):
            os.remove(partial)
        raise
    print(f"[prepare_checkpoint] downloaded -> {destination}")


def _report_progress(blocks, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = min(blocks * block_size, total_size)
    percent = downloaded * 100.0 / total_size
    print(
        f"\r[prepare_checkpoint] {percent:5.1f}% "
        f"({downloaded / 1024**2:.1f}/{total_size / 1024**2:.1f} MiB)",
        end="",
        flush=True,
    )


def main():
    args = parse_args()
    if osp.isfile(args.output):
        converted = load_vitl14(args.output)
        patch_size = tuple(converted["patch_embed.proj.weight"].shape[-2:])
        pos_tokens = converted["pos_embed"].shape[1]
        if patch_size == (16, 16) and pos_tokens == 1025:
            print(f"[prepare_checkpoint] already ready: {args.output}")
            return
        raise ValueError(
            f"{args.output} exists but has patch size {patch_size} and "
            f"{pos_tokens} position tokens; move it aside and run again"
        )

    if not osp.isfile(args.raw):
        download(args.url, args.raw)
    else:
        print(f"[prepare_checkpoint] using existing raw checkpoint: {args.raw}")

    checkpoint = load_vitl14(args.raw)
    import torch
    from convert_models.convert_dinov2 import (
        interpolate_patch_embed_,
        interpolate_pos_embed_,
    )

    interpolate_patch_embed_(checkpoint, kernel_conv=16)
    interpolate_pos_embed_(checkpoint, crop_size=(512, 512), kernel_conv=16)

    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    torch.save(checkpoint, args.output)
    # Verify the serialized artifact, not only the in-memory state dict.
    load_vitl14(args.output)
    print(f"[prepare_checkpoint] ready -> {args.output}")


if __name__ == "__main__":
    main()
