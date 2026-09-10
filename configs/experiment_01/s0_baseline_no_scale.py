# Experiment 01 / S0 — unmodified Cloud-Adapter (no residual scaling).
#
# This is the reference model: `use_layer_scale=False` builds the stock
# `ConvnextInteractiveModule`, so the forward pass is
# `x + attn(x, cache)` exactly as upstream. The diagnostics hooks still record
# per-layer residual statistics with an implicit alpha of 1.0, which keeps the
# S0 curves in the §8 plots comparable with the scaled variants.
_base_ = [
    "../adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py",
    "_base_experiment_01.py",
]

model = dict(
    backbone=dict(
        cloud_adapter_config=dict(
            use_layer_scale=False,
        ),
    ),
)
