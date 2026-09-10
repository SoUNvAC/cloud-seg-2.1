# Experiment 01 / LS* — the layer-scale variant selected by validation screening.
#
# `layer_scale_type` / `layer_scale_init` default to the protocol §5 defaults
# (scalar, 0.1), i.e. the same model as `s3_scalar_init0p1.py`. run_matrix.py
# records which screening variant actually won and, when it is not S3, passes
# the winning values here:
#
#     --cfg-options \
#         model.backbone.cloud_adapter_config.layer_scale_type=scalar \
#         model.backbone.cloud_adapter_config.layer_scale_init=0.01 \
#         randomness.seed=13
#
# The three main runs use seeds 13 / 42 / 3407.
_base_ = [
    "../adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py",
    "_base_experiment_01.py",
]

model = dict(
    backbone=dict(
        cloud_adapter_config=dict(
            use_layer_scale=True,
            layer_scale_type="scalar",
            layer_scale_init=0.1,
        ),
    ),
)
