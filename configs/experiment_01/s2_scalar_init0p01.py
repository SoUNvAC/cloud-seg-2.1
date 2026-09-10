# Experiment 01 / S2 — layer-wise scalar alpha, initialised at 0.01.
_base_ = [
    "../adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py",
    "_base_experiment_01.py",
]

model = dict(
    backbone=dict(
        cloud_adapter_config=dict(
            use_layer_scale=True,
            layer_scale_type="scalar",
            layer_scale_init=0.01,
        ),
    ),
)
