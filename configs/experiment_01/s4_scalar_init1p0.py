# Experiment 01 / S4 — layer-wise scalar alpha, initialised at 1.0.
#
# Numerically identical to S0 at initialisation, but alpha is trainable, so the
# pair (S0, S4) isolates the effect of *learning* the scale from the effect of
# merely attenuating the residual.
_base_ = [
    "../adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py",
    "_base_experiment_01.py",
]

model = dict(
    backbone=dict(
        cloud_adapter_config=dict(
            use_layer_scale=True,
            layer_scale_type="scalar",
            layer_scale_init=1.0,
        ),
    ),
)
