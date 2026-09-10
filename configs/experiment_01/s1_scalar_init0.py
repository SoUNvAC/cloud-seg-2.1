# Experiment 01 / S1 — layer-wise scalar alpha, initialised at 0.0.
#
# The adapter starts as the identity map on the frozen features and has to
# learn the residual from scratch.
_base_ = "_base_experiment_01.py"

model = dict(
    backbone=dict(
        cloud_adapter_config=dict(
            use_layer_scale=True,
            layer_scale_type="scalar",
            layer_scale_init=0.0,
        ),
    ),
)
