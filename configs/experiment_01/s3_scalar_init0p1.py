# Experiment 01 / S3 — layer-wise scalar alpha, initialised at 0.1.
#
# This is the protocol §5 default and the value used for the main experiment
# unless the validation screening picks another one.
_base_ = "_base_experiment_01.py"

model = dict(
    backbone=dict(
        cloud_adapter_config=dict(
            use_layer_scale=True,
            layer_scale_type="scalar",
            layer_scale_init=0.1,
        ),
    ),
)
