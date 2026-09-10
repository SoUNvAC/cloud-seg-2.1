# Experiment 01 / S5 — channel-wise alpha, initialised at 0.1.
#
# Extension ablation only (protocol §4: the main experiment is layer-wise
# scalar). Adds 24 x 1024 = 24,576 parameters.
_base_ = "_base_experiment_01.py"

model = dict(
    backbone=dict(
        cloud_adapter_config=dict(
            use_layer_scale=True,
            layer_scale_type="channel",
            layer_scale_init=0.1,
        ),
    ),
)
