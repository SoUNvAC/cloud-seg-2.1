# Shared settings for experiment 01 (layer-scale injection).
#
# Single inheritance chain: this file overrides the upstream baseline config,
# and each variant config (s0..s5, main) inherits THIS file. The overrides
# below win over the upstream defaults (child overrides base in mmengine).
#
# Do NOT list this file as a sibling of the adapter config in a variant's
# `_base_` — mmengine forbids two sibling bases from defining the same key
# ("Duplicate key is not allowed among bases").

_base_ = "../adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py"

crop_size = (512, 512)

# ---------------------------------------------------------------------------
# A genuine validation split.
#
# `configs/_base_/datasets/cloudsen12_high_l1c.py` points `val_dataloader` at
# `img_dir/test`, so upstream the "validation" metric is the test metric. The
# protocol screens layer-scale initialisations on validation and only reports
# test for the selected variant, which is only meaningful if validation is an
# independent split. CloudSEN12_High_L1C ships train/val/test (8490/535/975),
# so validation now reads `img_dir/val`.
# ---------------------------------------------------------------------------
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=crop_size),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CLOUDSEN12HIGHL1CDataset",
        data_root="data/cloudsen12_high_l1c",
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
        pipeline=val_pipeline,
    ),
)

# `PerImageIoUMetric` behaves exactly like `IoUMetric` but additionally writes
# per-image intersect/pred_area/label_area arrays, which is what the paired
# bootstrap of protocol §9 needs. run_matrix.py overrides `per_image_path` with
# a path inside the run's work_dir; the relative default keeps a manual
# `tools/test.py` invocation useful.
val_evaluator = dict(
    type="PerImageIoUMetric",
    iou_metrics=["mIoU", "mDice", "mFscore"],
    per_image_path="per_image_val.npz",
)

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CLOUDSEN12HIGHL1CDataset",
        data_root="data/cloudsen12_high_l1c",
        data_prefix=dict(img_path="img_dir/test", seg_map_path="ann_dir/test"),
        pipeline=val_pipeline,
    ),
)
test_evaluator = dict(
    type="PerImageIoUMetric",
    iou_metrics=["mIoU", "mDice", "mFscore"],
    per_image_path="per_image_test.npz",
)

# ---------------------------------------------------------------------------
# Protocol §6: log every 500 iterations.
# ---------------------------------------------------------------------------
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=500, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=1,
        save_best=["mIoU"],
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

custom_hooks = [
    dict(type="LayerScaleStatsHook", interval=500),
    dict(
        type="LayerScaleGuardHook",
        alpha_max=5.0,
        ratio_max=1.0,
        streak_iters=1000,
        frozen_check_interval=500,
    ),
]

# Fixed training protocol. The upstream adapter config only contains PolyLR,
# while the experiment protocol explicitly requires a 1,000-iteration linear
# warm-up from 1e-6 to the base learning rate (1e-4).
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.01,
        begin=0,
        end=1000,
        by_epoch=False,
    ),
    dict(
        type="PolyLR",
        eta_min=0,
        power=0.9,
        begin=1000,
        end=40000,
        by_epoch=False,
    ),
]

# Overridden per run by run_matrix.py via --cfg-options randomness.seed=<seed>.
randomness = dict(seed=42)

# ---------------------------------------------------------------------------
# alpha_l must reach the optimizer.
#
# `decay_mult=0.0` keeps alpha out of AdamW weight decay. This follows the
# LayerScale convention used by ConvNeXt / DINOv2 for exactly this kind of
# residual scale parameter: with `decay_mult=1.0` the decoupled decay shrinks
# alpha by (1 - lr*wd) every step (about 18% over 40k iterations at
# lr=1e-4, wd=0.05), which pulls the scale towards zero and works directly
# against the effect under test. Set `decay_mult=1.0` to reproduce the plain
# default treatment of a new parameter instead.
# ---------------------------------------------------------------------------
alpha_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            ".alpha": alpha_multi,
        },
    ),
)
