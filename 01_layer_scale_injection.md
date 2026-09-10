# 实验 01：Cloud-Adapter 分层残差缩放（Layer-Scale Injection）

## 1. 目标与硬性结论

本实验验证：Cloud-Adapter 在 24 个 Transformer block 后直接执行 `x + CrossAttention(x, context)`，可能对冻结 VFM 特征产生过强或不均匀扰动；为每个交互层增加可学习残差缩放系数 `alpha_l`，能否以几乎零参数代价稳定地提高云分割精度。

本实验不是“把代码跑通”即成功。只有第 10 节全部主验收条件同时满足，才记为成功；任一主条件未满足即记为失败。

## 2. 上游代码与固定基线

- 代码库：[XavierJiezou/Cloud-Adapter](https://github.com/XavierJiezou/Cloud-Adapter)
- 主干：冻结 DINOv2-L
- 分割头：Mask2Former
- 数据集：CloudSEN12_High_L1C，官方 train/val/test 划分
- 输入：`512 x 512`
- 训练：40,000 iter，batch size 4，AdamW，初始学习率 `1e-4`，weight decay `0.05`
- warm-up：前 1,000 iter，从 `1e-6` 线性升高；之后 PolyLR，power `0.9`
- 原论文 seed：42；本实验统一使用 `13、42、3407` 三个随机种子
- 论文参考结果：DINOv2-L Cloud-Adapter 在 L1C 上 `74.18 mIoU`

每次运行必须记录：

```text
git commit
完整 config
数据集文件列表及其 SHA256
PyTorch/CUDA/mmcv/mmseg 版本
GPU 型号
随机种子
训练参数量
峰值显存
训练总时长
单图推理延迟
每类 IoU、mIoU、mAcc、mDice、aAcc
```

## 3. 基线复现门槛

先运行未改动 Cloud-Adapter：

| run | seed | 用途 |
|---|---:|---|
| B0-13 | 13 | 配对比较 |
| B0-42 | 42 | 论文复现检查 |
| B0-3407 | 3407 | 配对比较 |

定义三次 test mIoU 的均值为 `B0`。

基线有效的必要条件：

1. `abs(mIoU_B0-42 - 74.18) <= 0.30`。
2. 三个种子之间的 mIoU 极差不超过 `0.80`。
3. 三次均无 NaN、梯度爆炸、数据漏用或 test-set 调参。

基线未通过时，本实验停止并标记为“环境/复现失败”，不得继续宣称结构实验失败或成功。

## 4. 网络改动

主要改动文件：

```text
cloud_adapter/models/backbones/cloud_adapter.py
configs/adapter/<本实验配置文件>.py
```

官方 `ConvnextInteractiveModule.forward` 的核心形式是：

```python
return (x + self.attn(x, cache)).permute(1, 0, 2)
```

修改为：

```python
class ScaledConvnextInteractiveModule(nn.Module):
    def __init__(self, emd_dim, context_dim, rank_dim, init_scale=0.1,
                 scale_type="scalar"):
        super().__init__()
        self.attn = CrossAttention(emd_dim, context_dim, rank_dim=rank_dim)
        if scale_type == "scalar":
            self.alpha = nn.Parameter(torch.tensor(float(init_scale)))
        elif scale_type == "channel":
            self.alpha = nn.Parameter(
                torch.full((emd_dim,), float(init_scale))
            )
        else:
            raise ValueError(scale_type)

    def forward(self, x, cache, index):
        # 保持原实现中的 cache 选择、插值、flatten 和 batch-first 转换。
        delta = self.attn(x_bnc, cache_bnc)
        if self.alpha.ndim == 0:
            out = x_bnc + self.alpha * delta
        else:
            out = x_bnc + self.alpha.view(1, 1, -1) * delta
        return out.permute(1, 0, 2)
```

要求：

- 24 个交互层各有独立 `alpha_l`，禁止全层共享一个标量。
- DINOv2 参数继续完全冻结。
- `alpha_l` 必须进入 optimizer、checkpoint 和仅保存 adapter 权重的逻辑。
- 不允许同时修改 loss、数据增强、decoder、rank 或训练轮数。
- 默认主实验使用 layer-wise scalar；channel-wise 只作为扩展消融。

## 5. 配置项

新增配置：

```python
cloud_adapter_config = dict(
    ...,
    use_layer_scale=True,
    layer_scale_type="scalar",
    layer_scale_init=0.1,
)
```

至少执行以下单种子筛选，全部只看 validation：

| ID | scale 类型 | 初始化值 | seed |
|---|---|---:|---:|
| S0 | 无缩放，原模型 | 等价于 1.0 固定 | 42 |
| S1 | scalar | 0.0 | 42 |
| S2 | scalar | 0.01 | 42 |
| S3 | scalar | 0.1 | 42 |
| S4 | scalar | 1.0 | 42 |
| S5 | channel | 0.1 | 42 |

按 validation mIoU 选择一个最佳 scalar 初始化值，记为 `LS*`。禁止根据 test 结果选择初始化值。随后只对 `LS*` 跑三随机种子 test。

## 6. 训练和日志要求

每 500 iter 记录：

- 24 个 `alpha_l` 的数值、均值、标准差和最大绝对值；
- 每层 `||alpha_l * delta_l||_2 / (||x_l||_2 + 1e-6)`；
- `alpha_l` 梯度范数；
- 总 loss、分类 loss、mask loss、dice loss；
- learning rate 和峰值显存。

自动中止条件：

- 任一 `alpha_l`、loss 或梯度出现 NaN/Inf；
- 连续 1,000 iter 出现 `|alpha_l| > 5`；
- 残差比值连续 1,000 iter 大于 1.0；
- 冻结 DINOv2 中任何参数产生非零梯度。

## 7. 评价指标

主指标：test mIoU，三种子均值。

辅助指标：

- Clear Sky、Thin Cloud、Thick Cloud、Cloud Shadow 的 IoU；
- mDice；
- 三种子标准差；
- 参数增量；
- batch=1、预热 100 次后运行 500 次的中位推理延迟；
- 相同 batch 和输入尺寸下的峰值显存。

延迟测试须锁定同一 GPU、同一 precision、同一 cudnn 配置；每个模型交替测试三轮，报告三轮中位数。

## 8. 需要提交的结果表

```text
| Model | Seed | mIoU | CRS IoU | TNC IoU | TKC IoU | CDS IoU |
| Params | Peak Mem | Latency | alpha mean/std | Residual-ratio mean |
```

另外提交：

- 24 层最终 `alpha_l` 折线图；
- 24 层残差比值折线图；
- 三随机种子逐一差值 `mIoU_LS* - mIoU_B0`。

## 9. 统计规则

- 所有增量均做配对比较：seed 13 对 seed 13，以此类推。
- 报告三种子均值、标准差和 95% bootstrap CI（按测试图像进行 10,000 次配对重采样）。
- 本实验不要求三种子 t-test 显著，因为 n=3 太小；最终显著性在实验 06 统一检查。

## 10. 硬性成功标准

以下条件必须全部满足：

1. `mean(mIoU_LS*) >= B0 + 0.25`。
2. 至少 2/3 个种子的 mIoU 高于其配对基线。
3. 任一种子的退化不得超过 `0.15 mIoU`。
4. 任一类别 IoU 的三种子均值退化不得超过 `0.40`。
5. 新增可训练参数不超过 `0.001M`；scalar 主实验理论上仅增加约 24 个参数。
6. 中位推理延迟增幅不超过 `1%`，峰值显存增幅不超过 `1%`。
7. DINOv2 保持完全冻结，且 checkpoint 能独立恢复全部 `alpha_l`。

任一条件不满足，本实验结论为失败，不得用“趋势为正”替代成功。

## 11. 失败诊断

- 精度无提升且 `alpha_l` 基本保持初值：检查参数是否加入 optimizer、是否被 checkpoint/filter 丢弃。
- `alpha_l` 快速变大：减小 adapter 学习率到主学习率的 0.1 倍，但这属于新实验，不能覆盖原失败记录。
- 只有 channel-wise 有效：记录 scalar 实验失败；channel-wise 可作为后续独立方向。
- seed=42 提升但三种子均值未过线：判定不可复现，实验失败。
- 速度超线：检查是否因日志 hook、同步计时或额外 tensor copy；修复计时后重测，但不得修改网络以规避结果。

## 12. 交付物

```text
configs/experiment_01/*.py
work_dirs/experiment_01/<run_id>/config.py
work_dirs/experiment_01/<run_id>/metrics.json
work_dirs/experiment_01/<run_id>/env.txt
work_dirs/experiment_01/summary.csv
work_dirs/experiment_01/alpha_layers.png
work_dirs/experiment_01/residual_ratio.png
EXPERIMENT_01_REPORT.md
```

