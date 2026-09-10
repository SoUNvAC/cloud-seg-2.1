# 实验 01：Layer-Scale Injection — 实施报告

方案依据：`01_layer_scale_injection.md`（以下简称"协议"）。
数值约定沿用仓库既有惯例：**分割指标一律以百分数记录；凡未从日志或实验输出中
实际取得的数据一律标为"未记录"，不做推测。**

---

## 1. 一句话结论

**环境建立、训练入口、监控、评测和断点续跑代码已修复并通过静态与 dry-run
验证；实验本身在本机无法运行，因此协议 §10 的七条主验收条件目前全部为
"未记录"，本次不构成成功，也不构成失败。**

判定依据（本机实测）：

| 项目 | 本机状态 |
|---|---|
| `data/cloudsen12_high_l1c` | 不存在 |
| `checkpoints/dinov2_converted_512x512.pth` | 本机不存在；可用准备脚本下载 ViT-L/14 并转换 |
| GPU | NVIDIA GeForce GTX 1060 6GB（6144 MiB） |
| Conda | 未安装/不在 `PATH` |
| `torch` / `mmengine` / `mmcv` / `mmseg` | 未安装 |
| 目标环境 | Python 3.10 / PyTorch 2.1.2+cu121 / mmcv 2.1.0 |

协议 §2 要求 ≥16 GB 显存（batch 4 × 512×512、DINOv2-L 主干）。本机 6 GB 显存、
且数据集与 VFM 权重均不在本地，**任何一条训练/评测/延迟测量都无法在此完成**。
因此本报告交付的是"可直接在满足条件的训练机上执行的完整实验实现 + 诚实的执行
状态说明"，而不是实验结果。`setup.sh` 未在本机真实安装依赖：当前没有 Conda，
且 6 GB 显存不满足训练条件。

---

## 2. 交付物清单

### 2.1 模型改动（协议 §4：分层残差缩放注入）

| 文件 | 改动 |
|---|---|
| `cloud_adapter/models/backbones/cloud_adapter.py` | 新增 `residual_stats()`；`ConvnextInteractiveModule` 拆出 `_prepare()` 以复用；新增 `ScaledConvnextInteractiveModule`（`x + α_l · CrossAttn(x, context)`）；`CloudAdapter.__init__` 增加 `use_layer_scale` / `layer_scale_type` / `layer_scale_init` 三个参数与构建分支 |

`α_l` 的两种粒度：

* `scale_type="scalar"`：`nn.Parameter(torch.tensor(init))`，每层 1 个参数 → 24 个；
* `scale_type="channel"`：`nn.Parameter(torch.full((emd_dim,), init))`，每层 1024 个 → 24576 个。

每层由独立模块持有，**不存在参数共享/绑定**（§10.7 要求可独立恢复）。

### 2.2 训练期观测与保护（协议 §6）

| 文件 | 作用 |
|---|---|
| `cloud_adapter/hooks/layer_scale.py` | `LayerScaleStatsHook`（每 500 iter 写 `layer_scale_stats.jsonl`）；`LayerScaleGuardHook`（非有限数、阈值连续越界及冻结检查，触发即写 `ABORTED.json` 并停止训练） |
| `cloud_adapter/hooks/__init__.py` | 导出上述两个 hook |
| `cloud_adapter/per_image_metric.py` | `PerImageIoUMetric`，返回每类 IoU，并跨 rank 汇总后落盘每图 `intersect` / `pred_area` / `label_area`，供 §9 配对 bootstrap 使用 |
| `cloud_adapter/__init__.py` | 导入 `PerImageIoUMetric` 完成注册 |

每次记录（24 层 × 每层）包含：`alpha`、`alpha_mean/std/max_abs`、
`alpha_grad_norm`、`x_norm`、`delta_norm`、`scaled_norm`、
`residual_ratio = ‖α_l·δ_l‖₂ / (‖x_l‖₂ + 1e-6)`，以及该步的 loss、lr、峰值显存。
`alpha_grad_norm` 通过 `param.register_hook` 抓取，不依赖优化器内部状态。

保护条件：loss、任一可训练梯度、α 或残差统计出现 NaN/Inf；`alpha_max` 或
`residual_ratio` 连续 `streak_iters` 步超限；以及冻结的 DINOv2 参数出现非零梯度。

本轮修复了三处会污染或中断训练的缺陷：S0 基线的 0 维 α 占位张量无法
`torch.cat`；学习率错误地从不存在的 `runner.get_lr()` 读取；原保护逻辑没有真正
检查可训练参数梯度的 NaN/Inf。

### 2.3 配置（协议 §5）

`configs/experiment_01/`：

| 文件 | 变体 | 说明 |
|---|---|---|
| `_base_experiment_01.py` | — | 共享基类：真实验证集、1,000 iter 线性 warm-up、每图指标、500 iter 日志、两个 hook、`.alpha` 不进 weight decay |
| `s0_baseline_no_scale.py` | S0 | 无缩放（等于原始 Cloud-Adapter） |
| `s1_scalar_init0.py` | S1 | scalar，init 0.0 |
| `s2_scalar_init0p01.py` | S2 | scalar，init 0.01 |
| `s3_scalar_init0p1.py` | S3 | scalar，init 0.1 |
| `s4_scalar_init1p0.py` | S4 | scalar，init 1.0 |
| `s5_channel_init0p1.py` | S5 | channel，init 0.1 |
| `main_ls_star.py` | LS* | 主实验，`layer_scale_type`/`init` 由 `--cfg-options` 注入 |

### 2.4 实验流水线

`tools/experiment_01/`：

| 文件 | 作用 |
|---|---|
| `common.py` | 路径、run 命名、run 矩阵定义、work_dir 递归发现 |
| `collect_env.py` | 环境快照 + 数据集 SHA256 清单（§3 可复现性） |
| `eval_run.py` | 单次评测，指标以 dict 返回（不靠爬日志） |
| `count_params.py` | 按配置实际构建模型统计参数量（§10.5） |
| `collect_run.py` | 把一个 run 的全部产物归一为 `metrics.json` |
| `run_matrix.py` | 全矩阵驱动：`env → baseline → gate → screen → main → bench → verify → collect → analyze` |
| `bench_latency.py` | §7 延迟/显存基准 |
| `verify_checkpoint.py` | §10.7 冻结与 α 往返校验 |
| `make_summary.py` | 生成 `summary.csv` |
| `analyze.py` | 配对统计、bootstrap、绘图、§10 判定 |

流水线特性：每个阶段在其产物已存在时自动跳过（中断后可直接重跑 `all`）；
`--force` 重做、`--only <run_id>` 重做单个 run、`--dry-run` 只打印命令。
不完整 run 若已有中间 checkpoint 会自动用该 checkpoint 续训；只有状态为
`complete` 的 `metrics.json` 才会触发跳过，`ABORTED.json` 则要求先定位原因。
基线 gate 失败会**终止流水线**（符合协议 §3：此时只能记为环境/复现失败，
不得对结构改动下任何结论），除非显式传 `--ignore-gate`，该覆盖会写入
`gate_override.json` 并需在报告中披露。

---

## 3. 必须披露的实现决策

以下四点协议未逐字规定，但会影响结论解读，**执行前需确认**。

### 3.1 验证集与测试集分离（上游 bug 修正）

`configs/_base_/datasets/cloudsen12_high_l1c.py` 的 `val_dataloader` 指向
`img_dir/test`——即上游的"验证"指标就是测试指标。协议 §5 要求在**验证集**上
筛选 LS*、只在主实验报告测试集，若沿用上游配置，筛选就变成在测试集上做选择，
§10 的结论不成立。

因此 `_base_experiment_01.py` 把 `val_dataloader` 改指 `img_dir/val`。

证据：`dataset/cloudsen12_high.py:25-27` 声明 `train_size=8490` / `val_size=535` /
`test_size=975`，并在 `:103` 与 `:57` 两处按 `["train","val","test"]` 三划分处理；
`tools/convert_datasets/create_l1c_l2a.py:31-32,57-59` 确实为三个 phase 分别
生成 `img_dir/<phase>` 与 `ann_dir/<phase>`。**`img_dir/val` 一定存在**，
上游 `val_dataloader` 指向 `test` 属于把验证集写成了测试集。

**这是对上游行为的刻意偏离**，所有实验 01 的 run 都受影响。

*影响*：论文的 74.18 是测试集（且是"最优测试"）数字；本实验报告的 mIoU 是
**最优验证 checkpoint 的测试 mIoU**。两者不完全可比，§3 基线 gate 的
`|mIoU(B0-42) − 74.18| ≤ 0.30` 因此可能因选点规则不同而失败，而不代表复现失败。
每个 run 同时记录 `test_mIoU_final_ckpt`（最终迭代 checkpoint）作为诊断，
报告中需同时给出两者以便区分"复现问题"与"选点规则差异"。

### 3.2 `.alpha` 不参与 weight decay

`_base_experiment_01.py` 中：

```python
alpha_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={".alpha": alpha_multi}))
```

理由：ConvNeXt / DINOv2 对 LayerScale 的 `gamma` 参数一律豁免 weight decay。
若用默认 `decay_mult=1.0`，在 lr=1e-4、wd=0.05、40k iter 下解耦衰减会把 α 每步
乘上 (1 − lr·wd)，累计收缩约 18%，**直接把待测效应往零拉**，与实验目的相反。

配置里已写明一行改回 `decay_mult=1.0` 的方法。**这是本实验最需要确认的一个约定**：
若评审要求"新参数按默认处理"，改动这一行即可，但届时应以同一设定重跑全部 run。

### 3.3 LS* 只在 scalar 变体中选

协议 §5 的措辞把 LS* 定义为"最佳初始化"，而 S5（channel）不是初始化差异。
`run_matrix.select_ls_star()` 因此在 S1–S4 中选优；同时记录 S0（不缩放）与 S5
的验证 mIoU 作为关键诊断：

* 若 **S0 胜过所有 scalar 变体** → 缩放本身无益，§10.1 大概率失败，这是**负结果**，
  必须如实记录；
* 若 S5 明显更好但参数量超 §10.5 上限（24576 > 1000），则说明"按层缩放"的收益
  实际来自"按通道缩放"，属于对协议的重要反馈。

两者都会写进 `work_dirs/experiment_01/screening.json`。

### 3.4 §10.6 的延迟口径

`bench_latency.py` 按 §7 实现：batch=1、100 次预热、500 次计时、3 轮交替、
同一 GPU / precision / cudnn 配置、取中位数、记录峰值显存。
两个模型用同一条前向路径（`predict`，含数据预处理与解码头；若该 mmseg 版本
要求调用方预处理则自动回退），**实际使用的路径会写进 `latency.json` 的
`protocol.forward_mode`**，避免两条路径被悄悄混用。

被比较的是 `B0_seed42` 与 `LS*_seed42` 的最优 checkpoint。

---

## 4. 运行矩阵与执行状态

| 阶段 | run | 种子 | 评测集 | 状态 |
|---|---|---|---|---|
| env | — | — | — | 未执行 |
| baseline | `B0_seed{13,42,3407}` | 13/42/3407 | val + test | 未执行 |
| gate | 协议 §3 | — | test | 未执行 |
| screen | `S0…S5_seed42` | 42 | **仅 val** | 未执行 |
| main | `LSstar_<type>_init<v>_seed{13,42,3407}` | 13/42/3407 | val + test | 未执行 |
| bench | B0-42 / LS*-42 | 42 | — | 未执行 |
| verify | 三个 LS* run | 13/42/3407 | — | 未执行 |
| collect / analyze | 全部 | — | — | 未执行 |

共 3（基线）+ 6（筛选）+ 3（主实验）= **12 次训练**，另加评测、基准与校验。

---

## 5. 协议 §10 验收清单

> 以下每一条均为**未记录**：本机无法产出任何训练/评测数值。

| §10 | 条件 | 结果 | 数值 |
|---|---|---|---|
| 1 | `mean(mIoU_LS*) ≥ B0 + 0.25` | 未记录 | 未记录 |
| 2 | 至少 2/3 种子优于配对基线 | 未记录 | 未记录 |
| 3 | 任一种子退化 ≤ 0.15 mIoU | 未记录 | 未记录 |
| 4 | 任一类别三种子均值退化 ≤ 0.40 | 未记录 | 未记录 |
| 5 | 新增可训练参数 ≤ 0.001M | 未记录 | 设计值：scalar 24 个 / channel 24576 个（**未由实际构建验证**） |
| 6a | 中位延迟增幅 ≤ 1% | 未记录 | 未记录 |
| 6b | 峰值显存增幅 ≤ 1% | 未记录 | 未记录 |
| 7 | DINOv2 全冻结且 checkpoint 可独立恢复全部 α_l | 未记录 | 未记录 |

§3 基线 gate：**未记录**（`|mIoU(B0-42) − 74.18| ≤ 0.30`、三种子极差 ≤ 0.80）。

配对统计（§9）：三种子配对差值、10,000 次按测试图像的配对 bootstrap 95% CI、
`bootstrap.json`、`summary.csv`、`acceptance.json`、`acceptance.md`、
`alpha_layers.png`、`residual_ratio.png` —— 全部为**未记录**。

按协议 §1 的判据：**"只有第 10 节全部主验收条件同时满足，才记为成功；
任一主条件未满足即记为失败。"** 当前既未满足也未失败，状态是**未执行**。
`analyze.py` 在数据缺失时一律 fail-closed（判定为 FAIL 并注明
"missing ..."），不会把缺失误报成通过。

---

## 6. 如何执行

在 Linux/WSL 训练机（建议 ≥16 GB NVIDIA GPU）上，从仓库根目录执行：

```bash
# 0. 一键创建/更新 ca21 Conda 环境并做导入/算子自检
sh setup.sh
conda activate ca21

# 可用其他环境名：CLOUD_ADAPTER_ENV=myenv sh setup.sh

# 1. 准备 DINOv2-L/14 权重（首次会从 Meta 官方地址下载并转换）
python tools/prepare_dinov2_checkpoint.py

# 2. 确认数据集与 VFM 权重就位
#    data/cloudsen12_high_l1c/{img_dir,ann_dir}/{train,val,test}
#    checkpoints/dinov2_converted_512x512.pth

# 3. 先只跑 dry-run，确认完整命令矩阵与路径无误
python tools/experiment_01/run_matrix.py --dry-run all

# 4. 全流程（含 §3 基线 gate；gate 失败会自动停下）
python tools/experiment_01/run_matrix.py all
```

中断后重跑 `all` 会跳过已完成的步骤。单步执行：

```bash
python tools/experiment_01/run_matrix.py baseline   # 三个基线
python tools/experiment_01/run_matrix.py gate       # §3 复现 gate
python tools/experiment_01/run_matrix.py screen     # S0..S5，选 LS*
python tools/experiment_01/run_matrix.py main       # LS* × 3 种子
python tools/experiment_01/run_matrix.py bench      # §7 延迟/显存
python tools/experiment_01/run_matrix.py verify     # §10.7
python tools/experiment_01/run_matrix.py collect
python tools/experiment_01/run_matrix.py analyze
```

单次训练（不经流水线）：

```bash
python tools/train.py configs/experiment_01/main_ls_star.py \
  --cfg-options \
    randomness.seed=42 \
    model.backbone.cloud_adapter_config.layer_scale_type=scalar \
    model.backbone.cloud_adapter_config.layer_scale_init=0.01
```

**执行前请先确认第 3 节的四点**，尤其是 3.1（验证集口径影响 §3 gate 的可比性）
与 3.2（`.alpha` 是否豁免 weight decay）。

---

## 7. 未记录数据总清单

以下数据协议明确要求，但本次**均未取得**，报告中不得以任何形式推测填充：

* 全部 12 次训练的 mIoU / aAcc / mAcc / mDice / 各类别 IoU（val 与 test）；
* 三种子均值、标准差、95% bootstrap CI；
* §3 基线 gate 的三个数值（B0-42 的 mIoU、与 74.18 的差值、三种子极差）；
* S0–S5 的验证 mIoU 与 LS* 的选择结果；
* 24 层 α_l 的最终值、均值/标准差/最大绝对值、梯度范数；
* 24 层的残差比 `‖α_l δ_l‖/‖x_l‖`（整体、末段、逐层）；
* 训练峰值显存、单次训练墙钟时间；
* 推理中位延迟与峰值显存（B0 与 LS*），以及两者增幅；
* 实际新增可训练参数量（设计值为 24，未经实际构建验证）；
* `alpha_layers.png`、`residual_ratio.png` 两张图。

---

## 8. 修复记录与验证边界

本轮修复：

* 新增兼容 POSIX sh/Bash 的幂等 `setup.sh`（默认环境名 `ca21`），固定 Python
  3.10、PyTorch 2.1.2+CUDA 12.1、
  mmcv/mmengine/mmseg/mmdet 兼容组合；`xformers==0.0.23.post1` 与 PyTorch 2.1.2
  精确绑定，避免 pip 解析时升级 PyTorch；安装末尾执行 `pip check`、版本断言、
  `mmcv.ops` 与项目导入自检。
* 固定 `setuptools==81.0.0` 并在自检中显式导入 `pkg_resources`。setuptools 82
  起不再附带该模块，而当前 OpenMMLab 依赖链仍需要它；已有 `ca21` 环境再次执行
  `sh setup.sh` 即会自动降级并修复。
* 补充 `future==1.0.0` 和 `tensorboard==2.21.0`。项目的
  `TensorboardVisBackend` 在 Runner 初始化时会导入这两项；安装后自检现在
  会实际导入 `torch.utils.tensorboard.SummaryWriter`，防止训练启动后才发现缺包。
* 新增 `tools/prepare_dinov2_checkpoint.py`，从 Meta 官方地址下载
  DINOv2-L/14，校验 1024 维、24 层结构后，生成实验需要的
  `dinov2_converted_512x512.pth`。`run_matrix.py` 也会在训练前检查该文件；
  若只找到 DINOv2-S（384 维、12 层），会明确报告架构不兼容，不会建议误改名。
* `tools/train.py` 支持 `--resume` 自动或指定 checkpoint，且 `--amp` 能处理配置中
  省略 `optim_wrapper.type` 的常见写法；`tools/dist_train.sh` 改用 PyTorch 2.x 的
  `torch.distributed.run` 并补齐参数引用和错误退出。
* 实验配置补回协议规定但上游配置遗漏的 1,000 iter warm-up：从 `1e-6` 线性升至
  `1e-4`，再执行到 40,000 iter 的 PolyLR。
* 修复 S0 日志崩溃、LR 记录、全量梯度有限性检查、backbone `train()` 返回值与
  `state_dict()` 默认签名。
* 修复 `PerImageIoUMetric`：每类 IoU 现在进入指标 dict；每图统计随
  `BaseMetric.results` 跨 rank 汇总，且显式从 CPU tensor 转成 NumPy，避免丢样本或
  CUDA tensor 转换错误。
* 修复流水线把 `incomplete` 错当成已完成的问题；中断 run 会从最新 checkpoint
  续训，评测失败也会向上返回失败状态；完整 dry-run 可继续展示 LS* 后续阶段，
  且不再创建 run 目录或写 `gate_override.json`。

已执行的无数据验证：

* `python -m compileall -q cloud_adapter tools configs dataset hugging_face`：通过；
* `python tools/experiment_01/run_matrix.py --dry-run all`：退出码 0，训练、验证、
  测试、LS* 主实验、基准、checkpoint 校验、收集和分析命令均可生成；执行后确认
  `work_dirs/experiment_01` 不存在（无写入副作用）；
* `python tools/experiment_01/analyze.py --help`：退出码 0；
* `python tools/prepare_dinov2_checkpoint.py --help`：退出码 0；
* 在 DINOv2-L 权重缺失时执行 `run_matrix.py baseline`：在 Runner 构建前
  fail-fast，列出所需架构和准备命令，且不创建实验目录；
* `_current_lr` 对 MMEngine 的 `{'lr': [value]}` 和 list 两种返回形式的单元用例：
  通过；
* `sh -n setup.sh`、`bash -n setup.sh` 与 `bash -n tools/dist_train.sh`：通过；
* 使用不执行安装的伪 `conda` 命令走完 `setup.sh` 控制流：退出码 0；
* `git diff --check`：通过。

未执行：`sh setup.sh` 的真实安装、DINOv2-L 的实际下载/转换、模型构建、
前向/反向、训练与评测。本机没有 Conda、PyTorch、数据集和 backbone checkpoint，
且 GTX 1060 6 GB 不满足协议训练显存要求。
