# FastFlow4Wan2.1

这是一个面向 Wan2.1 T2V 1.3B 的 FastFlow 即插即用推理脚本。它不训练、不蒸馏，也不修改 Wan DiT 权重；模型结构和 checkpoint 与官方 Wan2.1 完全兼容。

实现依据仓库中的 FastFlow 论文：

- 用最近两次真实速度预测做有限差分，外推被跳过时间步的速度；
- 每个采样时间位置维护一个 UCB bandit，用“跳步收益 - 速度 MSE”更新；
- 外推结束后在终点重新做一次完整 DiT 推理，用真实速度校正 bandit；
- Wan 的 `flow_prediction` 输出直接作为速度场，使用论文中的 Euler 轨迹推进。

与 TeaCache 的区别是：本脚本没有使用 TeaCache 的多项式阈值或残差缓存，而是实现 FastFlow 的速度外推和 bandit 决策。

## 依赖

不需要安装新库。依赖与 Wan2.1 原仓库一致：

```text
torch
diffusers
numpy
tqdm
```

脚本还使用 Wan2.1 仓库自带的 `wan`、`FlowUniPCMultistepScheduler` 和视频保存工具。

## 使用方法

把 `fastflow_generate.py` 放到 Wan2.1 仓库根目录，然后执行：

```bash
python fastflow_generate.py \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --size '832*480' \
  --frame_num 81 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 42 \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --offload_model True \
  --t5_cpu \
  --fastflow_arms 0,2,4,6
```

显存较充足时建议设置 `--offload_model False --t5_cpu False`，可以明显减少 CPU/GPU 搬运开销。

也可以在已有的 Wan2.1 Python 推理代码中直接接入，不需要改模型或 checkpoint：

```python
from fastflow_generate import FastFlowConfig, install_fastflow

pipeline = wan.WanT2V(...)
install_fastflow(
    pipeline,
    FastFlowConfig(
        arms=(0, 2, 4, 6),
        gamma=2.0,
        stats_path="./fastflow_stats.json",
    ),
)
video = pipeline.generate(prompt, sampling_steps=50)
```

## 参数

- `--fastflow_arms 0,2,4,6`：允许的跳步数。论文 50 步设置使用 `0,2,4,6`。
- `--fastflow_gamma 2.0`：UCB 探索系数，论文默认值。
- `--fastflow_mu`：速度收益和误差惩罚的绝对权重。不指定时按论文使用 `mu = max_mse / total_steps` 的尺度归一化形式。
- `--fastflow_stats stats.json`：保存并复用每个时间位置的 bandit 统计。统计文件会随多次生成逐渐收敛，后续 prompt 通常会比第一次更快、更稳定。
- `--fastflow_verbose`：打印每一步的跳步、MSE、奖励和下一步目标跳步。
- `--fastflow_error_low 0.03` / `--fastflow_error_high 0.10`：首次运行热启动时，相对速度误差低于/高于阈值会增大/减小目标跳步。

首次运行没有历史 bandit 统计时，脚本会从中间跳步（默认 `2`）开始，并根据真实外推误差自动增减目标跳步。这是为了让单次运行也能获得加速；当 `--fastflow_stats` 中已有统计时，会优先使用论文的 UCB 选择。

## 说明

- 该脚本只支持 `t2v-1.3B`，没有加入 i2v 的 CLIP 条件和条件 latent 分支。
- FastFlow 的轨迹推进使用 Euler 更新。Wan 原版默认 UniPC 是高阶求解器，数值轨迹不会与 UniPC 逐步一致，这是 FastFlow 论文和本实现的设计选择。
- 跳步数越大速度越快，但复杂视频更容易出现运动或细节退化。建议先用 `0,2,4,6`，质量不足时改为 `0,1,2,3`。
- 实际端到端加速比会受 VAE 解码、T5 编码、模型 offload 和 GPU 型号影响；脚本日志中的倍率只统计 DiT 推理次数。
