# 第二阶段：官方 OPAL + 本地 Qwen3-32B 编程后端

本模式接入论文作者的 [OPAL](https://github.com/xingjian-bai/opal)，固定提交
`e999c21ee1a05891ea09d7d29bf2cd6597ca686b`（包版本 0.3.10，MIT）。不复制或修改 OPAL 内部实现。
这是将论文核心机制迁移到 TxPert 的实现，不是论文结果的严格复现：任务、Agent 模型、工具接口和计算预算不同；尚未做真实 Qwen3/GPU/生物数据效果实验。

## 实际功能

- 使用原 TxPert 模型、损失、图和 Hydra 配置，以及第一阶段的整数图索引适配、参数分组、验证集隔离。
- 官方 `opal.wrap` 包装 AdamW。未修改程序前执行原优化器，`keep` 基线不调用 LLM。
- Qwen 可以返回完整 `analysis.py`、`updater.py`，不再受学习率等四种动作的菜单限制。
- `analysis.py` 自行测量参数/梯度/优化器状态、维护诊断 memory，输出报告或 wake 请求。
- `updater.py` 可以实现不同算法、分组策略、动量处理和设备兼容的精度策略。编程范围遵守 OPAL precision contract，不允许改变模型参数/梯度 dtype 或设备。
- 官方引擎按更新边界加载代码，管理版本、试运行、程序状态、故障恢复、分析 memory 和审计记录。
- `blocking: true` 同步等待；`false` 使用官方异步机制，训练继续，响应在后续更新边界应用。记录请求与实际应用步数；短训练可能在首次响应前结束。

Qwen 后端使用 OpenAI-compatible chat-completions JSON 接口，每次决策一轮代码生成；它没有 Claude Code 的 shell/文件浏览工具。后端主动提供当前代码、OPAL 系统提示、参数形状/初始优化器配置、训练消息、最近失败记录，以及限长的 inbox/reference 文本。超出上下文预算的附加文件可能省略；完整实际请求会保存。不允许创建额外 helper 文件。这比官方通用 coding agent 的工具范围窄，但确实支持重写诊断与更新算法，而非把代码替换成固定动作菜单。

## 安装和运行

从仓库根目录，在已有 TxPert Python 3.12 环境中执行：

```bash
source .venv/bin/activate
# 只增加官方 OPAL；沿用原有 torch/numpy，不升级根依赖。
uv pip install --no-deps -r reactive_optimizer/opal_mode/requirements.txt

# 不调用 LLM：先验证桥接与本机环境。
python -m reactive_optimizer.opal_mode.verify \
  --steps 20 --output reactive_optimizer/runs/opal-parity

# 真实数据、原 GAT 配置，官方 keep 基线；数据需要提前准备。
python -m reactive_optimizer.train --engine opal --baseline \
  --output reactive_optimizer/runs/opal-keep seed=1

# 本地 Qwen3-32B 服务部署方法见上层 README。
# MODEL_REVISION 是实际服务权重的 40 位提交 SHA，不要填写示例占位值。
python -m reactive_optimizer.train --engine opal \
  --revision "$MODEL_REVISION" \
  --output reactive_optimizer/runs/opal-qwen seed=1 trainer.max_epochs=200
```

依然可选 `--config config-gat`、`--agent-config`、`--endpoint`、`--download-data`、Hydra 覆盖。
`--opal-config` 指向第二阶段配置，默认本目录 `config.yaml`。Qwen 身份、endpoint、temperature、HTTP 超时、seed 等来自上层 agent YAML；输出 token 上限来自第二阶段配置，默认 8192。不需要 Claude 账户、Claude API 或额外远程 LLM 调用。

当前 Lightning 桥只支持 **单训练设备、FP32、无外部学习率调度器**；本地 LLM 服务可以另占多张 GPU。原 GAT 配置的调度器为空，可直接使用。其他结构虽然可选择配置，但集成测试仅覆盖 GAT。训练裁剪交给 OPAL 起始 updater，Lightning 的裁剪设为 0，避免裁剪两次。第一阶段的固定验证阈值反馈器不在此模式运行，避免它与生成的更新算法互相覆盖。

启动脚本会先估计完整 optimizer-step 预算。`interval_steps` 指定计划决策间隔，`every` 指定官方检查间隔；前者必须是后者的整数倍。`max_decisions` 限制包括诊断唤醒在内的总决策次数，因此不能简单把它当作每 N 步必有一次调用。最后一步不安排新的计划决策。

若改为 5000 步：复制本目录 YAML，设 `interval_steps: 5000`，通过 `--opal-config` 传入。如果要异步运行，将同一文件中的 `blocking` 改为 false。计划步数、实际调用与应用步数均以运行日志为准。

## 候选代码如何进入训练

1. 保存完整 Qwen 请求、权重 revision/precision 声明和完整响应。
2. 校验 JSON、文件路径、源码大小、语法、必需接口与 import 范围；将官方 boundary 检查的发现作为拒绝条件。
3. 在独立 CPU 子进程中，用与真实模型同形状的随机参数/梯度执行有限步预检。限制超时、CPU 时间、内存、文件大小，不向子进程传递认证环境变量。
4. 预检检查 build 是否改坏共享对象、analysis 是否修改参数/梯度/优化器、参数 shape/dtype/device、数值有限性、参数覆盖、状态可序列化与 load_state_dict 往返。
5. 通过后才写入官方工作目录。官方 OPAL 在真实训练更新边界构造 updater，并进入 `trial_steps` 窗口。

预检使用合成 CPU 状态，不是当前完整训练状态；不能证明 GPU 性能、所有分支正确或预测效果提高。大模型超过 `preflight_max_numel` 时拒绝候选，需要操作者调整 CPU 预算。被拒绝的诊断和失败日志尾部会提供给下一次 Qwen 决策，不会自动强行执行。

**子进程和静态检查不是安全沙箱。** 当前执行环境无法启用所测的 bwrap 网络隔离。与官方 OPAL 一样，实际生成的 Python 程序在训练进程内运行，拥有该进程权限；静态启发式可被间接访问绕过。仅应在隔离的科研训练环境使用代码模式，且不放无关凭证。`--engine opal` 明确选择执行生成代码；默认 `--engine bounded` 保持有限动作模式。代码模式没有宣称具备任意 Python 的强隔离。

## 故障恢复边界

遵循固定版本官方机制：

- 新程序构造或第一次更新失败，且 CPU 快照在 `snapshot_limit_gb` 预算内：恢复参数、梯度及优化器相关状态，回到上一个程序。
- 试运行窗口中较晚的更新失败：官方切回旧程序，但不能保证撤销已部分执行的更新。
- 快照超过预算、程序耗尽资源、在实际训练中死循环等情况，不保证自动完整恢复。预检只能减少此类情况。
- 这不是“验证 MSE 退化就完整重放训练”。任意算法切换后的质量改进需靠对照实验判断。

## 日志和 checkpoint

上层 `manifest.json` 继续保存解析配置、源文件哈希、包版本、声明的模型版本和提示词。额外保存：

- `opal-settings.json`：上游提交、版本、计划决策步数、参数形状与配置；
- `opal/workspace/analysis.py` 和 `updater.py`：官方工作区（以 programs 版本和事件日志确认实际应用）；
- `opal/programs/vNNN/`：代码版本；
- `opal/decisions/NNN/`：官方消息，以及 `qwen-request.json`、`qwen-response.json`、`qwen-result.json`、`candidate.json`、`preflight.log`；
- 官方 `events.jsonl`、训练/指标日志和报告，记录实际应用和失败恢复；
- `opal-checkpoints/step-*.ckpt`、最佳/last/final checkpoint：模型、OPAL 代码/优化器/分析状态、Lightning 训练状态和 RNG。

```bash
python -m reactive_optimizer.train --engine opal \
  --revision "$MODEL_REVISION" \
  --resume reactive_optimizer/runs/opal-qwen/checkpoints/last.ckpt \
  --output reactive_optimizer/runs/opal-qwen-resume seed=1 trainer.max_epochs=200
```

使用相同 engine、模型、数据、Agent/OPAL 配置、提示词和权重 revision 恢复；只加载可信 checkpoint，其中包含可执行代码。不要把 bounded 模式 checkpoint 当成 OPAL checkpoint。建议在 epoch 边界恢复；不保证数据游标、GPU 运算或 LLM 输出逐位重现。退出时可能有未应用的异步回答，以实际应用日志为准。

## 验证范围

```bash
python -m unittest discover -s reactive_optimizer/opal_mode/tests -p 'test_*.py' -v
python -m unittest discover -s reactive_optimizer/tests -v
```

本次 CPU 验证使用原 GAT 配置和合成小图/表达数据。独立进程 20 步 keep 对照逐步损失、最终权重完全一致。动态代码测试使用确定性 fixture，验证改写两个程序、AdamW→SGD 算法切换、状态保存恢复、预检拒绝和官方首次试运行故障恢复。不是 Qwen3-32B 成功生成程序或真实生物数据收益的证明。

下一步在你的 GPU 上先跑 keep 基线和少量实际 Qwen 决策，核查代码版本确实应用、延迟和显存；再进行固定数据划分、相同预算、多种子的 bounded / OPAL / 常规优化对照。论文中的图像/语言建模结果不能直接作为 TxPert 收益证据。
