# 第二阶段：官方 OPAL + 本地 Qwen3-32B 编程后端

本模式接入论文作者的 [OPAL](https://github.com/xingjian-bai/opal)，固定提交
`e999c21ee1a05891ea09d7d29bf2cd6597ca686b`（包版本 0.3.10，MIT）。不复制或修改 OPAL 内部实现。
这是将论文核心机制迁移到 TxPert 的实现，不是论文结果的严格复现：任务、Agent 模型、工具接口和计算预算不同。OPAL 能否改善 TxPert 的生物学预测性能必须通过独立、多随机种子的真实数据对照实验确认，代码预检或试运行通过不代表性能提升。

## 实际功能

- 使用原 TxPert 模型、损失、图和 Hydra 配置，以及第一阶段的整数图索引适配、参数分组、验证集隔离。
- 官方 `opal.wrap` 包装 AdamW。Lightning 在每次真实 optimizer update 前通过 `observe(loss)` 将该次更新对应的训练损失交给 OPAL；梯度累积时按 micro-batch 样本数加权汇总。未修改程序前执行原优化器，`keep` 基线不调用 LLM。
- Qwen 可以多轮读取报告、修改 `analysis.py`、`updater.py` 和 `helper_*.py`，运行检查、读取错误并在同一次决策内修复，不再受学习率等四种动作的菜单限制。
- `analysis.py` 自行测量参数/梯度/优化器状态、维护诊断 memory，输出报告或 wake 请求。
- `updater.py` 可以实现不同算法、分组策略、动量处理和设备兼容的精度策略。编程范围遵守 OPAL precision contract，不允许改变模型参数/梯度 dtype 或设备。
- 官方引擎按更新边界加载代码，管理版本、试运行、程序状态、故障恢复、分析 memory 和审计记录。
- `blocking: true` 同步等待；`false` 使用官方异步机制，训练继续，响应在后续更新边界应用。记录请求与实际应用步数；短训练可能在首次响应前结束。

Qwen 后端现在是多轮编程 Agent，使用 OpenAI-compatible chat-completions 接口。每次决策可调用 `list_files`、`read_file`、`write_files`、`run_diagnostics`、`run_preflight` 和 `finish`。模型主动读取官方 inbox/reference、诊断报告和错误日志，在暂存区编辑、测试、修复，最后提交或保持不变。只有最终提交的合格候选进入官方试运行。

默认 `agent_protocol: json` 使用 JSON 工具请求，不要求模型服务安装特定工具解析器；`openai` 使用原生 function calling，需要服务支持对应的工具调用格式。两种传输执行相同的实际工具。每个决策默认最多 12 轮模型请求、32 次工具调用、600 秒；每轮最多输出 8192 token。因此 **500 步决策一次不再等于 500 步只发一次 LLM 请求**。失败修复会增加调用和等待时间，可通过第二阶段 YAML 调整预算。超过预算且尚未提交时保留原程序。

当前提供源码、文件清单、官方提示、参数形状、训练消息和近期决策；报告由 Agent 按需分段读取。上下文超限时按完整工具交互移除最早消息，实际每轮请求和移除计数均保存。仍与论文的通用 coding-agent 工具环境有差异：没有任意 shell，编辑限于两个入口和根目录 `helper_*.py`，诊断工具执行合成 CPU 探针。真实训练的自定义测量通过发布 `analysis.py` 后由官方引擎执行，再反馈给后续决策。这些限制不应表述为工具环境逐项等价。

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

共享验证桥同时计算验证 MSE 和 Pearson Δ，并通过 `opt.log(..., show_agent=True)` 将两者交给 OPAL。默认以 Pearson Δ 为主要指标（最大化），MSE 为辅助诊断；`selection_metric` 可设为 `pearson_delta` 或 `mse`，最佳 checkpoint 按该指标选择。每次 optimizer update 的训练 loss 通过 OPAL 官方 Lightning 接口 `observe(loss)` 传递；梯度累积时使用样本数加权的 micro-batch 平均损失。上层新增的 evidence 调用门控、hold 退避和确定性 Pearson 反馈**只用于 bounded 模式**；OPAL 保留自己的调度与程序试运行机制。注意 `trial_steps` 只用于检测候选 updater 的运行故障，不是验证集性能验收或质量回滚。新版训练状态与目标配置会使旧 OPAL checkpoint 的实验身份发生变化，应开启新实验，不做静默迁移。

`share_code: true` 会把只读模型/训练源码提供给 Agent，以便其结合 TxPert 架构分析更新程序。默认 Qwen endpoint 为本机地址；如果改用远程服务，应先确认可以向该服务发送项目源码。若不希望共享源码，将 `share_code` 设为 `false`。

启动脚本会先估计完整 optimizer-step 预算。`interval_steps` 指定计划决策间隔，`every` 指定官方检查间隔；前者必须是后者的整数倍。`max_decisions` 限制包括诊断唤醒在内的总决策次数，因此不能简单把它当作每 N 步必有一次调用。最后一步不安排新的计划决策。

若改为 5000 步：复制本目录 YAML，设 `interval_steps: 5000`，通过 `--opal-config` 传入。如果要异步运行，将同一文件中的 `blocking` 改为 false。计划步数、实际调用与应用步数均以运行日志为准。

## 候选代码如何进入训练

1. 保存模型身份及每轮完整请求、响应、工具参数与结果。
2. Agent 读取训练报告并暂存源码。校验路径、大小、语法、必需接口与 import 范围；将官方 boundary 检查的发现作为拒绝条件。
3. Agent 调用预检，在独立 CPU 子进程中用与真实模型同形状的随机参数/梯度执行有限步检查。限制超时、CPU 时间、内存、文件大小，不向子进程传递认证环境变量。
4. 检查 build 是否改坏共享对象、analysis 是否修改参数/梯度/优化器、参数 shape/dtype/device、数值有限性、参数覆盖和状态保存恢复。结果和诊断报告立即反馈到同一次决策；Agent 可读日志、修改和重测。
5. 最终 `finish(submit)` 要求预检通过的代码哈希与当前暂存版本一致。编辑后必须重测；临时诊断通过不能代替候选预检。中断、超时、未最终提交不会发布暂存代码。
6. 提交后官方 OPAL 在真实训练更新边界构造 updater，进入 `trial_steps` 窗口。

预检使用合成 CPU 状态，不是当前完整训练状态；不能证明 GPU 性能、所有分支正确或预测效果提高。大模型超过 `preflight_max_numel` 时拒绝候选，需要操作者调整 CPU 预算。失败日志立即反馈，修复仍须通过检查。

**子进程和静态检查不是安全沙箱。** 当前执行环境无法启用所测的 bwrap 网络隔离。与官方 OPAL 一样，实际生成的 Python 程序在训练进程内运行，拥有该进程权限；静态启发式可被间接访问绕过。仅应在隔离的科研训练环境使用代码模式，且不放无关凭证。`--engine opal` 明确选择执行生成代码；默认 `--engine bounded` 保持有限动作模式。代码模式没有宣称具备任意 Python 的强隔离。

## 故障恢复边界

遵循固定版本官方机制：

- 新程序构造或第一次更新失败，且 CPU 快照在 `snapshot_limit_gb` 预算内：恢复参数、梯度及优化器相关状态，回到上一个程序。
- 试运行窗口中较晚的更新失败：官方切回旧程序，但不能保证撤销已部分执行的更新。
- 快照超过预算、程序耗尽资源、在实际训练中死循环等情况，不保证自动完整恢复。预检只能减少此类情况。
- 这不是“验证 MSE 退化就完整重放训练”。任意算法切换后的质量改进需靠对照实验判断。

## 日志和 checkpoint

上层 `manifest.json` 继续保存解析配置、源文件哈希、包版本、声明的模型版本和提示词。额外保存：

- `opal-settings.json`：上游提交、版本、计划决策步数、参数形状、源码共享设置与 checkpoint 选择指标；
- `opal/workspace/analysis.py` 和 `updater.py`：官方工作区（以 programs 版本和事件日志确认实际应用）；
- `opal/programs/vNNN/`：代码版本；
- `opal/decisions/NNN/turns/TTT/`：每轮 `request.json`、`response.json`、`tool-result.json` 或 `error.json`；
- 同一决策的 `checks/CCC/`：每次检查的完整候选与 `preflight.log`；`transcript.json` 保存交互，`qwen-result.json` 保存决策、轮次、工具/检查次数、服务返回的累计 token 用量和上下文省略计数；没有 usage 的服务不能据此计算真实 token 成本；
- 兼容摘要 `qwen-request.json` 是首轮请求，`qwen-response.json` 是最后响应，`preflight.log` 是最后检查日志；以逐轮和逐次记录为准；
- 官方 `events.jsonl`、训练/指标日志和报告，记录实际应用和失败恢复；
- `opal-checkpoints/step-*.ckpt`、最佳/last/final checkpoint：模型、OPAL 代码/优化器/分析状态、Lightning 训练状态和 RNG。

```bash
python -m reactive_optimizer.train --engine opal \
  --revision "$MODEL_REVISION" \
  --resume reactive_optimizer/runs/opal-qwen/checkpoints/last.ckpt \
  --output reactive_optimizer/runs/opal-qwen-resume seed=1 trainer.max_epochs=200
```

使用相同 engine、模型、数据、Agent/OPAL 配置、提示词和权重 revision 恢复；只加载可信 checkpoint，其中包含可执行代码。不要把 bounded 模式 checkpoint 当成 OPAL checkpoint。建议在 epoch 边界恢复；不保证数据游标、GPU 运算或 LLM 输出逐位重现。退出时可能有未应用的异步回答，以实际应用日志为准。

多轮后端改变了提示词和配置身份，旧单轮 OPAL checkpoint 会被配置一致性检查拒绝，不能直接当作同一实验续跑。当前正在运行的 bounded 实验不自动切换模式；新 OPAL 对照使用独立输出目录。

## 验证范围

```bash
python -m unittest discover -s reactive_optimizer/opal_mode/tests -p 'test_*.py' -v
python -m unittest discover -s reactive_optimizer/tests -v
```

CPU 验证使用原 GAT 配置和合成小图/表达数据。独立进程 20 步 keep 对照逐步损失、最终权重完全一致。动态代码测试使用确定性 fixture，验证改写两个程序、AdamW→SGD 算法切换、状态保存恢复、预检拒绝和官方首次试运行故障恢复。多轮测试覆盖同次决策失败修复、辅助模块、临时诊断、JSON/原生工具协议、预算耗尽、取消、过期预检和路径限制。不是 Qwen3-32B 成功生成程序或真实生物数据收益的证明。

下一步在你的 GPU 上先跑 keep 基线和少量实际 Qwen 决策，核查代码版本确实应用、延迟和显存；再进行固定数据划分、相同预算、多种子的 bounded / OPAL / 常规优化对照。论文中的图像/语言建模结果不能直接作为 TxPert 收益证据。
