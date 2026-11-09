# TxPert + Qwen3-32B 受约束优化控制器

本目录实现 Reactive Optimizers / OPAL 思路的第一阶段：Qwen3-32B 观察训练统计，输出受约束 JSON 动作。复用 `gspp.models.txpert.TxPert`、`PertPredictor.training_step`、原损失函数、数据模块、图和 `configs/config-*.yaml`。未添加 foundation model，也未替换为仓库中的 `forward_model`。

这不是完整 OPAL 复现，尚未证明提高真实数据预测性能。本页的 bounded 模式采用同步调用、单训练设备，便于审计；不支持 DDP/FSDP。模型服务可另占多张 GPU。

## 行为

AdamW 按 `cell_encoder`、`graph_encoder`、`decoder` 和必要的 `other` 参数分组。保留原配置的初始学习率和权重衰减。默认 GAT 配置没有调度器。自动反馈开启时禁止同时使用学习率调度器，避免恢复旧学习率时覆盖调度器更新；`--baseline` 或 `feedback_enabled: false` 仍支持原调度器。

每 `interval_steps` 个 optimizer step（不是微批次）检查一次是否允许调用；自动反馈模式还需要近期有效验证基准、没有待评估动作且冷却期已结束。允许调用时收集各组梯度范数、参数范数、当前批次训练损失、最近验证 MSE 及其 step、近期决策。LLM 可以：

- `hold`：保持不变；
- `scale_lr`：单组学习率乘以限定倍数，结果也必须在绝对边界内；
- `set_weight_decay`：设置单组权重衰减；
- `set_clip_norm`：设置全局梯度范数裁剪阈值。

未知动作、额外字段、越界数值、NaN/Infinity、非法 JSON 和连接失败不会改变优化器。原始响应（若收到）及失败类型保留在日志中。没有 `exec`、动态代码导入或源码重写。

为隔离测试集，新入口覆盖原来在验证结束时读取测试集的行为，只用验证表达 MSE。训练目标保持不变。该 MSE 用于控制器和最佳 checkpoint 选择，不代表差异表达或跨细胞泛化表现。未执行最终测试集评估；应在方法选择完成后另行做固定测试评测。

运行时适配还将原 TxPert 构造出的浮点图边索引转换为整数（PyG 要求），不改图连接或网络结构。默认开启 norm=1 的梯度裁剪；如原 Trainer 配置已指定 norm 裁剪，则优先采用原值。这些适配同样用于 `--baseline` 对照。

## 依赖和模型服务

训练使用仓库原 Python 3.12 环境，不增加训练依赖，不修改根锁文件。原依赖安装命令为 `uv sync --frozen --python 3.12`。真实训练需要原来的 Zenodo 数据和 STRING/GO 图。训练从随机初始化开始；只有 `--resume` 才加载训练检查点。无需预训练 TxPert checkpoint 即可从头训练。

Qwen3-32B 使用独立 vLLM 等 OpenAI-compatible 服务。建议单独建立服务环境，避免 vLLM 的 PyTorch 版本与 TxPert 的锁定版本冲突。以下是部署示例，需要在有 GPU 的机器上自行验证所用 vLLM 版本支持 Qwen3、JSON 输出和 `chat_template_kwargs`：

```bash
# MODEL_REVISION 必须是实际下载/部署的 Qwen3-32B 权重的 40 位提交 SHA。
export MODEL_REVISION='<实际权重提交SHA>'
# 在独立模型服务环境中执行；自行分配 GPU，避免与训练争用显存。
vllm serve Qwen/Qwen3-32B \
  --revision "$MODEL_REVISION" \
  --served-model-name Qwen/Qwen3-32B \
  --dtype bfloat16 --host 127.0.0.1 --port 8000
```

模型服务不由训练脚本安装或启动。多 GPU 服务可按实际硬件加 `--tensor-parallel-size`；服务与训练的 `CUDA_VISIBLE_DEVICES` 分别配置。量化时修改 agent YAML 中的 `precision`，记录真实设置。模型 revision 和 precision 是操作者声明，客户端不能从 chat-completions 响应独立验证权重；请同时保留服务启动命令和部署版本。

如果服务需要鉴权，在运行环境设置 `REACTIVE_LLM_API_KEY`；不写入 YAML、命令行参数或实验日志。endpoint 不允许嵌入凭证。日志只发送汇总统计，不传原始表达矩阵。

## 运行

从仓库根目录执行：

```bash
source .venv/bin/activate

# 先查看最终配置；只覆盖 mode=train，其他模型/数据配置仍来自原文件。
python -m reactive_optimizer.train --config config-gat \
  --baseline --output reactive_optimizer/runs/config-check --print-config

# 对照：相同参数分组、裁剪、验证、检查点开销，不调用 LLM。
python -m reactive_optimizer.train --config config-gat \
  --baseline --output reactive_optimizer/runs/baseline-seed1 seed=1

# Reactive：固定模型 revision，默认 endpoint 为本机 8000/v1。
python -m reactive_optimizer.train --config config-gat \
  --revision "$MODEL_REVISION" \
  --output reactive_optimizer/runs/reactive-seed1 seed=1
```

`--output` 必须是新目录，防止覆盖日志。通过 `--agent-config path.yaml` 使用修改过的控制器配置。Hydra 覆盖如 `trainer.max_epochs=20`、`datamodule.batch_size=32` 继续有效；`--config config-exphormer` 等可以选择其他原配置，但当前集成测试只验证 GAT。

如缺少原数据，可加 `--download-data` 调用仓库原 Zenodo 下载器。它可能下载大量文件，应先确认网络和磁盘容量。默认不自动下载。

为了只确认真实 LLM 接口工作，可复制 agent YAML，将 `interval_steps` 暂设为 2、`feedback_enabled` 暂设为 false，然后运行 `trainer.max_steps=4`。自动反馈需要足够长的训练和真实验证循环，不能用这段短跑验证。检查 `summary.json` 的 attempts/accepted/failed 和实际 `response.json`；短跑只验证接线，不能判断有效性。正式对照恢复较长间隔，并固定相同训练预算和种子。

## 审计和恢复

每个 run 保存：

- `manifest.json`：原配置解析结果、agent 配置、声明的 Qwen revision/precision、代码 commit/哈希、环境包版本、提示词哈希、恢复来源及其 SHA256；
- `prompt.txt`：本次使用的系统提示词；
- `step-.../request.json`：完整消息、采样参数和观测；
- `response.json`：完整模型响应，保留服务返回的模型名、usage 等信息；
- `decision.json`：接受/拒绝状态、动作、调用耗时、修改前后的超参数；
- `before.ckpt`、`after.ckpt`：动作前后完整模型、AdamW moments、scheduler、Lightning loops、控制器历史、Python/NumPy/Torch/CUDA RNG；
- `checkpoints/last.ckpt`、最佳验证 checkpoint、`final.ckpt`、`summary.json`；
- `feedback/*.json` 和对应 `.ckpt`：每次有效反馈的比较值、结论、撤销范围和处理后的训练状态。

完整快照会占磁盘；每次决策有两份，每次有效反馈也保存一份，正式实验需估算空间。`summary.json` 的计数包含恢复前的历史；若 accepted=0，不能将该运行当作有效 LLM 干预实验。

```bash
python -m reactive_optimizer.train --config config-gat \
  --revision "$MODEL_REVISION" \
  --resume reactive_optimizer/runs/reactive-seed1/checkpoints/last.ckpt \
  --output reactive_optimizer/runs/reactive-seed1-resume seed=1
```

只加载可信 checkpoint。恢复时必须使用相同模型结构、数据划分、agent 配置、提示词、权重 revision 和 baseline 模式；控制器会检查其配置/提示词身份。优先在 epoch 边界恢复：原数据加载器没有可恢复游标，mid-epoch 恢复不保证相同样本顺序。保存 RNG/温度为 0 也不保证 GPU 运算或服务输出逐位一致。审计文件用于解释和复现实验条件，不宣称逐位确定性。

## 测试

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" XDG_CACHE_HOME="$PWD/.cache" \
  python -m unittest discover -s reactive_optimizer/tests -v
```

测试用合成小图/表达样本运行真实 TxPert GAT、原损失与 Lightning optimizer/checkpoint 循环；LLM 使用明确标注的确定性传输 fixture。覆盖动作边界、失败保持配置、参数分组、原 scheduler、训练参数更新和 checkpoint 恢复。不下载真实生物数据，不把 fixture 视为 Qwen3-32B 的效果验证。

## 自动反馈与撤销（默认开启）

流程：近期验证基准 → 一个 LLM 动作 → 观察新验证结果 → 保留或撤销 → 冷却期 → 下一动作。
反馈判断由确定性程序完成，LLM 会在下次请求中收到最近的 `feedback_history`。

默认设置：

```yaml
feedback_enabled: true
feedback_min_steps: 500                 # 动作后至少训练这么多步，再作常规评估
feedback_baseline_max_age_steps: 5000   # 过旧的基准不能启动新动作
feedback_improvement: 0.01              # 验证 MSE 比基准降低至少 1%
feedback_regression: 0.05               # 验证 MSE 比基准增加至少 5%
feedback_patience: 2                    # 连续两次满足条件
feedback_max_evaluations: 4             # 最多观察四次符合条件的验证
feedback_cooldown_steps: 500            # 结束一次试验后暂停新干预
```

参考值为动作前最近一次正常验证的 MSE，比较采用 `(当前MSE - 基准MSE) / max(abs(基准MSE), 1e-12)`。这些阈值是可配置的初始策略，不是经过生物数据验证的最佳值。

- 连续两次改善至少 1%：`kept`，保留设置。
- 连续两次退化至少 5%：`rollback_regression`，撤销修改。
- 四次评估仍无连续改善或退化结论：`rollback_inconclusive`，保守撤销。
- 动作之后出现非有限或负 MSE：`rollback_nonfinite`，不等待最小观察步数，立即撤销；同一步的验证不计入。
- sanity check、同一步重复验证以及等待步数不足的普通验证不计入反馈次数。没有验证指标时不会调用 Agent；动作待评估时也不叠加新动作。
- 训练结束仍没有充分反馈时，`summary.json.pending_trial` 明确保留为未决，不标为改善，也不会擅自宣称试验通过。

**撤销仅恢复该动作修改的学习率、权重衰减或裁剪阈值。模型权重、AdamW 动量和训练步数不会回退。** 因此无法消除此前不良更新，也不能修复已经 NaN 的权重；完整回退需停止后从动作前的 `before.ckpt` 恢复。自动在 Lightning 训练循环内倒退样本游标/全模型状态不属于当前实现。

观察记录、连续计数、基准、待评估动作和冷却期会进入 checkpoint，恢复时继续同一试验；恢复输出写入新 run 目录。旧版本（没有反馈状态字段）的 checkpoint 不支持直接续训新版控制器，必须开启新实验。关闭反馈可运行无自动撤销的对照，但仍需新版 checkpoint。

反馈只观察验证集，不使用测试集。验证 MSE 随训练下降本身不能证明是动作造成的；数据噪声或自然收敛也会影响判断。正式判断价值仍需固定预算、多种子和 baseline 对照。这一反馈机制不自动开启代码修改能力。

## 第二阶段：官方 OPAL 代码模式（独立、显式启用）

现已提供 [opal_mode/README.md](opal_mode/README.md) 中的官方 OPAL 集成与本地 Qwen3-32B 编程后端。
使用 `--engine opal` 才启用 `analysis.py` / `updater.py` 改写；默认仍是本页的 bounded 模式。
OPAL 模式支持多轮读报告、编辑算法与辅助模块、执行诊断/预检、读取错误并在同一次决策内修复，以及代码版本、官方试运行与同步/异步决策。每次决策可能包含多次 LLM 请求；不会同时启用本页固定阈值的超参数撤销逻辑。

实现可供实验，不代表效果已验证，也没有自动升级开关。正式比较应使用相同数据划分和训练预算、至少三个种子，报告固定测试集扰动/差异表达指标、运行时间、token、失败次数。不能仅因候选代码通过预检或训练 loss 下降就判定有效。
