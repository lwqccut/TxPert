# LLM_agent

这是独立于原有 TxPert 训练和推理脚本的 Agent 项目。它把流程拆成四层：

1. 从 KG 和 RAG 的 JSONL 数据源检索证据；
2. 让 LLM 生成模型选择和证据摘要；
3. 把摘要和证据传给数值 forward model；
4. 输出统一的 JSON 结果。

第一次完整运行会安装 `requirements.txt` 并下载 Qwen 与 BGE-M3 到 `LLM_agent/models/`：

```bash
bash LLM_agent/run.sh
```

真实运行会加载原 TxPert 预处理后的 `cache/K562_single_cell_line` 数据和 checkpoint；Qwen 只接收当前扰动基因的结构化摘要、KG 邻居和 RAG 证据，完整表达向量始终由 TxPert forward model 处理。

也可以传入请求和证据文件：

```bash
python -m LLM_agent.cli --input request.json --kg kg.jsonl --rag docs.jsonl --output result.json
```

默认使用 `K562_unseen_pert_gat.ckpt`。如果要使用自己在 `forward_model/` 训练的模型，将 `forward.backend` 改为 `custom_forward`，并使用更新后的训练脚本生成带有 `model_spec.edge_indices` 的 checkpoint；两种 checkpoint 不能混用。

如果已经在 Python 中加载了 `ConditionedMultiViewPerturbationModel`，可以使用
`TxPertForwardAdapter(model)` 将它接到 `LLMAgent`，适配器会负责 tensor 转换和推理模式切换。
