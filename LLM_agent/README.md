# LLM_agent

这是独立于原有 TxPert 训练和推理脚本的真实 Agent 项目。它只读取仓库已有的数据和模型，不包含示例 KG、示例文献或演示预测器。

流程分成四层：

1. 从 KG 和 RAG 的 JSONL 数据源检索证据；
2. 让 LLM 生成模型选择和证据摘要；
3. 把摘要和证据传给数值 forward model；
4. 输出统一的 JSON 结果。

运行会安装 `requirements.txt` 并下载 Qwen 与 BGE-M3 到 `LLM_agent/models/`：

```bash
bash LLM_agent/run.sh
```

运行时会加载以下已有资源：

- `cache/K562_single_cell_line/de_adata_test.h5ad`
- `cache/checkpoints/K562_unseen_pert_gat.ckpt`
- `data/graphs/string/v11.5.parquet`
- `data/graphs/go/go_top_50.csv`
- 仓库已有的 `README.md` 和 `cache/Readme.md`

Qwen 只接收当前扰动基因的结构化摘要、真实 KG 邻居和 RAG 文档片段，完整表达向量始终由 TxPert forward model 处理。

准备一个任务文件（例如 `task.json`，由你自己的任务提供，不由项目生成示例数据）：

```bash
python -m LLM_agent.run --input task.json --output LLM_agent/runs/prediction.json
```

默认使用 `K562_unseen_pert_gat.ckpt`。如果要使用自己在 `forward_model/` 训练的模型，将 `forward.backend` 改为 `custom_forward`，并使用更新后的训练脚本生成带有 `model_spec.edge_indices` 的 checkpoint；两种 checkpoint 不能混用。

如果使用你自己在 `forward_model/` 训练的 checkpoint，将 `forward.backend` 改为 `custom_forward`，并填写 checkpoint 路径；项目会拒绝缺少图结构元数据的权重，避免产生不可复现的错误预测。
