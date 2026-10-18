# LLM_agent

这是独立于原有 TxPert 训练和推理脚本的真实 Agent 项目。它只读取仓库已有的数据和模型，不包含示例 KG、示例文献或演示预测器。

流程分成四层：

1. 从 KG 和 RAG 的 JSONL 数据源检索证据；
2. 让 LLM 生成模型选择和证据摘要；
3. 把摘要和证据传给数值 forward model；
4. 输出统一的 JSON 结果。

路径一一对应关系如下：

| 用途 | 现有路径 |
| --- | --- |
| 原始预处理表达数据 | `cache/K562_single_cell_line/de_adata_test.h5ad` |
| 数据集划分 | `cache/K562_single_cell_line/splits/` |
| 基因全集 | `data/master_gene_set_sorted.csv` |
| 扰动 ID 映射 | `data/gears_gene_set.csv` |
| STRING 图 | `data/graphs/string/v11.5.parquet` |
| GO 图 | `data/graphs/go/go_top_50.csv` |
| Pharos 知识等级 | `data/pharos_mean_rank.csv` |
| 原 TxPert checkpoint | `cache/checkpoints/` |
| Agent 预测输出 | `LLM_agent/runs/` |

运行会安装 `requirements.txt` 并下载 Qwen 与 BGE-M3 到 `LLM_agent/models/`：

```bash
bash LLM_agent/run.sh
```

运行时会加载以下已有资源：

- `cache/K562_single_cell_line/de_adata_test.h5ad`
- `cache/checkpoints/K562_unseen_pert_gat.ckpt`
- `data/graphs/string/v11.5.parquet`
- `data/graphs/go/go_top_50.csv`
- `data/master_gene_set_sorted.csv`、`data/gears_gene_set.csv` 和 `data/pharos_mean_rank.csv`

当前仓库没有真实论文语料，因此默认不伪造 RAG 文档；Qwen 只接收当前扰动基因的结构化摘要和真实 KG 邻居。获得真实论文语料后，再把路径加入 `retrieval.rag_paths`。

适合加入 RAG 的文献应是目标基因的原始扰动研究、同一细胞类型的转录组研究、基因调控网络研究和通路数据库的可引用版本，并保留 PMID/DOI、物种、细胞类型、实验方法和年份。综述可以作为背景，但不应替代原始证据。

`agent.explain_mechanism` 默认是 `false`。此模式下 Qwen 只做知识检索后的证据归纳、相关性判断和置信度评估；机制解释由研究者完成，Forward Model 仍只负责数值预测。

准备一个任务文件（例如 `task.json`，由你自己的任务提供，不由项目生成示例数据）：

```bash
python -m LLM_agent.run --input task.json --output LLM_agent/runs/prediction.json
```

默认使用 `K562_unseen_pert_gat.ckpt`。如果要使用自己在 `forward_model/` 训练的模型，将 `forward.backend` 改为 `custom_forward`，并使用更新后的训练脚本生成带有 `model_spec.edge_indices` 的 checkpoint；两种 checkpoint 不能混用。

如果使用你自己在 `forward_model/` 训练的 checkpoint，将 `forward.backend` 改为 `custom_forward`，并填写 checkpoint 路径；项目会拒绝缺少图结构元数据的权重，避免产生不可复现的错误预测。
