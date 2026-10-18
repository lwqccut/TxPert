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

服务器没有 GPU 或暂时不想下载权重时，可以先用离线冒烟模式验证流程：

```bash
python -m LLM_agent.run --offline
```

也可以传入请求和证据文件：

```bash
python -m LLM_agent.cli --input request.json --kg kg.jsonl --rag docs.jsonl --output result.json
```

`DemoForwardModel` 只是运行示例。接入真实 TxPert 时，将它替换为调用 checkpoint 的函数即可；原有训练代码不会被修改。

如果已经在 Python 中加载了 `ConditionedMultiViewPerturbationModel`，可以使用
`TxPertForwardAdapter(model)` 将它接到 `LLMAgent`，适配器会负责 tensor 转换和推理模式切换。
