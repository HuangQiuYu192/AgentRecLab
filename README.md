# AgentRecLab

指令感知推荐的可审计离线实验框架。iAgent、i²Agent 直接包装官方类；MemRec
调用官方 graph、domain rules、packer、manager、reranker 和 Stage-W。
默认采用合成数据和 mock LLM，不需要 API Key。MARS 为 `experimental/not_ready`。

**当前验证是基础设施与上游适配器验证，不是论文指标复现。**
真实 InstructRec 对齐和付费实验尚未完成；详见 [风险清单](docs/reproduction_risks.md)。

## 10 分钟 quickstart

Python 3.11+；本机已在 Python 3.12.14 验证。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe scripts/setup_upstream.py
.\.venv\Scripts\agentreclab.exe doctor
.\.venv\Scripts\agentreclab.exe data fixture
.\.venv\Scripts\agentreclab.exe validate --manifest datasets/fixtures/books/dataset_manifest.json
.\.venv\Scripts\agentreclab.exe run --config configs/runs/smoke_books.yaml --model iagent
.\.venv\Scripts\agentreclab.exe run --config configs/runs/smoke_books.yaml --model i2agent
.\.venv\Scripts\agentreclab.exe run --config configs/runs/smoke_books.yaml --model memrec
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

Linux/macOS 使用 `.venv/bin/python`、`.venv/bin/agentreclab`。
`requirements.lock` 保存本次验证依赖；可先 `pip install -r requirements.lock` 再安装项目。
上游首次下载需要联网；之后 fixture/mock 的运行和测试均不访问外部 API。
同一 run_id 再次运行必须加 `--resume`；配置、代码、数据变化会拒绝恢复。

```powershell
.\.venv\Scripts\agentreclab.exe run --config configs/runs/smoke_books.yaml --model iagent --resume
.\.venv\Scripts\agentreclab.exe run --config configs/runs/books_1k.yaml --model memrec --dry-run
.\.venv\Scripts\agentreclab.exe evaluate --run outputs/runs/smoke_books_iagent
.\.venv\Scripts\agentreclab.exe compare --runs outputs/runs/smoke_books_iagent --runs outputs/runs/smoke_books_i2agent --runs outputs/runs/smoke_books_memrec --paired-only
.\.venv\Scripts\agentreclab.exe report --run outputs/runs/smoke_books_memrec --out outputs/reports/smoke_books_memrec.md
```

`compare` 以重复 `--runs` 指定多个目录。候选顺序、标签、case IDs、历史、训练快照
或 provider 条件不一致时，严格配对比较会拒绝运行。

## 数据与协议

原始数据来源：[iAgent 官方 README 的 InstructRec 下载链接](https://github.com/agiresearch/iAgent#-dataset-preparation)。
下载时自行确认数据访问与许可；两份上游代码在固定版本中未发现许可证文件，许可标为 unknown，
本仓库只提交 URL/commit，不复制或重新许可上游源码及真实用户文本。

将 `booksAll_recagent.pkl`、`combined_books_asin_mapping.csv` 放在 `datasets/raw/books/`。
核对 `configs/data/books.yaml` 的字段映射，只有来源可信时才设置 `trusted_pickle: true`
（pickle 读取会执行代码）。缺少原始 user ID 时保留 row namespace，并明确记录 unknown。
支持 books、movietv、goodreads、yelp；其他领域按相同配置格式指定文件路径。

```powershell
.\.venv\Scripts\agentreclab.exe data inspect --dataset books --raw-dir datasets/raw/books
.\.venv\Scripts\agentreclab.exe data convert --dataset books --config configs/data/books.yaml
.\.venv\Scripts\agentreclab.exe data freeze --dataset books --seed 42 --candidates 10 --limit 1000 --out datasets/manifests/books_1k.json
```

原始候选存在时绝不重新采样。转换后的 catalog/interactions/cases 分别为 Parquet，
manifest 保存原始文件、候选、case IDs 的 SHA-256。训练文本按 event IDs 引用。
原始时间未知时记录 `sequence_only`，不能宣称严格日历时间一致。
`controlled_comparison` 和 `paper_reproduction` 分开；后者在缺少原始对齐证据时明确拒绝。

## 输出与真实 provider

每个 run 包含 config/manifest、SQLite 状态与 response cache、JSONL 请求日志/预测、
原始 response 内容寻址文件、准备阶段状态快照、指标和费用报告。指标同时报告失败排除和
失败计零两种口径。成功 case 恢复时不重新请求，跨进程 run 锁阻止重复启动。

支持 OpenAI、Azure 和 OpenAI-compatible 服务。API Key 从环境变量读取；所有真实调用
需要显式 `--allow-paid`。先配置 provider 型号/snapshot/deployment/能力，再执行 dry-run。
不内置未经验证的实时价格；CNY 预算需要配置带日期的每百万 token 价格。
首次真实实验按 5→20→100 cases 推进。不得将本地 Qwen 结果标为原论文结果。

详细说明：[架构](docs/architecture.md)、[协议](docs/protocols.md)、
[复现边界](docs/reproduction.md)、[限制](docs/limitations.md)。
