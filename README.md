# AgentRecLab

指令感知推荐实验项目，原生实现 iAgent、i²Agent 和 MemRec。默认使用真实 Books 100 用户子集和 mock LLM，不依赖官方模型源码。

## 目录

```text
AgentRecLab/
├── configs/
│   ├── models/          模型参数
│   ├── api/             mock、OpenAI、Azure、Qwen 的 API 配置
│   ├── datasets/        数据来源与转换配置
│   └── experiments/     实验组合配置
├── datasets/
│   ├── raw/             原始数据，真实文件不提交 Git
│   └── processed/       处理后数据，包含真实 books100 子集
├── scripts/
│   ├── run_iagent.sh
│   ├── run_i2agent.sh
│   ├── run_memrec.sh
│   └── run.sh           公共实验启动脚本
├── models/
│   ├── iagent.py        静态 iAgent
│   ├── i2agent.py       动态 i²Agent
│   ├── memrec.py        记忆协同推荐
│   └── __init__.py      简单模型注册
├── utils/
│   ├── core/            公共基础定义
│   │   ├── base.py      模型基类、JSON 调用、ID 映射
│   │   ├── schema.py    数据结构与异常
│   │   └── config.py    配置读取、项目根目录定位
│   ├── data/
│   │   └── dataset.py   数据转换、抽样、冻结、读取与校验
│   ├── experiment/     实验执行与结果管理
│   │   ├── runner.py    实验调度、恢复、源码校验
│   │   ├── llm.py       即时 LLM 请求、格式校验、缓存、预算
│   │   ├── batch.py     Batch 任务提交、查询、恢复与结果匹配
│   │   └── storage.py   SQLite、快照、日志
│   └── evaluation/
│       └── metrics.py   测试集 HR、NDCG、MRR 与模型比较
├── main.py              主入口
└── outputs/             自动生成的实验结果，不提交 Git
```

`configs/models` 当前仅 MemRec 有可调参数；iAgent 系列保留固定算法流程，模型选择在实验配置中指定。测试集的最终指标统一由 `utils/evaluation/metrics.py` 计算。

## 环境

Python 3.11+，Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe main.py doctor
```

Linux/macOS 使用 `.venv/bin/python`。`requirements.lock` 保存验证过的依赖版本。

## 启动实验

直接通过主文件运行：

```powershell
.\.venv\Scripts\python.exe main.py --config configs/experiments/books100.yaml --model iagent
.\.venv\Scripts\python.exe main.py --config configs/experiments/books100.yaml --model i2agent
.\.venv\Scripts\python.exe main.py --config configs/experiments/books100.yaml --model memrec
```

Linux、WSL 或 Git Bash 中可直接运行脚本：

```bash
bash scripts/run_iagent.sh
bash scripts/run_i2agent.sh
bash scripts/run_memrec.sh
bash scripts/run_memrec.sh --provider qwen --dry-run
```

脚本从任意工作目录启动，自动定位项目根目录和 `.venv`；可用 `PYTHON=/path/to/python` 指定解释器。Windows 请在 Git Bash 中运行 `.sh`，或使用上面的 PowerShell/Python 命令。
脚本参数原样传给主入口，包括 `--config`、`--provider`、`--run-id`、`--resume`。同一 run ID 再次运行需 `--resume`，配置/数据/代码变化会拒绝恢复。

## 调用流程

`main.py → utils/core/config.py → utils/experiment/runner.py → utils/data/dataset.py → models/ → utils/experiment/llm.py → utils/experiment/storage.py → utils/evaluation/metrics.py`。

| 模型 | 逻辑调用顺序 |
|---|---|
| iAgent | 指令知识 → 历史与候选重排 |
| i²Agent | 模拟训练推荐 → 训练反馈画像 → 知识 → 动态兴趣/画像 → 重排 |
| MemRec | 训练记忆预热；测试时读取邻居/提取偏好 → 重排 |

i²Agent 保留最近 15 条历史与 `[-2]` 训练反馈索引。MemRec 基础图排除每用户最后一条预热反馈，预热按时间截断全部训练事件，测试阶段只读。领域规则中的 metadata overlap=0.5、memory similarity=0 是参考路径占位值；近因天数由位置换算。
测试标签只用于指标，不进入提示词、修复或记忆写入。固定候选顺序，非法排列默认失败；缓存和 SQLite 支持请求重放与中断恢复。

## 数据、API 与测试

原始数据已放在 `datasets/raw/{books,movietv,goodreads,yelp}/`，不提交 Git。Books 通过 `reviewerID` 识别真实用户；`configs/datasets/books100.yaml` 按 seed=42 抽取 100 位用户，保留原始行、历史和候选顺序，只保留相关物品及 100 个训练备用物品。未知时间戳标为 sequence-only。`trusted_pickle: true` 仅用于可信文件，pickle 读取会执行代码。

```bash
python main.py data convert --dataset books --config configs/datasets/books100.yaml
python main.py --config configs/experiments/books100.yaml --model iagent
bash scripts/run_memrec.sh --config configs/experiments/books100.yaml
python main.py validate --manifest datasets/processed/books100/dataset_manifest.json
python main.py compare --runs outputs/runs/run_a --runs outputs/runs/run_b --paired-only
python -m ruff check main.py models utils
```

API Key 从环境变量读取，provider 参数在 `configs/api/` 配置。真实调用先 `--dry-run`，再显式 `--allow-paid`。CNY 预算需带日期的 token 价格；请求发出后、响应保存前崩溃会留下 inflight 标记，需核对服务端记录。

当前为 native controlled comparison。MemRec 提示词已精简，未宣称严格论文复现；`paper_reproduction` 拒绝运行。已在真实 Books 100 用户子集验证 mock 流程，尚无付费/真实 LLM 的论文指标验证。实验结果保存在 `outputs/runs/<run_id>/metrics.json`，预测明细保存在 `predictions.jsonl`。

## Books 原始文件格式

`booksAll_recagent.pkl` 是普通版本（7,377 用户、9 列、每用户 10 个候选）；`booksAll_recagent_1ads.pkl` 是扩展版本（同样的用户、指令与历史，新增 `ads_title`、`ads_description`，每行在原候选末尾追加一个物品，共 11 个候选）。当前转换配置使用普通版本。扩展版本需另设 11 候选的实验协议，不能混入当前 10 候选比较。

pkl 是上游发布的 Python pickle 序列化文件，`pandas.read_pickle` 读取为 DataFrame；每行对应一个用户测试样例。`instruction`、`persona` 是字符串；`title`、`description`、`reviewText`、`asin_old`、`asin` 是列表；`ranked_lists` 是 NumPy 数组。`asin_old` 保存原始商品标识，`asin` 保存映射后的整数 ID。历史列表最后一项为待预测目标，运行时移除最后一项及其评论。部分 description 字符串仍含列表字面文本和 HTML，并非全部是干净的自然语言。`combined_books_asin_mapping.csv` 是整数 ID、原始 ASIN、标题与描述的物品映射表。

该数据已由上游处理成 InstructRec，属于本项目的输入原始数据，并非 Amazon 未加工的原始日志。pickle 能保留列表和数组等嵌套对象；作者选择此格式的具体动机未在 README 中说明。项目内部转换为 Parquet，实验配置指向 `datasets/processed/books100/dataset_manifest.json`。

## 测试集评估

实验运行完成后会自动生成 `metrics.json`。也可读取已有预测记录重新汇总指标：

```powershell
.\.venv\Scripts\python.exe main.py evaluate --run outputs/runs/books100_real_mock_iagent
.\.venv\Scripts\python.exe main.py compare --runs outputs/runs/books100_real_mock_iagent --runs outputs/runs/books100_real_mock_i2agent --paired-only
```

`exclude` 仅统计成功样例，`count_as_failure` 将失败样例记为零；同时输出成功数与失败数。当前输出 HR@1/3/5/10、NDCG@3/10 和 MRR。mock 的这些数值只能用于验证运行流程，真实推荐质量需使用真实 LLM 实验结果。

## Batch 调用

默认 `use_batch: false`。iAgent 和 i²Agent 的模型类声明支持 Batch；MemRec 默认使用即时调用，显式开启 Batch 会报错。即时与 Batch 共用模型 `workflow()` 的提示词和解析逻辑：即时模式逐步执行，Batch 模式由 `predict_batch()` 汇集各案例的同阶段请求。iAgent 为两阶段，i²Agent 为五阶段；当前保留原始执行顺序，没有合并独立步骤。

先使用真实 Books100 数据和本地 mock 验证批处理路径：

```powershell
.\.venv\Scripts\python.exe main.py --config configs/experiments/books100.yaml --model iagent --use-batch --run-id books100_batch_mock
.\.venv\Scripts\python.exe main.py --config configs/experiments/books100.yaml --model i2agent --use-batch --run-id books100_batch_mock_i2
```

真实 OpenAI Batch 使用原有环境变量 API Key。先预检查，再显式允许真实提交：

```powershell
.\.venv\Scripts\python.exe main.py --config configs/experiments/books100.yaml --model iagent --provider openai --use-batch --dry-run
.\.venv\Scripts\python.exe main.py --config configs/experiments/books100.yaml --model iagent --provider openai --use-batch --allow-paid --run-id books100_openai_batch
```

第一版仅支持 OpenAI Batch 和本地 mock，不把 Azure/Qwen 的 Chat Completions 兼容接口自动当成 Batch 接口。支持的模型和额度取决于服务账号。[官方 Batch 文档](https://developers.openai.com/api/docs/guides/batch)说明批处理提供 50% 费用折扣，使用 24 小时处理窗口。

`batch` 配置控制每文件请求数、文件字节数、查询间隔与本次查询等待时间。`concurrency` 不控制 Batch 内部并行度；`max_calls` 按文件内的模型请求条数计数，包括失败或仍预留的请求，不按批任务数量计数。超出整个文件的请求/费用额度时，不提交该文件。大阶段会按数量及字节数拆成多个文件。

默认每次最多查询等待 120 秒。任务仍在处理时主入口返回 `batch_pending`，远端任务继续运行；用原命令加 `--resume` 继续查询并进入下一阶段。相同实验必须保留原配置、源码、数据和 run ID。Batch 不支持即时模式的 `one_retry` 排序修复；可用 `fail` 或 `append_missing`。

本地 `outputs/runs/<run_id>/batches/` 保存每个任务的请求、任务 ID、状态和下载结果；结果按 `custom_id` 匹配。请求级结果保存在 SQLite，恢复时复用已完成步骤和原任务，不重复提交。终态错误结果同样保留，不会在 `--resume` 时自动重投。若提交已发出但响应丢失，会拒绝自动重投：需要核对服务端任务，并在对应 `job.json` 中补回 `batch_id` 后恢复。

Batch 费用只读取 `batch.pricing` 的显式价格表，字段与 API 配置中的 `pricing` 相同（CNY、effective_date、input_per_million、output_per_million、可选 cached_per_million）。未配置则标记费用未知，不套用标准请求单价；使用 `budget_cny` 时必须配置经核验的 Batch 价格。
