# AgentRecLab

精简的指令感知推荐实验框架，原生实现 iAgent、i²Agent 和 MemRec。运行不依赖官方代码；默认合成数据 + mock LLM，无需 API Key。

## 目录

```text
src/agentreclab/
  cli.py          命令行
  config.py       配置
  schema.py       数据结构
  data.py         数据转换、冻结、读取和校验
  models/
    base.py       JSON 调用、ID 映射、状态保存
    iagent.py     iAgent / i²Agent
    memrec.py     图、邻居选择、记忆读写、重排
    __init__.py   模型注册
  llm.py          mock / HTTP、格式校验、缓存、预算
  storage.py      SQLite、快照、日志
  runner.py       运行与恢复
  evaluation.py  指标与配对比较
configs/          一个运行示例、数据和 provider 配置、MemRec 参数
scripts/          验收与依赖锁定
tests/            原生模型、数据、provider 和恢复测试
datasets/fixtures/books/  最小合成样例
docs/validation/acceptance.json  最新验收摘要
```

13 个 Python 源文件。没有官方源码副本、参考下载器、参考审计测试或占位模型。

## 运行

Python 3.11+，Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\agentreclab.exe doctor
.\.venv\Scripts\agentreclab.exe run --config configs/runs/smoke_books.yaml --model iagent
.\.venv\Scripts\agentreclab.exe run --config configs/runs/smoke_books.yaml --model i2agent
.\.venv\Scripts\agentreclab.exe run --config configs/runs/smoke_books.yaml --model memrec
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/acceptance.py --prefix my_validation
```

Linux/macOS 使用 `.venv/bin/python` 和 `.venv/bin/agentreclab`。`requirements.lock` 保存已验证依赖版本。
验收脚本在 `outputs/fixtures/` 自动生成 10/100 样例，不再提交重复样例数据。重复实验使用新 run ID；恢复已有实验加 `--resume`，配置/数据/代码变化会拒绝恢复。

## 模型流程

公共入口：`cli.py → config.py → runner.py → data.py → models/ → llm.py → storage.py → evaluation.py`。

| 模型 | 每样例逻辑调用顺序 |
|---|---|
| iAgent | 指令知识 → 结合历史和候选重排 |
| i²Agent | 模拟训练推荐 → 训练反馈画像 → 知识 → 动态兴趣/画像 → 重排 |
| MemRec | 训练记忆预热；测试时邻居读取/偏好提取 → 重排 |

i²Agent 保留最近 15 条历史与训练反馈 `[-2]` 索引，画像按样例重建。MemRec 基础图排除各用户最后一条训练预热反馈，预热按时间截断全部训练事件并顺序写记忆；测试阶段只读。领域规则中的 metadata overlap=0.5、memory similarity=0 为所审计路径的占位值，近因天数由序列位置换算。

测试标签只用于指标计算，不进入提示词、修复条件或记忆写入。候选顺序固定，非法排列默认失败，指标同时报告失败排除与失败计零。缓存减少重复调用，SQLite 状态支持恢复；已完成样例不重新请求。

## 真实数据与 LLM

通过 `configs/data/books.yaml` 转换可信 InstructRec pickle，原始候选不重采样。`trusted_pickle: true` 仅用于可信文件，读取会执行代码。未知用户 ID 使用行 namespace，未知时间戳标为 sequence-only；真实数据与输出不提交 Git。

支持 mock、OpenAI、Azure 和 Qwen/兼容 HTTP 服务。修改 provider 配置，密钥通过环境变量设置；先 `run --dry-run` 查看调用估计，再显式 `--allow-paid`。CNY 预算需配置带日期的 token 价格。JSON-schema 不可用时添加 JSON 格式说明。请求发出后、响应保存前崩溃会保留 inflight 标记并阻止自动重发，需核对服务端记录。

输出：`outputs/runs/<run_id>/`。配对比较：`compare --runs <目录1> --runs <目录2> --paired-only`，核对相同样例、候选、训练历史与 provider。

## 验证边界

当前采用 native controlled comparison。iAgent 系列有效 mock 路径及 MemRec 四领域规则曾通过原版对照，历史证据保存在 Git 提交 `ad31d4c`；参考工具已删除。
MemRec 提示词已精简，未宣称严格复现；`paper_reproduction` 拒绝运行。没有真实数据或付费 LLM 的论文指标验证，mock 排序只验证流程。

最新验收结果见 [acceptance.json](docs/validation/acceptance.json)。
