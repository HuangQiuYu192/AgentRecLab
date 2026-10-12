# AgentRecLab

精简的指令感知推荐实验框架。**iAgent、i²Agent、MemRec 均在项目内原生实现**，运行和普通测试不下载、不导入官方模型代码。默认使用合成数据和 mock LLM。

## 目录

```text
src/agentreclab/
  cli.py          命令行入口
  config.py       实验和 provider 配置
  schema.py       数据结构
  data.py         数据转换、冻结、校验、读取
  models/
    base.py       JSON 调用、ID 映射和状态保存
    iagent.py     iAgent 与 i²Agent
    memrec.py     训练图、邻居规则、记忆读写和重排
    __init__.py   三个模型的简单注册表
  llm.py          mock / HTTP、JSON 校验、缓存和预算
  storage.py      SQLite、快照和请求日志
  runner.py       准备、运行、恢复和输出
  evaluation.py  排名指标与配对比较
configs/          实验、provider、数据配置；模型参数仅 MemRec 需要
tests/            单元与集成测试；reference/ 为可选原版对照
scripts/          验收、依赖锁定和可选参考源码获取
docs/             调用流程、重实现边界和验收记录
```

13 个 Python 源文件，无同步桥接、动态加载或额外在线协议层。

## 运行

Python 3.11+，Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\agentreclab.exe doctor
.\.venv\Scripts\agentreclab.exe data fixture
.\.venv\Scripts\agentreclab.exe run --config configs/runs/smoke_books.yaml --model iagent
.\.venv\Scripts\agentreclab.exe run --config configs/runs/smoke_books.yaml --model i2agent
.\.venv\Scripts\agentreclab.exe run --config configs/runs/smoke_books.yaml --model memrec
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOS 使用 `.venv/bin/python`、`.venv/bin/agentreclab`。完整 10/100 样例验收：`python scripts/acceptance.py --prefix my_validation`。重复运行使用新 run ID；恢复已有实验加 `--resume`。配置、数据或代码变化会拒绝恢复旧实验。

## iAgent 调用流程

`cli.py → config.py → runner.py → data.py → models/iagent.py → llm.py → storage.py → evaluation.py`

iAgent 每样例两次逻辑 LLM 调用：根据指令生成知识，结合历史和候选重排。
i²Agent 为五次：模拟训练推荐 → 训练反馈画像 → 知识 → 动态兴趣/画像 → 重排。
MemRec 先完成训练记忆预热，再读取邻居、提取偏好、重排；测试阶段不写记忆。重复请求可命中缓存，实际请求数可能少于逻辑调用数。

## 实验边界

当前是 **native controlled comparison**。mock 指标只验证流程，不证明推荐效果。iAgent 系列有效返回路径已通过可选原版请求对照；MemRec 保留图和领域规则，采用更短的原生提示词，不宣称严格复现。`paper_reproduction` 明确拒绝运行。

真实数据通过 `configs/data/books.yaml` 转换并保留原始候选顺序；未知时间戳标为 sequence-only。真实 LLM 调用须显式 `--allow-paid`，先用 `--dry-run` 查看调用量。支持 OpenAI、Azure、Qwen 和兼容 HTTP 服务，密钥从环境变量读取。

输出位于 `outputs/runs/<run_id>/`，包含预测、指标、费用、请求缓存和准备状态。`compare --runs <目录1> --runs <目录2> --paired-only` 核对相同样例、候选、训练数据及 provider。

详见 [架构与调用流程](docs/architecture.md)、[重实现边界](docs/reproduction.md)、[验证记录](docs/validation.md)。官方源码只用于可选审计；普通运行无需获取。
