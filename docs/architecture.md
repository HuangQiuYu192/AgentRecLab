# 原生框架与调用流程

v0.2 将原来的 37 个 Python 文件合并为 13 个，算法直接在 `models/` 中实现。没有官方类导入、SDK 注入、同步/异步桥接，也没有未使用的在线生命周期接口。

## 公共流程

1. `cli.py:run` 调用 `config.py:load_run`，解析实验、provider 和可选 MemRec 参数。
2. `runner.py:run_experiment` 调用 `data.py:load_dataset`，核对校验和、训练历史及固定候选。
3. `models/__init__.py:create_model` 创建模型，`base.py:prepare` 建立 ID 映射。
4. MemRec 在 `memrec.py:warmup` 中进行一次训练反馈预热；runner 保存准备状态。
5. runner 调用 `predict`；`base.py:ask` 直接 await 统一 LLM 接口。
6. `llm.py:TracedProvider` 查询缓存、预留预算、请求 mock/HTTP、保存响应；`storage.py` 提供 SQLite、内容哈希日志与原子快照。
7. runner 检查完整候选排列，`evaluation.py` 使用测试标签计算指标并汇总输出。

## 三个模型

| 模型文件 | predict 内的顺序 | 测试阶段写入 |
|---|---|---|
| `models/iagent.py:IAgent` | `inputs` → `knowledge` → `ask(rerank)` → `prediction` | 无 |
| `models/iagent.py:I2Agent` | 训练推荐 → 反馈画像 → `knowledge` → 动态兴趣/画像 → rerank | 无 |
| `models/memrec.py:MemRec` | `graph` → `neighbors` → `pack` → facets → scores 排序 | 无 |

iAgent 系列画像按样例构建。i²Agent 保留最近 15 条历史和 `[-2]` 训练反馈索引。MemRec 把图、领域评分、混合邻居选择、文本打包、Stage-R/W 放在同一文件，便于修改算法；公共框架不统一内部记忆表示。

## 状态与扩展

缓存和状态支持中断恢复、费用记录与同样例比较。跨进程 SQLite 锁防止重复启动；恢复要求配置/数据/代码指纹相同。已完成样例不再执行，未提交样例重放已缓存响应；预热中断从确定性的训练序列及缓存重建。

测试预测可并发调度，统一 provider 的缓存/预算操作串行；whole-rank one_retry 也串行。
新增模型只需实现 `prepare`/`predict`、可选 `warmup`，加入 `create_model` 字典，无需增加独立协议或数据管线。
