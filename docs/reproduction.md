# 重实现边界

用户要求在框架内重写模型、尽可能精简。v0.2 替换 v0.1 官方 wrapper，采用本地原生实现；历史源码审计保存在 `docs/reference/`，其中 wrapper 计划已失效。普通运行、doctor、CI、单元与集成测试均不依赖 `third_party`。

## 保留与变化

| 模型 | 保留 | 明确变化 |
|---|---|---|
| iAgent | 知识→重排、提示词/JSON 字段、候选顺序、文本裁剪 | 直接异步调用；整排列校验；不执行依赖测试标签的修复 |
| i²Agent | 五步流程、最近 15 条历史、训练反馈 `[-2]`、每样例画像 | 冻结训练负例；统一缓存、预算和错误处理 |
| MemRec | 训练图、重叠/位置近因、四领域规则、混合邻居、预算打包、Stage-R/W | 提示词重写精简；仅训练反馈写记忆；时间截断；冻结候选；严格 fanout 校验 |

MemRec metadata overlap=0.5、memory similarity=0 保留所审计路径的占位值，不是已训练的相似度模型；近因天数为位置换算。基础图训练历史不含每用户最后一条预热反馈；预热时按时间戳截断全部用户事件。按稳定时间顺序完成一次预热，训练负例排除全部已知训练历史。

测试阶段只有指标层使用 positive_item_id，提示词、修复条件和记忆写入不读取它。非法 JSON 或排列报告失败，无默认排序兜底。显式 `append_missing` 仅补漏，`one_retry` 为一次完整重排重试。指标同时输出失败排除和失败计零口径。

## 数据与真实服务

原始 InstructRec 数据尚未提供，没有真实数据或付费 LLM 的论文指标验证。转换器支持 books/movietv/goodreads/yelp，保留预生成候选；未知用户 ID 采用行 namespace，未知时间戳标为 sequence-only。

只有可信来源的 pickle 才能设置 `trusted_pickle: true`，读取会执行其中的代码。参考源码许可证未找到，数据再分发权未知；仓库只记录来源/commit，不提交参考源码或真实文本。输出和原始数据被 Git 忽略，当前支持源码 checkout 内运行。

真实服务需设置密钥和 provider 参数，先 dry-run，再显式 `--allow-paid`。价格不猜测，CNY 预算需要带日期的 token 价格。超时重试可能重复计费；在请求发出后、响应持久化前崩溃，会留下 inflight 标记并阻止自动重发，需核对服务端记录。

## 可选原版对照

普通 pytest 跳过参考审计，需要核对时才执行：

```powershell
python scripts/fetch_reference.py
python -m pip install openai
$env:AGENTRECLAB_REFERENCE_TESTS='1'
python -m pytest tests/reference -q
Remove-Item Env:AGENTRECLAB_REFERENCE_TESTS
```

参考代码只在这些测试中加载，所有请求仍使用 mock。JSON-schema 不可用的 HTTP 服务会得到格式说明 system 消息，实际 payload 可能不同；参数、能力和模型请求在本地记录。严格论文复现和 MARS 未就绪。
