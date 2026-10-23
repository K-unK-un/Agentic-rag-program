# 司典智答 —— 企业员工制度智能问答系统 技术文档

## 一、项目概述

**司典智答**是一套面向企业内部员工的制度智能问答系统，用户以自然语言提问，系统自动检索 6 份共 1.6 万字的企业制度文档，返回带原文引用的准确答案。

### 业务背景

企业制度文档普遍存在以下痛点：

| 痛点 | 具体表现 |
|---|---|
| **检索困难** | 制度文档分散在假期/薪酬/考勤/报销等多个文件中，员工查一条规定需要翻遍全文 |
| **结构化关系缺失** | 制度文本包含大量阈值条件（工龄满 10 年年休假 10 天）、触发链（连续旷工 3 天→解除合同）、审批层级（部门经理→HR→总经理），纯关键词或纯向量检索无法精准捕获这些跨条款关联 |
| **人工答复成本高** | 高频问题（年假几天、病假工资怎么算）重复出现在 HR 咨询中，占用人力 |

### 目标与约束

- **准确性优先**：对制度内问题，答案必须能追溯到原文条款，禁止幻觉
- **覆盖阈值关系**：能回答"工龄 6 年病假 10 天工资怎么发"这类跨字段联动问题
- **低成本可部署**：Embedding 与 Reranker 本地部署（RTX 4060 8GB），LLM 混合云端+本地降本
- **可解释性**：答案带引用来源，Agent 工具调用链可追踪

---

## 二、系统架构

### 整体架构图

```
┌────────────────────── 阶段一：离线索引构建（一次性） ──────────────────────┐
│                                                                            │
│  resources/*.md（6 份制度文档，1.6 万字）                                    │
│        │                                                                   │
│        ▼                                                                   │
│  DocumentLoader → DocumentSplitter（标题切 + 512字二次切）                   │
│        │                                                                   │
│        ▼                                                                   │
│     chunks ─────────────┬─────────────────────┐                            │
│                          │                     │                            │
│              ┌───────────▼──────────┐  ┌───────▼──────────┐               │
│              │ VectorIndex          │  │ Neo4jIndex        │               │
│              │ nomic-embed-text     │  │ qwen3.8-max       │               │
│              │ → Chroma 持久化       │  │ →清洗→Neo4j图     │               │
│              └──────────────────────┘  └───────────────────┘               │
│                                                                    474实体/ │
│                                                                    391关系/ │
│                                                                    15种类型 │
│  Bm25Index ← Chroma 读回 chunks → 内存 BM25 倒排索引                        │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────── 阶段二：在线问答（每请求） ───────────────────────┐
│                                                                        │
│  前端 SSE ──▶ FastAPI /chat/ask                                        │
│                    │                                                   │
│                    ▼                                                   │
│        会话级 DocStore（内存 dict，请求结束即丢）                         │
│                    │                                                   │
│        LangGraph ReAct Agent（6 工具 + TodoList 中间件）                │
│                    │                                                   │
│   strategy_tool：判断检索方式 + 拆子问题（Redis 缓存 TTL 1 天）          │
│                    │                                                   │
│   Agent 自主路由（可多轮交叉印证）：                                      │
│     BM25（关键词精确） / Vector（语义模糊） / Neo4j 图谱（阈值关系）      │
│     / Hybrid（RRF 融合 c=60）                                            │
│                    │                                                   │
│   每篇文档 → md5[:16] → DocStore，Agent 上下文只传 {id,snippet,score}    │
│                    │                                                   │
│   rerank_doc：bge-reranker-large CrossEncoder 精排 top5                │
│                    │                                                   │
│   qwen3.8-max 生成答案 ──SSE token 流──▶ 前端 marked.js 渲染             │
└────────────────────────────────────────────────────────────────────────┘
```

### 模块分层

```
rag_program/
├── indexer/          # 索引构建（离线）
│   ├── document_loader.py    # 文档加载
│   ├── document_splitter.py  # 分块
│   ├── vector_index.py       # 向量入库
│   ├── neo4j_index.py        # 图谱构建
│   ├── bm25_index.py         # BM25 索引
│   └── rag_index.py          # 索引入口
├── retriever/        # 检索器
│   ├── bm25_retriever.py
│   ├── vector_retriever.py
│   ├── neo4j_retriever.py
│   └── hybrid_retriever.py
├── tools/            # LangChain @tool 封装
│   ├── retriever_tools.py    # 4 个检索工具
│   ├── reranker_tool.py      # rerank_doc
│   ├── strategy_tool.py      # 策略分析
│   └── doc_store.py          # doc_id 映射
├── agent/            # Agent 编排
│   ├── agent.py              # create_retriever_agent
│   └── prompts.py            # System Prompt + ANALYZE_PROMPT
├── reranker/         # 重排
│   └── reranker.py           # CrossEncoder 封装（模块级单例）
├── models/           # LLM / Embedding
│   └── llm.py                # qwen_llm + embeddings（模块级单例）
├── service/          # 业务服务
│   ├── chat_service.py       # SSE 问答主流程
│   └── redis_service.py      # Redis 封装
├── api/              # FastAPI 路由
│   ├── chat_api.py
│   └── ingest_api.py
├── loader/           # 文档加载与分块
├── utils/            # 配置、日志
├── resources/        # 制度文档（6 份 md）
└── front/            # 前端（原生 HTML + marked.js）
```

---

## 三、技术选型与决策理由

### 3.1 为什么选择 LangGraph 而非固定 LCEL 链

**问题**：固定管线（检索→rerank→生成）只能按预设顺序执行，无法应对"向量召回不全→补调 BM25"这类动态场景。

**决策**：采用 LangGraph 的 `create_agent` 构建 ReAct Agent，Agent 自主判断调哪些工具、调几次。

**理由**：

| 方案 | 灵活性 | 可控性 | 适合场景 |
|---|---|---|---|
| 固定 LCEL 链 | 低 | 高 | 检索策略固定的简单 RAG |
| LangGraph Agent | 高 | 中 | 多路检索 + 动态路由 |

**代价**：Agent 存在工具选择的非确定性，需要精心设计 system prompt 约束调用顺序。

### 3.2 为什么选择 Neo4j 图谱增强 RAG

**问题**：纯向量检索对阈值条件（工龄满 10 年）、触发链（旷工→解除合同）、审批层级（部门→HR→总经理）这类结构化关系召回不全——向量空间中"工龄 6 年病假 10 天"和"病假工资 80%"语义距离较远，chunk 粒度切分后可能分到两段。

**决策**：用 LLM 从制度文本抽取三元组构建 Neo4j 知识图谱。

**理由**：

- **精确匹配关系**：Cypher `MATCH (s {name: "工龄满6年"}) -[r]-> (o)` 能直接拿到该实体参与的所有关系链
- **阈值条件可存储**：实体名直接包含数值（"工龄满 6 年"），关系属性存约束（`constraint: 80%`）
- **可解释性**：返回三元组字符串（"旷工 3 天 - 触发 -> 解除劳动合同"），前端可直接展示

**代价**：三元组抽取耗时长（本地 qwen3:8b 串行约 40 分钟），需额外维护一套 schema 对齐逻辑（抽取端和检索端实体名字面一致才能 MATCH）。

### 3.3 为什么用 RRF 融合而非其他排序方法

**问题**：多路检索结果合并排序，BM25 的 TF-IDF 分数和向量的余弦相似度不在同一标尺，无法直接加权求和。

**决策**：采用 Reciprocal Rank Fusion（RRF），公式 `score = Σ weight / (rank + c)`，c=60。

**理由**：

- RRF 不依赖各检索器的分数绝对值，只依赖排序位置，天然解决分数尺度不统一问题
- c=60 是经验常数（LangChain EnsembleRetriever 默认值），平衡 top1 主导和长尾贡献
- 子类化重写 `weighted_reciprocal_rank` 将融合分数写入 metadata.rrf_score，下游 rerank 和前端可复用

### 3.4 为什么 Embedding 和 Reranker 本地部署

**约束**：qwen3.8-max 云端 API 按 token 计费，Embedding 和 Reranker 每请求都要调，累积成本高。

**决策**：

| 组件 | 模型 | 部署方式 | 原因 |
|---|---|---|---|
| Embedding | nomic-embed-text (274MB) | Ollama 本地 | 向量化是高频操作，本地零延迟零成本 |
| Reranker | bge-reranker-large (2.1GB) | PyTorch 本地 | CrossEncoder 计算量大，本地 GPU（RTX 4060）延迟可控 |
| 主 LLM | qwen3.8-max | 阿里云 DashScope 云端 | 生成质量优先，云端模型效果更稳 |
| 策略 LLM | qwen-turbo | 阿里云 DashScope 云端 | 策略分析只需结构化输出，turbo 便宜快速 |

### 3.5 为什么 DocStore 用内存 dict 而非 Redis

**问题**：Agent 多工具调用间需传递检索结果，但完整 chunk（512 字 × 15 条 = 1.2 万 token）在 Agent 上下文里累积会拖慢推理。

**决策**：检索工具只返回 `md5(page_content.strip())[:16]` 作为 doc_id，原文存会话级 Python dict，rerank 工具凭 id 回查。

**理由**：

- DocStore 存的是单次请求中间结果，请求结束即丢，不需要跨请求持久化
- 纯内存零成本，无外部依赖，不存在 Redis 挂了回查空的风险
- 同一 chunk 被多路检索召回时 md5 相同，天然去重

### 3.6 为什么策略分析用 Redis 缓存

**问题**：相同问题（如"年假几天"）反复出现，每次都调 LLM 分析检索策略浪费 token 和延迟。

**决策**：strategy_tool 输出以 `md5(query.strip().lower())` 为 key 存 Redis（TTL 1 天），命中直接返回。

**理由**：

- 策略结果跨请求复用，需要持久化（Redis 跨进程）
- TTL 1 天平衡复用率和制度更新后的时效性
- Redis 异常自动降级走 LLM，不影响主流程

---

## 四、遇到的难点与解决方案

### 难点 1：Neo4j Driver 与 PyTorch 反复加载导致进程崩溃

**现象**：连续 3-5 次 `/chat/ask` 请求后，uvicorn 进程以 `0xC0000005`（STATUS_ACCESS_VIOLATION）崩溃，PowerShell 报"请求异常: network error"。

**排查过程**：

1. 最初怀疑是 Ollama 或 Chroma 网络问题，但重启服务后同样复现
2. 加日志定位到：`chat_service.py` 每请求 `create_retriever_agent()` → `create_rerank_tool()` → `Reranker()` → `get_reranker()` → `HuggingFaceCrossEncoder()`，每请求重新加载 2.1GB 模型
3. 同理 `Neo4jRetriever()` 每请求新建 Neo4jGraph driver
4. PyTorch 和 neo4j-python-driver 底层均为 C 扩展，反复加载累积内存碎片化触发崩溃

**解决方案**：将 `CrossEncoderReranker` 和 `Neo4jGraph` 改为模块级单例懒加载：

```python
# neo4j_retriever.py
_graph_instance = None
def _get_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = Neo4jGraph(url=..., username=..., password=...)
    return _graph_instance

# reranker.py
_reranker_instance = None
def _get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        cross_encoder = HuggingFaceCrossEncoder(model_name=...)
        _reranker_instance = CrossEncoderReranker(model=cross_encoder, top_n=5)
    return _reranker_instance
```

**效果**：修复前 3-5 次请求必崩，修复后连续 20+ 次请求稳定运行。

### 难点 2：EnsembleRetriever 融合分数无法下游复用

**现象**：Hybrid 检索返回的文档没有融合得分，rerank 工具和前端展示拿不到 RRF 分数。

**排查**：LangChain `EnsembleRetriever.weighted_reciprocal_rank` 只返回排序后的 Document 列表，不将融合分数写回 metadata，父类设计缺口。

**解决方案**：子类化 `RrfEnsembleRetriever`，重写 `weighted_reciprocal_rank`：

```python
class RrfEnsembleRetriever(EnsembleRetriever):
    def weighted_reciprocal_rank(self, ...):
        # 1. 按父类公式算各文档 RRF 分数
        # 2. 调 super() 复用父类去重排序逻辑
        # 3. 将分数写回 doc.metadata["rrf_score"]
```

**关键细节**：`_make_key` 方法必须与父类保持一致（按 page_content + metadata 去重），否则同一条 chunk 被多路召回时会被当成不同文档重复精排。

### 难点 3：Neo4j 图谱 schema 对齐问题

**现象**：部分问题图谱检索返回空，例如"工龄 6 年病假 10 天"查不到，但 Neo4j Browser 里手动 MATCH 能查到。

**根因**：图谱写入端（LLM 抽取）和检索端（LLM 实体抽取）对同一概念的命名不一致：

| 写入端实体名 | 检索端实体名 | 结果 |
|---|---|---|
| 工龄满 6 年 | 工龄 6 年 | ❌ MATCH 不上 |
| 带薪年休假 | 年休假 | ❌ MATCH 不上 |
| 旷工 | 旷工行为 | ❌ MATCH 不上 |

**解决方案**：

1. 抽取 prompt 内嵌同义词归一规则（带薪年休假→年休假、年度累计→年度合计）
2. 抽取 prompt 要求阈值条件必须带数值（工龄满 X 年、旷工 X 天）
3. 写入后批量校验：对每条关系的 subject/object 跑同义词替换，修正不一致
4. 检索端实体抽取 prompt 复用同一套 8 类实体清单

### 难点 4：Agent 工具选择的非确定性

**现象**：相同问题 5 次查询，1 次召回条数不同，偶尔 Agent 只调 vector 不调 neo4j，导致阈值条件召回不全。

**根因**：LangGraph ReAct Agent 的工具选择由 LLM 在 runtime 自主决策，system prompt 是建议不是强约束。

**解决方案**：

1. 设计 strategy_tool 前置策略分析，LLM 结构化输出 retrieval_types + need_decomposition + sub_questions
2. AGENT_SYSTEM_PROMPT 明确调用顺序：第一步 strategy_tool → 第二步按 retrieval_types 调工具 → 第三步 rerank → 第四步生成
3. 允许 Agent "一次结果不足可继续调用其他工具补充"（兜底机制）
4. 降低 LLM temperature 至 0.3（从默认 0.7），减少非确定性

---

## 五、量化结果

### 5.1 语料与索引规模

| 维度 | 数值 |
|---|---|
| 制度文档数 | 6 份 |
| 总字数 | 16,355 字 |
| 分块数 | ~65 个（标题切 + 512 字二次切） |
| 知识图谱实体节点 | 474 个 |
| 知识图谱关系数 | 391 条 |
| 关系类型数 | 15 种 |
| 三元组全量抽取耗时 | ~40 分钟（本地 qwen3:8b 串行） |

### 5.2 性能指标

| 指标 | 数值 | 备注 |
|---|---|---|
| 单轮问答平均响应 | ~70 秒 | 含多路检索 + rerank + LLM 生成 |
| Redis 缓存命中后响应 | ~45 秒 | 跳过 LLM 策略分析，节省约 36% |
| 首次 rerank 加载 | ~8 秒 | CrossEncoder 懒加载，之后复用 |
| 相同问题 5 次查询结果一致率 | 80% | 1 次条数波动（LLM 实体抽取非确定性） |

### 5.3 四路检索召回对比

| 问题类型 | BM25 | Vector | Neo4j | Hybrid(RRF) |
|---|---|---|---|---|
| 阈值条件（工龄 6 年病假） | 2 | 1 | 4 | 3 |
| 模糊语义（病假工资怎么算） | 0 | 4 | 0 | 3 |
| 触发链（旷工几天解除） | 1 | 2 | 5 | 4 |
| 精确术语（年假规定） | 3 | 2 | 0 | 3 |
| 复合问题（产假+报销） | 1 | 2 | 2 | 3 |

**Hybrid 优势**：在阈值条件和触发链场景下，Hybrid 兼顾 BM25 的关键词精确和 Vector 的语义召回，平均召回条数比纯 Vector 多 40%。Neo4j 对阈值条件类问题召回最全，但对模糊语义问题返回空（图谱里没有"病假工资怎么算"这个实体，只有具体的"病假工资比例 80%"）。

### 5.4 DocStore Token 节省

| 方案 | 单次请求 Agent 上下文 token | rerank 调用 token | 总计 |
|---|---|---|---|
| 直接传 Document 全文 | ~12,000 | ~12,000 | ~24,000 |
| DocStore 传 id + snippet | ~4,500 | ~12,000（rerank 回查后） | ~16,500 |

**节省约 31% token/请求**。如果 Agent 多轮调用检索工具（实际平均 2-3 轮），节省比例更高。

### 5.5 崩溃修复前后对比

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 连续请求稳定性 | 3-5 次必崩 | 20+ 次稳定 |
| 进程崩溃类型 | 0xC0000005 | 无 |
| 首次 rerank 延迟 | ~20 秒（每次重载） | ~8 秒（懒加载一次） |

---

## 六、不足与改进方向

| 项目 | 当前状态 | 改进方向 |
|---|---|---|
| RAGAS 离线评估 | 无量化数据，靠主观判断 | 100 题题库跑 Faithfulness / Answer Relevancy / Context Precision |
| 三元组抽取 | 串行 for 循环，40 分钟 | asyncio.gather 并发 5-8 个 chunk，预计降至 5-8 分钟 |
| Agent 路由稳定性 | 80% 一致率 | 降低 temperature + 固化高频问题策略到 Redis |
| BM25 元数据丢失 | 从 Chroma 读回时只取 page_content | include=["documents","metadatas"] 带回来源信息 |
| 前端历史记录 | 无 | localStorage 存对话历史 |
| Docker Compose | 无统一编排 | 写 docker-compose.yml 一键启动 Neo4j/Chroma/Redis |

---

## 七、软件环境

| 类别 | 组件 | 版本 |
|---|---|---|
| 操作系统 | Windows 11 | 23H2 |
| Python | Anaconda rag_env | 3.10.x |
| Web 框架 | FastAPI | 0.115.x |
| ASGI 服务器 | Uvicorn | 0.32.x |
| LangChain | LangChain + LangGraph | 1.x |
| 向量库 | Chroma | latest (Docker) |
| 图谱库 | Neo4j | 5.x (Docker) |
| 缓存 | Redis | 7 (Docker) |
| 本地 LLM | Ollama / qwen3:8b | 5.2 GB |
| 本地 Embedding | Ollama / nomic-embed-text | 274 MB |
| 云端 LLM | qwen3.8-max / qwen-turbo | DashScope API |
| 本地 Reranker | bge-reranker-large (2.1GB) | PyTorch CrossEncoder |
| GPU | NVIDIA RTX 4060 | 8GB 显存 |
| 前端 | marked.js | 12.0.2 (CDN) |
