# 司典智答 · 企业员工制度智能问答系统

基于 **Agentic RAG** 的企业规章制度问答系统。用户用自然语言提问，Agent 自主选择 BM25 / 向量 / 混合 / Neo4j 图谱多路检索，经 CrossEncoder 精排后由大模型生成带原文出处的答案，并通过 SSE 流式返回前端。

## 项目简介

企业制度文档分散在假期、薪酬、考勤、报销、培训、办公等多个文件中，且包含大量阈值条件（工龄满 10 年年休假 10 天）、触发链（连续旷工 3 天→解除合同）、审批层级等结构化关系，纯关键词或纯向量检索难以准确召回。

本项目把 6 份制度文档（`resources/*.md`）离线处理为：

- **向量索引**：`nomic-embed-text` → Chroma
- **BM25 倒排索引**：从 Chroma 读回 chunk 后内存构建
- **知识图谱**：大模型抽取三元组 → Neo4j（`ENTITY` 节点 + `UPPER_SNAKE` 关系 + `constraint` 属性）

在线问答时由一个 ReAct Agent 自主编排检索与精排，答案必须溯源到条款，禁止编造。

## 功能特性

- **Agent 自主路由**：基于 LangGraph `create_agent`，按问题类型自主选择检索工具，可多路交叉印证、结果不足时补调
- **四路检索**
  - `bm25_search`：关键词 / 编号 / 专有名词精确命中
  - `vector_search`：语义相似度检索（Chroma，余弦距离阈值 0.5，k=5）
  - `hybrid_search`：RRF 倒数排名融合向量 + BM25（权重 0.5/0.5，c=60），返回 top 3
  - `neo4j_search`：LLM 抽取实体 → 图谱双向关系检索 → 三元组文本
- **策略分析与缓存**：`strategy_tool` 先输出 `retrieval_types` / `need_decomposition` / `sub_questions`，结果以问题 MD5 为 key 缓存到 Redis（TTL 1 天），异常自动降级
- **复合问题拆解**：需要时拆分独立子问题分别检索后合并
- **DocStore 降 token**：检索工具只回传 `doc_id + snippet`，原文存会话级内存字典，rerank 时凭 id 回查
- **CrossEncoder 精排**：`bge-reranker-large` 本地推理，取 top 5；精排前最多保留 50 篇
- **SSE 流式输出**：答案 token 实时推送，附带工具执行状态与引用来源
- **前端**：Vue 3 + marked.js，单文件 HTML，支持 Markdown 渲染与残缺标记清理

## 系统架构

### 离线索引构建

```text
resources/*.md（6 份制度文档）
        │
        ▼
DocumentLoader（markdown / txt / pdf·doc·docx）
        │
        ▼
DocumentSplitter（按 #/##/### 标题切 + 512 字、重叠 50 二次切）
        │
        ├──────────────► VectorIndex ──► nomic-embed-text ──► Chroma（cosine）
        │
        └──────────────► Neo4jIndex ──► qwen3.8-max 抽取三元组
                                        → 清洗 / 同义词归一 / 去重
                                        → Neo4j 图数据库

BM25：运行时从 Chroma 读回全部 chunk → data/chunks.json → BM25Retriever(k=3)
```

### 在线问答

```text
前端（front/index.html）
   │  GET /api/chat/ask?query=...
   ▼
FastAPI（main.py）
   │
   ▼
AgentChatService.generator
   │  每个请求独立创建 DocStore + Agent
   ▼
strategy_tool ──► Redis 缓存命中？──是──► 直接返回策略
   │ 否
   ▼
qwen-turbo 结构化输出检索策略
   │
   ▼
Agent 按 retrieval_types 调用检索工具（可多轮 / 多路）
   │  bm25_search / vector_search / hybrid_search / neo4j_search
   ▼
DocStore（md5(page_content)[:16] → Document，Agent 上下文只传 id + 200 字摘要）
   │
   ▼
rerank_doc ──► bge-reranker-large CrossEncoder 精排 top 5
   │
   ▼
qwen3.8-max 生成答案 ──SSE token 流──► 前端渲染
```

## 目录结构

```text
rag_program/
├── main.py                     # FastAPI 入口，挂载 /api 路由与 CORS
├── agent/
│   ├── agent.py                # create_agent：6 个工具 + TodoListMiddleware
│   └── prompts.py              # Agent 系统提示、策略分析提示、RAG 作答提示
├── api/
│   └── chat_api.py             # GET /api/chat/ask（SSE）
├── service/
│   ├── chat_service.py         # Agent 执行、SSE 事件封装、引用收集
│   └── redis_service.py        # Redis 策略缓存 / 验证码工具
├── tools/
│   ├── strategy_tool.py        # 检索策略分析（结构化输出 + Redis 缓存）
│   ├── retriever_tools.py      # 4 个检索工具封装
│   ├── reranker_tool.py        # rerank_doc
│   └── doc_store.py            # doc_id ↔ Document 会话级映射
├── retriever/
│   ├── bm25_retriever.py       # BM25Retriever 封装
│   ├── vector_retriever.py     # 向量检索（阈值 / 带分数）
│   ├── hybrid_retriever.py     # RrfEnsembleRetriever（补写 rrf_score）
│   └── neo4j_retriever.py      # 实体抽取 + Cypher 双向检索 + 三元组格式化
├── indexer/
│   ├── rag_index.py            # create_index 入口（全量构建）
│   ├── vector_index.py         # Chroma 写入
│   ├── neo4j_index.py          # 三元组抽取 / 清洗 / 入库
│   └── bm25_index.py           # 从 Chroma 读回 chunk 构建 BM25
├── loader/
│   ├── document_loader.py      # md / txt / pdf·doc·docx（MinerU）
│   └── document_splitter.py    # 标题切 + RecursiveCharacterTextSplitter
├── models/
│   ├── llm.py                  # qwen_llm / strategy_llm / embeddings
│   └── chroma.py               # Chroma 客户端（localhost:8081）
├── reranker/
│   └── reranker.py             # CrossEncoder 单例（懒加载）
├── utils/
│   ├── config.py               # .env 配置读取
│   └── logger.py               # 控制台日志
├── resources/                  # 6 份企业制度 Markdown
├── data/chunks.json            # BM25 运行时缓存（自动生成）
├── front/index.html            # 单文件前端（Vue 3 + marked.js）
├── project_document.md         # 详细技术文档与设计决策
├── .env.example                # 环境变量模板
└── README.md
```

## 技术栈

| 类别 | 组件 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| Agent 编排 | LangChain 1.x `create_agent` + LangGraph `TodoListMiddleware` |
| 云端 LLM | 阿里云 DashScope：`qwen3.8-max`（回答）、`qwen-turbo`（策略分析） |
| 本地 Embedding | Ollama `nomic-embed-text` |
| 向量库 | Chroma（HTTP server，cosine，collection `default`） |
| 关键词检索 | `BM25Retriever`（k=3） |
| 图谱库 | Neo4j（`ENTITY` 节点，`UPPER_SNAKE` 关系，可选 `constraint` 属性） |
| 精排 | `BAAI/bge-reranker-large`（HuggingFace CrossEncoder，top 5） |
| 缓存 | Redis（策略分析结果 TTL 1 天） |
| 前端 | Vue 3.4 + marked.js 12（CDN，无需构建） |

## 快速开始

### 1. 环境准备

需要提前启动以下服务：

| 服务 | 默认地址 | 说明 |
|---|---|---|
| Ollama | `localhost:11434` | `ollama pull nomic-embed-text` |
| Chroma | `localhost:8081` | 例：`chroma run --host 0.0.0.0 --port 8081 --path ./chroma_data` |
| Neo4j | `bolt://localhost:7687` | 需与 `.env` 中账号密码一致 |
| Redis | `redis://localhost:6379/0` | 策略缓存 |

另外需要：

- Python 3.10+
- 可用的 DashScope API Key（[阿里云百炼控制台](https://dashscope.console.aliyun.com/)）
- 本地已下载 `BAAI/bge-reranker-large`。代码默认从 `$HF_HOME/hub/models--BAAI--bge-reranker-large/snapshots/<hash>` 加载；如路径不同，请修改 `reranker/reranker.py` 中的 `model_path`（或改为直接传 `"BAAI/bge-reranker-large"`）。

### 2. 安装依赖

项目未提供 `requirements.txt`，可直接安装：

```bash
pip install fastapi uvicorn python-dotenv pydantic redis neo4j chromadb \
    langchain langchain-core langchain-community langchain-classic \
    langchain-openai langchain-ollama langchain-chroma langchain-neo4j \
    langchain-text-splitters rank_bm25 sentence-transformers torch unstructured
```

> 仅当需要加载 PDF / Word 文档时才需要 `langchain-mineru`：`pip install langchain-mineru`。

### 3. 配置环境变量

```bash
cp .env.example .env
```

然后填写 `.env`（该文件已被 `.gitignore` 忽略，不会提交）：

| 变量 | 说明 | 示例 |
|---|---|---|
| `QWEN_MODEL_NAME` | 回答模型 | `qwen3.8-max` |
| `STRATEGY_LLM_MODEL_NAME` | 策略分析模型 | `qwen-turbo` |
| `QWEN_API_KEY` | DashScope API Key | `sk-...` |
| `QWEN_BASE_URL` | OpenAI 兼容地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDINGS_MODEL_NAME` | Ollama 向量模型 | `nomic-embed-text` |
| `NEO4J_URL` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | Neo4j 连接 | `bolt://localhost:7687` |
| `REDIS_URL` | Redis 连接 | `redis://localhost:6379/0` |
| `STRATEGY_CACHE_TTL` | 策略缓存秒数（默认 86400） | `86400` |
| `MAX_RERANK_DOCS` | 精排前最大文档数 | `50` |
| `RETRIEVAL_TOP_K` | 单路检索召回数（预留） | `20` |
| `SNIPPET_MAX_CHARS` | 返回给 Agent 的摘要长度 | `200` |
| `CONNECTION_STRING` | MySQL 连接串（预留，当前代码未使用） | `mysql+pymysql://...` |

> `reranker/reranker.py` 直接读取系统环境变量 `HF_HOME`，也可写入 `.env`。

### 4. 构建索引（首次运行）

```bash
python -c "from indexer.rag_index import create_index; create_index()"
```

- 默认加载 `resources/` 下全部文件
- 默认清空并重建 Chroma 向量（`vector_delete_flag=True`）
- 默认**追加**写入 Neo4j；若需先清空图谱，调用 `create_index(neo4j_delete_flag=True)`
- 三元组抽取逐 chunk 串行调用 LLM，耗时较长，请耐心等待

BM25 无需单独构建：`Bm25Index.load_bm25()` 会在每次构建 Agent 工具时从 Chroma 读回全部 chunk 并重建 `data/chunks.json`。

### 5. 启动服务

```bash
python main.py
```

服务监听 `0.0.0.0:8000`（注意：`main.py` 中端口是硬编码的，`.env` 的 `API_HOST` / `API_PORT` 当前未生效）。

### 6. 打开前端

```bash
cd front
python -m http.server 5500
```

浏览器访问 <http://localhost:5500>。前端请求地址硬编码为 `http://127.0.0.1:8000`，如需修改请改 `front/index.html` 中的 `API_BASE`。

## API 接口

### `GET /api/chat/ask`

| 参数 | 位置 | 类型 | 说明 |
|---|---|---|---|
| `query` | query string | string | 用户问题 |

返回 `text/event-stream`，事件帧格式：

```text
event: <event>
data: <payload>

```

| 事件 | data | 说明 |
|---|---|---|
| `tool_end` | `{"name": "工具名"}` | 某工具执行完成 |
| `answer` | 字符串 | 答案 token（工具执行后产生的 AI 文本，逐段下发） |
| `done` | `{"todos": [...], "references": [{"doc_id", "snippet", "source"}]}` | 流结束，附带引用文档 |
| `error` | `{"message": "..."}` | 执行异常 |

示例：

```bash
curl -N "http://127.0.0.1:8000/api/chat/ask?query=工龄6年病假10天工资怎么发"
```

非流式调用 `AgentChatService.invoke()` 已实现（返回 `answer` / `todos` / `references`），但尚未挂载路由。

## Agent 检索策略

`strategy_tool` 根据问题特征输出 1–3 种检索方式：

| 检索方式 | 适用场景 | 示例 |
|---|---|---|
| `vector` | 语义模糊、概念性泛问 | “病假工资怎么算” |
| `bm25` | 精确术语、编号、条款名 | “21.75 是什么意思” |
| `hybrid` | 既有具体名词又有泛问，默认兜底 | “年休假 10 天怎么算” |
| `graph` | 阈值核算、触发链、审批层级、多跳关系 | “工龄 6 年病假 10 天工资怎么发” |

判断准则：含工龄 / 天数 / 金额 / 比例等阈值条件必加 `graph`；既有精确词又有模糊语义用 `hybrid`；复合问题先拆子问题再分别检索。

## 关键设计

1. **RRF 融合分数回写**：`RrfEnsembleRetriever` 继承 LangChain `EnsembleRetriever`，重写 `weighted_reciprocal_rank`，将 `weight / (rank + 60)` 的融合分数写入 `metadata.rrf_score`，供 rerank 与前端复用；去重键（`_make_key`）与父类保持一致，避免同一 chunk 重复精排。
2. **会话级 DocStore**：文档以 `md5(page_content.strip())[:16]` 为 id 存入内存字典，多路召回同一 chunk 时天然去重；Agent 上下文只看到 id 与摘要，显著降低 token。
3. **模块级单例**：`Neo4jGraph` 与 `CrossEncoderReranker` 均为懒加载单例，避免每请求重复创建 driver / 加载 2.1GB 模型导致 Windows 下 `0xC0000005` 进程崩溃。
4. **策略缓存降级**：Redis 读写异常只记录 warning，自动回退到 LLM 分析，不影响问答主流程。
5. **图谱 schema 对齐**：抽取端与检索端使用同一套实体类别和 UPPER_SNAKE 关系类型，清洗阶段做同义词归一（带薪年休假→年休假、年假→年休假等），保证 Cypher `name` 精确匹配。

## 已知限制与改进方向

- Agent 工具选择由 LLM 运行时决定，存在非确定性；可通过降低温度、固化高频问题策略改善
- 三元组抽取当前为串行 for 循环，可改为并发以缩短建图时间
- 尚未接入 RAGAS 等离线评估，缺少量化指标
- `BM25` 从 Chroma 读回时只保留 `page_content`，来源元数据会丢失
- `STRATEGY_CACHE_TTL`、`RETRIEVAL_TOP_K`、`API_HOST`、`API_PORT`、`CONNECTION_STRING` 为预留配置，当前代码未真正使用
- 前端无对话历史持久化，刷新即丢失

更多设计背景、性能数据与踩坑记录见 [project_document.md](project_document.md)。
