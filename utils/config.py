import os

from dotenv import load_dotenv

# 从.env文件加载环境变量，.env文件要放在项目的根目录下
load_dotenv()

# os.getenv：读取环境变量，要load_dotenv()执行后才能生效
qwen_model_name = os.getenv("QWEN_MODEL_NAME")
strategy_llm_model_name = os.getenv("STRATEGY_LLM_MODEL_NAME")
qwen_api_key = os.getenv("QWEN_API_KEY")
qwen_base_url = os.getenv("QWEN_BASE_URL")

embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")

neo4j_url = os.getenv("NEO4J_URL", "")
neo4j_username = os.getenv("NEO4J_USERNAME", "")
neo4j_password = os.getenv("NEO4J_PASSWORD", "")
neo4j_database = os.getenv("NEO4J_DATABASE", "")

es_url = os.getenv("ES_URL", "")
es_index_name = os.getenv("ES_INDEX_NAME", "")


# FastAPI 服务
api_host = os.getenv("API_HOST", "127.0.0.1")
api_port = int(os.getenv("API_PORT", "8000"))


# redis地址
redis_url = os.getenv("REDIS_URL", "")

# ==================== Agent RAG 新增配置 ====================


# 策略分析结果缓存时间（秒）
strategy_cache_ttl = int(os.getenv("STRATEGY_CACHE_TTL", "86400"))

# 进入 CrossEncoder 精排前的最大文档数
max_rerank_docs = int(os.getenv("MAX_RERANK_DOCS", "50"))

# 单路检索默认召回数
default_top_k = int(os.getenv("RETRIEVAL_TOP_K", "20"))

# 返回给Agent的文档摘要最大字符数
snippet_max_chars = int(os.getenv("SNIPPET_MAX_CHARS", "200"))
