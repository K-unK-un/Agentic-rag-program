import os

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

from utils.config import qwen_model_name, qwen_api_key, qwen_base_url, embeddings_model_name, strategy_llm_model_name

# 会话模型
qwen_llm = ChatOpenAI(
    model=qwen_model_name,
    api_key=qwen_api_key,
    base_url=qwen_base_url
)

# 使用gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# qwen_llm = ChatOpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama",  # 任意值
#     model="qwen3:8b",
# )

strategy_llm = ChatOpenAI(
    model=strategy_llm_model_name,
    api_key=qwen_api_key,
    base_url=qwen_base_url,
    temperature=0
)

# 词嵌入模型
embeddings = OllamaEmbeddings(
    model=embeddings_model_name
)
