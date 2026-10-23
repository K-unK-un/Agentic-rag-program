"""
redis服务类
功能：
1. 保存随机验证码 / 读取随机验证码
2. 检索策略分析结果缓存（Agent RAG新增）

self.client.lpush(): 从左边添加
self.client.rpush()：从右边添加
self.client.lpop()：从左边获取
self.client.rpop()：从右边获取
这四个方法组合使用可以实现简单的栈、对列的数据结构操作
"""
import json

import redis

from utils.config import redis_url
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RedisService:
    def __init__(self):
        # decode_responses: 为True表示将redis中utf8字符解码为中文
        self.client = redis.Redis.from_url(
            url=redis_url,
            decode_responses=True
        )

    # 保存验证码
    def save_code(self, email, vcode):
        try:
            key = f"code:{email}"
            self.client.setex(key, 300, vcode)
            return True
        except Exception as e:
            logger.error(f"保存验证码失败: {e}")
            return False

    # 获取验证码
    def get_vcode(self, email):
        try:
            key = f"code:{email}"
            vcode = self.client.get(key)
            return vcode
        except Exception as e:
            logger.error(f"获取验证码失败: {e}")
            return None

    # ==================== 策略分析缓存（Agent RAG） ====================

    # 缓存策略分析结果，同类问题零成本复用；失败不影响主流程
    def set_strategy(self, query_hash: str, strategy: dict, ttl: int = 86400) -> bool:
        try:
            key = f"rag:strategy:{query_hash}"
            self.client.setex(key, ttl, json.dumps(strategy, ensure_ascii=False))
            return True
        except Exception as e:
            logger.warning(f"策略缓存写入失败（降级为无缓存）: {e}")
            return False

    # 读取策略缓存，未命中或异常返回None
    def get_strategy(self, query_hash: str) -> dict | None:
        try:
            key = f"rag:strategy:{query_hash}"
            value = self.client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.warning(f"策略缓存读取失败: {e}")
            return None
