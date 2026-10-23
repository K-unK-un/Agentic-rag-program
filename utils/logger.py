import logging

# 全局日志打印器
def setup_logger(name: str = "rag_system") -> logging.Logger:
    # 初始化logger对象
    logger = logging.getLogger(name)
    # 设置日志级别
    logger.setLevel(logging.DEBUG)


    if not logger.handlers:
        # StreamHandler: 初始化控制台日志打印
        # FileHandler: 文件日志控制器，将日志保存到本地文件中
        handler = logging.StreamHandler()

        handler.setLevel(logging.DEBUG)
        # Formatter：设置日志打印格式
        # asctime：时间
        # levelname：日志级别
        # name：程序执行的文件名
        # message：日志消息
        # datefmt：日期格式化
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
