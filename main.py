from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.chat_api import chat_router

from utils.logger import setup_logger
import uvicorn

logger = setup_logger(__name__)

app = FastAPI()
app.include_router(chat_router, prefix="/api")

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
