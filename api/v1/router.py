from fastapi import APIRouter
from api.v1.endpoints import teacher, chat

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(teacher.router)
api_router.include_router(chat.router)
