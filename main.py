from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.v1.router import api_router
from services.redis_client import redis_client
from core.config import Settings
from core.logger import logger
import uvicorn

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Allyra Chat Service...")
    
    # Test Redis connection
    if not await redis_client.ping():
        logger.error("Failed to connect to Redis")
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    logger.info("Redis connected successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down Allyra Chat Service...")

app = FastAPI(
    title="Allyra Chat Service",
    description="Dynamic AI Teacher-Student Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

@app.get("/")
async def root():
    return {
        "message": "Allyra Chat Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    redis_status = await redis_client.ping()
    return {
        "status": "healthy" if redis_status else "unhealthy",
        "redis": "connected" if redis_status else "disconnected"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )