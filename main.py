from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.v1.router import api_router
from services.redis_client import redis_client
from services.milvus_client import milvus_client
from prometheus_fastapi_instrumentator import Instrumentator
from core.config import Settings
from core.logger import logger
import uvicorn
import time

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting MTXOLABS Chat Service...")
    
    # Test Redis connection
    if not await redis_client.ping():
        logger.error("Failed to connect to Redis")
        logger.warning("Redis fallback mechanisms will be used")
    else:
        logger.info("Redis connected successfully")
    
    # Connect to Milvus and initialize RAG components
    try:
        milvus_connected = await milvus_client.connect()
        if milvus_connected:
            logger.info("Milvus connected successfully")
            # Ensure RAG collection exists
            collection_created = await milvus_client.create_collection()
            if collection_created:
                logger.info("RAG collection initialized successfully")
            else:
                logger.warning("Failed to initialize RAG collection, some features may be limited")
        else:
            logger.warning("Failed to connect to Milvus, RAG features will be limited")
    except Exception as e:
        logger.error(f"Error connecting to Milvus: {e}", exc_info=True)
        logger.warning("RAG features will be disabled")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MTXOLABS Chat Service...")

app = FastAPI(
    title="MTXOLABS Chat Service",
    description="Dynamic AI Teacher-Student Platform with RAG Capabilities",
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

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Include API routes
app.include_router(api_router)

@app.get("/")
async def root():
    return {
        "message": "MTXOLABS Chat Service",
        "version": "1.0.0",
        "status": "running",
        "features": ["AI Teachers", "Dynamic Chat", "RAG Enhancement"]
    }

@app.get("/health")
async def health_check():
    start_time = time.time()
    
    # Check Redis
    redis_status = await redis_client.health_check()
    
    # Check Milvus
    try:
        milvus_status = await milvus_client.health_check()
    except Exception as e:
        logger.error(f"Error checking Milvus health: {e}")
        milvus_status = {
            "connected": False,
            "error": str(e)
        }
    
    return {
        "status": "healthy" if redis_status.get("connected") and milvus_status.get("connected", False) else "degraded",
        "services": {
            "redis": redis_status,
            "milvus": milvus_status
        },
        "response_time": time.time() - start_time
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )