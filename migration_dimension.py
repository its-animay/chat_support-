#!/usr/bin/env python
"""
Simple script to recreate a collection for a specific teacher with the correct dimensions
"""
import argparse
import re
import os
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("milvus-recreate")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("dotenv package not installed. Skipping .env loading.")

def sanitize_id(id_str):
    """Sanitize ID to make it Milvus-compatible (only letters, numbers, underscores)"""
    # Replace hyphens with underscores and remove any other invalid characters
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', id_str)
    
    # Ensure it doesn't start with a number (Milvus requirement)
    if sanitized and sanitized[0].isdigit():
        sanitized = 't_' + sanitized
        
    return sanitized

def recreate_collection(teacher_id, prefix="teacher_", dimension=3072):
    """Recreate a collection for a specific teacher with the specified dimension"""
    # Connect to Milvus
    try:
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530"),
            user=os.getenv("MILVUS_USER"),
            password=os.getenv("MILVUS_PASSWORD"),
            secure=os.getenv("MILVUS_SECURE", "False").lower() == "true"
        )
        logger.info(f"Connected to Milvus")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return False
        
    # Sanitize teacher ID for Milvus collection name
    sanitized_id = sanitize_id(teacher_id)
    collection_name = f"{prefix}{sanitized_id}"
    
    logger.info(f"Working with collection name: {collection_name}")
    
    # Check if collection exists and drop it
    if utility.has_collection(collection_name):
        try:
            utility.drop_collection(collection_name)
            logger.info(f"Dropped existing collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to drop collection {collection_name}: {e}")
            return False
    
    # Create the collection with the correct dimension
    try:
        # Define schema with correct embedding dimension
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Knowledge base for teacher {teacher_id} with {dimension} dimensions"
        )
        
        # Create the collection
        collection = Collection(name=collection_name, schema=schema)
        
        # Create an index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
        
        logger.info(f"Successfully created collection {collection_name} with embedding dimension {dimension}")
        return True
    except Exception as e:
        logger.error(f"Failed to create collection {collection_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Recreate a Milvus collection for a specific teacher")
    parser.add_argument("teacher_id", help="Teacher ID to recreate collection for")
    parser.add_argument("--dimension", type=int, default=3072, help="Embedding dimension (default: 3072)")
    parser.add_argument("--prefix", default="teacher_", help="Collection name prefix")
    
    args = parser.parse_args()
    
    success = recreate_collection(args.teacher_id, args.prefix, args.dimension)
    
    if success:
        logger.info("Collection recreation successful!")
    else:
        logger.error("Collection recreation failed!")

if __name__ == "__main__":
    main()