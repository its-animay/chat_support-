#!/usr/bin/env python3
"""
Script to clean all Milvus collections and documents.
This script will drop all collections but keep the Milvus installation intact.
"""

import os
import sys
import time
from pymilvus import connections, utility

# Text formatting for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

def print_color(color, message):
    """Print colored message"""
    print(f"{color}{message}{Colors.ENDC}")

def connect_to_milvus(host="localhost", port="19530"):
    """Connect to Milvus server"""
    print_color(Colors.BLUE, f"Connecting to Milvus at {host}:{port}...")
    try:
        connections.connect(
            alias="default", 
            host=host, 
            port=port
        )
        print_color(Colors.GREEN, "Connected to Milvus successfully.")
        return True
    except Exception as e:
        print_color(Colors.RED, f"Failed to connect to Milvus: {e}")
        return False

def list_collections():
    """List all collections in Milvus"""
    try:
        collections = utility.list_collections()
        if collections:
            print_color(Colors.BLUE, f"Found {len(collections)} collections:")
            for i, collection in enumerate(collections, 1):
                print(f"  {i}. {collection}")
        else:
            print_color(Colors.YELLOW, "No collections found in Milvus.")
        return collections
    except Exception as e:
        print_color(Colors.RED, f"Error listing collections: {e}")
        return []

def drop_collection(collection_name):
    """Drop a specific collection"""
    try:
        if utility.has_collection(collection_name):
            print_color(Colors.YELLOW, f"Dropping collection: {collection_name}")
            utility.drop_collection(collection_name)
            print_color(Colors.GREEN, f"Collection {collection_name} dropped successfully.")
            return True
        else:
            print_color(Colors.YELLOW, f"Collection {collection_name} does not exist.")
            return False
    except Exception as e:
        print_color(Colors.RED, f"Error dropping collection {collection_name}: {e}")
        return False

def drop_all_collections():
    """Drop all collections in Milvus"""
    collections = list_collections()
    if not collections:
        return
    
    print_color(Colors.YELLOW, "Preparing to drop all collections...")
    
    # Ask for confirmation
    confirm = input("Are you sure you want to drop all collections? This cannot be undone. (y/n): ")
    if confirm.lower() != 'y':
        print_color(Colors.BLUE, "Operation cancelled.")
        return
    
    success_count = 0
    failure_count = 0
    
    for collection in collections:
        if drop_collection(collection):
            success_count += 1
        else:
            failure_count += 1
    
    print_color(Colors.GREEN, f"Successfully dropped {success_count} collections.")
    if failure_count > 0:
        print_color(Colors.YELLOW, f"Failed to drop {failure_count} collections.")

def drop_teacher_collections():
    """Drop only teacher-related collections"""
    collections = list_collections()
    if not collections:
        return
    
    teacher_collections = [c for c in collections if c.startswith('teacher_')]
    
    if not teacher_collections:
        print_color(Colors.YELLOW, "No teacher collections found.")
        return
    
    print_color(Colors.YELLOW, f"Found {len(teacher_collections)} teacher collections:")
    for i, collection in enumerate(teacher_collections, 1):
        print(f"  {i}. {collection}")
    
    # Ask for confirmation
    confirm = input("Are you sure you want to drop all teacher collections? This cannot be undone. (y/n): ")
    if confirm.lower() != 'y':
        print_color(Colors.BLUE, "Operation cancelled.")
        return
    
    success_count = 0
    failure_count = 0
    
    for collection in teacher_collections:
        if drop_collection(collection):
            success_count += 1
        else:
            failure_count += 1
    
    print_color(Colors.GREEN, f"Successfully dropped {success_count} teacher collections.")
    if failure_count > 0:
        print_color(Colors.YELLOW, f"Failed to drop {failure_count} teacher collections.")

def main():
    """Main function to clean Milvus collections"""
    print_color(Colors.BLUE, "=== Milvus Collection Cleanup Tool ===")
    
    # Get Milvus connection details
    host = os.environ.get("MILVUS_HOST", "localhost")
    port = os.environ.get("MILVUS_PORT", "19530")
    
    # Connect to Milvus
    if not connect_to_milvus(host, port):
        sys.exit(1)
    
    # Show menu
    while True:
        print("\nPlease select an option:")
        print("1. List all collections")
        print("2. Drop a specific collection")
        print("3. Drop all collections")
        print("4. Drop only teacher collections")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            list_collections()
        elif choice == '2':
            collection_name = input("Enter collection name to drop: ")
            drop_collection(collection_name)
        elif choice == '3':
            drop_all_collections()
        elif choice == '4':
            drop_teacher_collections()
        elif choice == '5':
            print_color(Colors.GREEN, "Exiting. Goodbye!")
            break
        else:
            print_color(Colors.RED, "Invalid choice. Please try again.")

if __name__ == "__main__":
    main()