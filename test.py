import requests
import json
import time
import uuid
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your API server URL
TEACHER_ID = "27011fc5-de80-4d27-ae01-574493f8ad97"  # Your UUID teacher ID
USER_ID = "test-user-id"

# Helper functions
def print_separator():
    print("\n" + "="*80 + "\n")

def print_response(response, operation):
    print(f"⚡ {operation}")
    print(f"Status: {response.status_code}")
    
    try:
        json_response = response.json()
        if 'content' in json_response and len(json_response['content']) > 300:
            # Truncate long content for display
            display_content = json_response['content'][:300] + "... [truncated]"
            json_response['content'] = display_content
            
        print(f"Response: {json.dumps(json_response, indent=2)}")
        
        # Return the full response for further processing
        return response.json()
    except:
        print(f"Response: {response.text}")
        return None

def test_query(chat_id, query, headers=None):
    """Send a query and display the response"""
    if headers is None:
        headers = {
            "Content-Type": "application/json",
            "x-user-id": USER_ID
        }
    
    url = f"{BASE_URL}/api/v1/chat/{chat_id}/send"
    payload = {
        "content": query,
        "metadata": {}
    }
    
    print(f"\n🔍 Testing query: \"{query}\"")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        return print_response(response, "Query Response")
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def start_chat_session():
    """Start a new chat session with the teacher"""
    url = f"{BASE_URL}/api/v1/chat/start"
    headers = {
        "Content-Type": "application/json",
        "x-user-id": USER_ID
    }
    payload = {
        "teacher_id": TEACHER_ID,
        "title": "RAG Query Testing Session"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        result = print_response(response, "Start Chat Session")
        
        if result and 'id' in result:
            return result['id']
        else:
            print("❌ Failed to extract chat ID")
            return None
    except Exception as e:
        print(f"❌ Error starting chat: {e}")
        return None

def get_message_sources(chat_id, message_id):
    """Get the sources used for a message"""
    url = f"{BASE_URL}/api/v1/chat/{chat_id}/message/{message_id}/sources"
    headers = {
        "x-user-id": USER_ID
    }
    
    try:
        response = requests.get(url, headers=headers)
        return print_response(response, "Message Sources")
    except Exception as e:
        print(f"❌ Error getting sources: {e}")
        return None

def run_test_queries():
    """Run a series of test queries to evaluate the RAG system"""
    print("\n🧪 RUNNING RAG SYSTEM TEST QUERIES\n")
    print_separator()
    
    # Start a chat session
    chat_id = start_chat_session()
    if not chat_id:
        print("❌ Cannot proceed without a chat session")
        return
    
    print(f"\n📝 Chat session created with ID: {chat_id}")
    print_separator()
    
    # Define test queries across different domains and complexity levels
    test_queries = [
        # Programming queries
        {
            "query": "What are the key principles of Object-Oriented Programming?",
            "description": "Basic factual query about programming concepts"
        },
        {
            "query": "Explain how RESTful APIs work and what are the best practices for designing them.",
            "description": "Intermediate level query requiring comprehensive knowledge"
        },
        {
            "query": "Compare and contrast functional programming with object-oriented programming. What are the strengths and weaknesses of each?",
            "description": "Complex query requiring synthesis of multiple concepts"
        },
        
        # Mathematics queries
        {
            "query": "What is calculus and what are its main branches?",
            "description": "Basic factual query about mathematics"
        },
        {
            "query": "Explain how eigenvalues and eigenvectors are used in linear algebra and their real-world applications.",
            "description": "Intermediate level mathematical concept query"
        },
        {
            "query": "How does topology relate to modern physics, particularly in quantum field theory?",
            "description": "Advanced interdisciplinary query"
        },
        
        # Mixed domain queries
        {
            "query": "How is linear algebra used in machine learning algorithms?",
            "description": "Cross-domain query connecting mathematics and programming"
        },
        {
            "query": "What mathematical concepts are essential for understanding cryptography and secure communications?",
            "description": "Applied mathematics and computer security query"
        },
        
        # Meta-queries about the knowledge
        {
            "query": "What sources are you using for your information about programming languages?",
            "description": "Query about the knowledge sources themselves"
        }
    ]
    
    # Run each test query
    results = []
    for i, test in enumerate(test_queries):
        print(f"\nTEST QUERY {i+1}: {test['description']}")
        print(f"Query: {test['query']}")
        
        response = test_query(chat_id, test['query'])
        
        # Track results and check sources if available
        if response and 'message_id' in response:
            message_id = response['message_id']
            
            # Check if response contains source information
            metadata = response.get('metadata', {})
            if metadata.get('rag_enhanced', False):
                sources = metadata.get('sources_used', [])
                sources_count = len(sources)
                print(f"\n📚 Response used {sources_count} sources")
                
                # Get detailed source information
                source_info = get_message_sources(chat_id, message_id)
                
                results.append({
                    "query": test['query'],
                    "success": True,
                    "sources_used": sources_count,
                    "rag_enhanced": metadata.get('rag_enhanced', False)
                })
            else:
                print("\n⚠️ Response was not RAG-enhanced")
                results.append({
                    "query": test['query'],
                    "success": True,
                    "sources_used": 0,
                    "rag_enhanced": False
                })
        else:
            results.append({
                "query": test['query'],
                "success": False,
                "sources_used": 0,
                "rag_enhanced": False
            })
        
        print_separator()
        
        # Add a small delay between queries
        time.sleep(1)
    
    # Print summary
    print("\n📊 TEST QUERY SUMMARY\n")
    print(f"Total Queries: {len(test_queries)}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"Successful Responses: {successful}/{len(test_queries)}")
    
    rag_enhanced = sum(1 for r in results if r['rag_enhanced'])
    print(f"RAG-Enhanced Responses: {rag_enhanced}/{len(test_queries)}")
    
    avg_sources = sum(r['sources_used'] for r in results) / len(results)
    print(f"Average Sources Used: {avg_sources:.2f}")
    
    print("\n✅ RAG SYSTEM TEST QUERIES COMPLETED\n")

if __name__ == "__main__":
    run_test_queries()