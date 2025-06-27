from typing import List, Dict, Any, Optional, Union
import asyncio
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from core.config import Settings
from core.logger import logger

settings = Settings()

class LLMService:
    """Service for interacting with language models via APIs"""
    
    def __init__(self):
        """Initialize the LLM service"""
        # Setup OpenAI client if API key is available
        self.openai_client = None
        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Default system prompt for RAG
        self.default_system_prompt = """
        You are a helpful AI assistant with access to retrieved documents. 
        Use the following retrieved documents to answer the user's question.
        If you don't know the answer or the documents don't contain relevant information, 
        say so honestly rather than making up information.
        
        When referencing information from the documents, be specific about which document
        contains the information. You can refer to documents by their numbers.
        """
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate_response(
        self, 
        query: str, 
        retrieved_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate a response using OpenAI API with RAG context
        
        Args:
            query: User's query
            retrieved_documents: List of retrieved documents to use as context
            system_prompt: System prompt to use (optional)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with generated response and metadata
        """
        if not self.openai_client:
            return {
                "response": "LLM service is not configured with valid API credentials.",
                "sources_used": [],
                "rag_enhanced": False
            }
        
        try:
            # Format retrieved documents as context
            context = self._format_documents_as_context(retrieved_documents)
            
            # Create final prompt with context
            final_prompt = f"Context information:\n{context}\n\nUser question: {query}"
            
            # Use provided system prompt or default
            system_content = system_prompt or self.default_system_prompt
            
            # Call OpenAI API
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": final_prompt}
                ]
            )
            
            # Extract sources used in generation
            sources = []
            if retrieved_documents:
                sources = [
                    {
                        "id": doc.get("id", ""),
                        "content": doc.get("content", "")[:200] + "...",  # Truncate content preview
                        "metadata": doc.get("metadata", {}),
                        "score": doc.get("rerank_score", doc.get("score", 0))
                    }
                    for doc in retrieved_documents[:3]  # Include top 3 sources at most
                ]
            
            return {
                "response": response.choices[0].message.content,
                "sources_used": sources,
                "rag_enhanced": len(retrieved_documents) > 0,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, 'usage') else {}
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            return {
                "response": f"I'm sorry, I encountered an error while processing your request. Please try again later.",
                "sources_used": [],
                "rag_enhanced": False,
                "error": str(e)
            }
    
    def _format_documents_as_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string for the LLM"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})
            
            # Include document number and optional title
            header = f"Document {i+1}"
            if metadata and "title" in metadata:
                header += f" - {metadata['title']}"
            
            # Add source information if available
            source_info = ""
            if metadata:
                if "source" in metadata:
                    source_info = f"Source: {metadata['source']}"
                elif "url" in metadata:
                    source_info = f"Source: {metadata['url']}"
            
            # Format the document entry
            doc_entry = f"{header}:\n{content}"
            if source_info:
                doc_entry += f"\n{source_info}"
            
            context_parts.append(doc_entry)
        
        # Join all document entries with separators
        return "\n\n" + "\n\n---\n\n".join(context_parts)


# Global singleton instance
llm_service = LLMService()