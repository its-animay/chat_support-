from typing import List, Dict, Any, Optional
from services.rag_pipeline import rag_pipeline
from models.chat import Message, MessageRole
from core.logger import logger
import json

class ChatRAGIntegration:
    """Integration between chat service and RAG pipeline"""
    
    @staticmethod
    async def enhance_response_with_rag(
        user_message: str,
        chat_history: List[Message],
        teacher_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance a response using RAG pipeline
        
        Args:
            user_message: The user's message to respond to
            chat_history: Previous messages in the conversation
            teacher_id: ID of the teacher for the chat
            context: Additional context for response generation
        
        Returns:
            Dict containing the enhanced response and metadata
        """
        try:
            # Construct system prompt from context
            formality = context.get("formality_level", "neutral")
            teaching_style = context.get("teaching_style", "explanatory")
            domain = context.get("domain", "general")
            
            system_prompt = (
                f"You are an AI teacher specializing in {domain}. "
                f"Your teaching style is {teaching_style} and your formality level is {formality}. "
                "Use the retrieved information to provide an accurate and helpful response. "
                "When using information from the sources, be sure to integrate it naturally into your response."
            )
            
            # Extract recent conversation history for context
            formatted_history = []
            for msg in chat_history[-5:]:  # Use last 5 messages for context
                if msg.role != MessageRole.SYSTEM:  # Skip system messages
                    formatted_history.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })
            
            # Process through RAG pipeline
            rag_result = await rag_pipeline.process_query(
                query=user_message,
                top_k=context.get("top_k", 10),
                top_n=context.get("top_n", 3),
                temperature=0.7,
                teacher_id=teacher_id,
                system_prompt=system_prompt
            )
            
            # Extract necessary information from result
            response = {
                "content": rag_result.get("response", "I'm sorry, I couldn't generate a response."),
                "sources_used": rag_result.get("sources_used", []),
                "rag_enhanced": True,
                "processing_time": rag_result.get("processing_time", 0),
                "metadata": {
                    "teacher_id": teacher_id,
                    "rag_enhanced": True,
                    "sources_used": rag_result.get("sources_used", []),
                    "retrieval_count": rag_result.get("retrieval_count", 0),
                    "reranked_count": rag_result.get("reranked_count", 0),
                    "cached": rag_result.get("cached", False)
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"RAG enhancement failed: {e}", exc_info=True)
            
            # Fallback response
            return {
                "content": (
                    "I'm having trouble accessing my knowledge base right now. "
                    f"Let me answer based on what I already know about {context.get('domain', 'this topic')}."
                ),
                "sources_used": [],
                "rag_enhanced": False,
                "processing_time": 0,
                "metadata": {
                    "teacher_id": teacher_id,
                    "rag_enhanced": False,
                    "error": str(e)
                }
            }
    
    @staticmethod
    async def should_use_rag(message_data: Dict[str, Any], teacher: Any) -> bool:
        """
        Determine if RAG should be used for this message
        
        Args:
            message_data: The message data including metadata
            teacher: The teacher object with settings
        
        Returns:
            Boolean indicating whether to use RAG
        """
        # Explicit request in message metadata takes precedence
        if message_data.get("metadata") and "use_rag" in message_data["metadata"]:
            return message_data["metadata"]["use_rag"]
        
        # Otherwise, use teacher's default setting
        return teacher.specialization.enable_rag if hasattr(teacher.specialization, "enable_rag") else False
    
    @staticmethod
    def format_sources_for_response(sources: List[Dict[str, Any]]) -> str:
        """
        Format source information for including in response
        
        Args:
            sources: List of source documents
        
        Returns:
            Formatted string with source information
        """
        if not sources:
            return ""
            
        source_text = "\n\nSources:\n"
        for i, source in enumerate(sources, 1):
            title = source.get("metadata", {}).get("title", f"Source {i}")
            author = source.get("metadata", {}).get("author", "Unknown")
            date = source.get("metadata", {}).get("date", "")
            
            source_text += f"{i}. {title}"
            if author != "Unknown":
                source_text += f" by {author}"
            if date:
                source_text += f" ({date})"
            source_text += "\n"
            
        return source_text