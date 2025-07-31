# factory.py - Updated version
from typing import Dict, Any, Optional, List
from models.teacher import EnhancedTeacher
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from core.config import Settings
from core.logger import logger
import asyncio

settings = Settings()

class LangGraphAgentFactory:
    """Factory class for creating and running agents with teacher personalities"""
    
    @staticmethod
    async def generate_response(teacher: EnhancedTeacher, messages: List[Dict[str, str]], context: Dict[str, Any] = None) -> str:
        """Generate a response using the teacher's personality asynchronously"""
        try:
            llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model="chatgpt-4o-latest",
                temperature=0.7
            )
            
            system_prompt = teacher.generate_system_prompt(context or {})
            
            # Check if RAG is enabled for this teacher
            use_rag = teacher.specialization.enable_rag
            
            # Override with message-specific setting if present
            if context and 'use_rag' in context:
                use_rag = context.get('use_rag')
            
            # Get the latest user message
            latest_user_message = None
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    latest_user_message = msg.get('content', '')
                    break
            
            # If RAG is enabled and we have a user message, enhance with RAG
            rag_context = ""
            if use_rag and latest_user_message:
                from services.rag_pipeline import rag_pipeline
                
                # Get RAG results from teacher's knowledge base
                rag_result = await rag_pipeline.process_query(
                    query=latest_user_message,
                    top_k=5,
                    top_n=3,
                    temperature=0.0,  # Just get context, not response
                    teacher_id=teacher.id  # Use teacher-specific knowledge base
                )
                
                # Extract context from retrieved documents
                if rag_result.get("sources_used"):
                    rag_context = "\n\nRelevant information from my knowledge base:\n"
                    for i, source in enumerate(rag_result.get("sources_used", [])):
                        # Include source title if available
                        title = source.get("metadata", {}).get("title", f"Source {i+1}")
                        rag_context += f"\n---\n{title}:\n{source.get('content', '')}\n---\n"
            
            # Append RAG context to system prompt if available
            if rag_context:
                system_prompt += rag_context
            
            
            formatted_messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            for msg in messages[-10:]:  
                role = msg.get('role')
                content = msg.get('content', '')
                if role in ['user', 'assistant']:
                    formatted_messages.append({"role": role, "content": content})
            
            response = await llm.ainvoke(formatted_messages)  
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}", exc_info=True)
            
            style = teacher.personality.teaching_style.value
            domain = teacher.specialization.primary_domain
            
            if style == "socratic":
                return f"I'm having trouble processing your question about {domain} right now. Perhaps we could approach this from a different angle? What specific aspect interests you most?"
            elif style == "explanatory":
                return f"I apologize, but I'm experiencing technical difficulties at the moment. In the meantime, could you clarify what you'd like to learn about {domain}?"
            elif style == "practical":
                return f"I'm sorry, I'm having a technical issue. While we wait, could you tell me more about what practical {domain} problem you're trying to solve?"
            else:
                return f"I apologize for the technical difficulties. I'd still like to help you with your {domain} questions when the system is back online."