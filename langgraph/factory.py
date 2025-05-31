from typing import Dict, Any, Optional, List
from models.teacher import EnhancedTeacher
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from core.config import Settings
from core.logger import logger

settings = Settings()

class LangGraphAgentFactory:
    """Factory class for creating and running agents with teacher personalities"""
    
    @staticmethod
    def generate_response(teacher: EnhancedTeacher, messages: List[Dict[str, str]], context: Dict[str, Any] = None) -> str:
        """Generate a response using the teacher's personality"""
        try:
            llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            
            system_prompt = teacher.generate_system_prompt(context or {})
            
            formatted_messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            for msg in messages[-10:]:  
                role = msg.get('role')
                content = msg.get('content', '')
                if role in ['user', 'assistant']:
                    formatted_messages.append({"role": role, "content": content})
            
            # Generate response
            response = llm.invoke(formatted_messages)
            
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