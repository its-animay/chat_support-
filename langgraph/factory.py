from typing import Dict, Any, Optional
from models.teacher import Teacher
from langgraph import StateGraph
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from core.config import settings
from core.logger import logger

class TeacherAgentState:
    def __init__(self):
        self.messages: list[BaseMessage] = []
        self.teacher_profile: Optional[Dict[str, Any]] = None
        self.context: Dict[str, Any] = {}

class LangGraphAgentFactory:
    @staticmethod
    def build_agent(teacher: Teacher):
        """Build a LangGraph agent for the given teacher profile"""
        try:
            # Initialize LLM
            llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            
            # Create state graph
            def process_message(state: TeacherAgentState) -> Dict[str, Any]:
                # Get the last message
                if not state.messages:
                    return {"response": "Hello! How can I help you today?"}
                
                last_message = state.messages[-1]
                
                # Build system prompt with teacher personality
                system_prompt = f"""
                {teacher.system_prompt}
                
                Remember, you are {teacher.name}, an expert in {teacher.domain}.
                Your personality is: {teacher.personality}
                
                Respond in character, staying true to your expertise and personality.
                """
                
                # Create conversation context
                conversation = [
                    {"role": "system", "content": system_prompt}
                ]
                
                # Add recent message history
                for msg in state.messages[-5:]:  # Last 5 messages for context
                    if isinstance(msg, HumanMessage):
                        conversation.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        conversation.append({"role": "assistant", "content": msg.content})
                
                # Generate response
                response = llm.invoke([
                    HumanMessage(content=msg["content"]) 
                    for msg in conversation
                ])
                
                return {"response": response.content}
            
            # Build the graph
            workflow = StateGraph(TeacherAgentState)
            workflow.add_node("process", process_message)
            workflow.set_entry_point("process")
            workflow.set_finish_point("process")
            
            return workflow.compile()
            
        except Exception as e:
            logger.error(f"Failed to build agent for teacher {teacher.id}: {e}")
            return None
    
    @staticmethod
    def generate_response(teacher: Teacher, messages: list, context: Dict[str, Any] = None) -> str:
        """Generate a response using the teacher's agent"""
        try:
            agent = LangGraphAgentFactory.build_agent(teacher)
            if not agent:
                return "I'm sorry, I'm having trouble right now. Please try again."
            
            # Create state
            state = TeacherAgentState()
            state.teacher_profile = teacher.dict()
            state.context = context or {}
            
            # Convert messages to LangChain format
            for msg in messages:
                if msg.get('role') == 'user':
                    state.messages.append(HumanMessage(content=msg['content']))
                elif msg.get('role') == 'assistant':
                    state.messages.append(AIMessage(content=msg['content']))
            
            # Run the agent
            result = agent.invoke(state)
            return result.get("response", "I apologize, but I couldn't generate a response.")
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I'm experiencing technical difficulties. Please try again."
