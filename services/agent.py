from typing import List, Dict, Any, Optional, Tuple
import json
import google.generativeai as genai
from core.config import config
from core.logger import logger
from services.vector_store import MilvusVectorStore
from services.document_processor import DocumentProcessor

class RagAgent:
    """Agent for orchestrating RAG retrieval and response generation"""
    
    def __init__(self):
        self.vector_store = MilvusVectorStore()
        self.doc_processor = DocumentProcessor()
        self.top_k = config.rag.default_top_k
        self.similarity_threshold = config.rag.similarity_threshold
        self.max_context_length = config.rag.max_context_length
        self.multi_hop_max_iterations = config.rag.multi_hop_max_iterations
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Initialize the Gemini API client"""
        try:
            genai.configure(api_key=config.gemini.api_key)
            self.generation_model = genai.GenerativeModel(
                model_name=config.gemini.generation_model,
                generation_config={
                    "temperature": config.gemini.temperature,
                    "top_p": config.gemini.top_p,
                    "top_k": config.gemini.top_k,
                    "max_output_tokens": config.gemini.max_tokens
                }
            )
            logger.info(f"Gemini generation model initialized: {config.gemini.generation_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini generation model: {e}", exc_info=True)
            raise
    
    async def analyze_query(
        self, 
        teacher_id: str, 
        query: str,
        teacher_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the query to determine retrieval strategy"""
        try:
            prompt = f"""
            As an AI teaching assistant with expertise in {teacher_info.get('domain', 'various subjects')}, 
            analyze the following query to determine the best retrieval strategy.
            
            Query: "{query}"
            
            Your task is to:
            1. Determine if external knowledge is needed to answer this query accurately
            2. Extract key search terms or questions that would help retrieve relevant information
            3. Identify any filters that should be applied (domain, difficulty level, etc.)
            
            Respond in JSON format with the following structure:
            {{
                "needs_retrieval": true/false,
                "search_queries": ["main query rephrased for search", "optional additional query"],
                "filters": {{"domain": ["specific domain"], "difficulty_level": "appropriate level"}},
                "reasoning": "Brief explanation of your analysis"
            }}
            """
            
            response = self.generation_model.generate_content(prompt)
            
            try:
                # Extract JSON from response
                json_str = response.text
                analysis = json.loads(json_str)
                logger.info(f"Query analysis complete: {analysis.get('reasoning', '')}")
                return analysis
            except json.JSONDecodeError:
                logger.error(f"Failed to parse query analysis response as JSON: {response.text}")
                # Return default analysis
                return {
                    "needs_retrieval": True,
                    "search_queries": [query],
                    "filters": {},
                    "reasoning": "Default analysis due to parsing error"
                }
        except Exception as e:
            logger.error(f"Query analysis failed: {e}", exc_info=True)
            # Return default analysis on error
            return {
                "needs_retrieval": True,
                "search_queries": [query],
                "filters": {},
                "reasoning": "Default analysis due to error"
            }
    
    async def retrieve_context(
        self, 
        teacher_id: str, 
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the vector store"""
        if top_k is None:
            top_k = self.top_k
            
        # Search the vector store
        results = await self.vector_store.search(
            teacher_id=teacher_id,
            query=query,
            top_k=top_k,
            filters=filters,
            similarity_threshold=self.similarity_threshold
        )
        
        return results
    
    async def multi_hop_retrieval(
        self, 
        teacher_id: str, 
        initial_query: str,
        teacher_info: Dict[str, Any],
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform multi-hop retrieval to gather comprehensive context"""
        if max_iterations is None:
            max_iterations = self.multi_hop_max_iterations
            
        # Initialize context and tracking variables
        all_contexts = []
        current_query = initial_query
        iterations = 0
        
        # First, analyze the initial query
        analysis = await self.analyze_query(teacher_id, initial_query, teacher_info)
        
        # If retrieval is not needed, return empty context
        if not analysis.get("needs_retrieval", True):
            return {
                "contexts": [],
                "formatted_context": "",
                "queries_used": [initial_query],
                "complete": True,
                "reasoning": analysis.get("reasoning", "Retrieval not needed for this query")
            }
        
        # Extract search queries and filters from analysis
        search_queries = analysis.get("search_queries", [initial_query])
        filters = analysis.get("filters", {})
        
        # Track all queries used
        all_queries = []
        
        # Perform retrieval iterations
        while iterations < max_iterations:
            iterations += 1
            
            # Get the next query to use
            if search_queries:
                current_query = search_queries.pop(0)
                all_queries.append(current_query)
            else:
                # No more queries to process
                break
                
            # Retrieve context for the current query
            context_results = await self.retrieve_context(
                teacher_id=teacher_id,
                query=current_query,
                filters=filters
            )
            
            # If we found relevant context, add it
            if context_results:
                all_contexts.extend(context_results)
                
            # If we've collected enough context or have no more queries, stop
            if len(all_contexts) >= self.top_k * 2 or not search_queries:
                break
                
            # If we need more context, generate follow-up queries
            if len(all_contexts) < self.top_k and iterations < max_iterations:
                # Format the current context for analysis
                current_context = self.doc_processor.format_retrieved_contexts(
                    all_contexts,
                    include_metadata=False
                )
                
                # Generate follow-up queries if needed
                follow_up_analysis = await self._generate_follow_up_queries(
                    initial_query=initial_query,
                    current_context=current_context,
                    teacher_info=teacher_info
                )
                
                # Add new queries to the queue
                new_queries = follow_up_analysis.get("follow_up_queries", [])
                if new_queries:
                    search_queries.extend(new_queries)
                    
                # Update filters if provided
                new_filters = follow_up_analysis.get("filters", {})
                if new_filters:
                    filters.update(new_filters)
        
        # Deduplicate contexts by ID
        unique_contexts = []
        seen_ids = set()
        for ctx in all_contexts:
            if ctx['id'] not in seen_ids:
                unique_contexts.append(ctx)
                seen_ids.add(ctx['id'])
        
        # Sort by relevance score
        sorted_contexts = sorted(unique_contexts, key=lambda x: x.get('score', 0), reverse=True)
        
        # Limit to top_k results
        final_contexts = sorted_contexts[:self.top_k]
        
        # Format the final context
        formatted_context = self.doc_processor.format_retrieved_contexts(
            final_contexts,
            include_metadata=config.rag.include_metadata
        )
        
        # Truncate if too long
        if len(formatted_context) > self.max_context_length:
            formatted_context = formatted_context[:self.max_context_length] + "...[truncated]"
        
        return {
            "contexts": final_contexts,
            "formatted_context": formatted_context,
            "queries_used": all_queries,
            "complete": True,
            "reasoning": analysis.get("reasoning", "")
        }
    
    async def _generate_follow_up_queries(
        self, 
        initial_query: str,
        current_context: str,
        teacher_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate follow-up queries based on retrieved context"""
        try:
            prompt = f"""
            As an AI teaching assistant with expertise in {teacher_info.get('domain', 'various subjects')}, 
            analyze the following query and the context retrieved so far to determine if follow-up queries are needed.
            
            Original Query: "{initial_query}"
            
            Context Retrieved So Far:
            {current_context}
            
            Your task is to:
            1. Determine if the retrieved context is sufficient to answer the original query
            2. If not, generate follow-up queries to retrieve additional relevant information
            3. Suggest any filters that might help narrow down the results
            
            Respond in JSON format with the following structure:
            {{
                "context_sufficient": true/false,
                "follow_up_queries": ["specific follow-up query 1", "specific follow-up query 2"],
                "filters": {{"domain": ["specific domain"], "difficulty_level": "appropriate level"}},
                "reasoning": "Brief explanation of your analysis"
            }}
            """
            
            response = self.generation_model.generate_content(prompt)
            
            try:
                # Extract JSON from response
                json_str = response.text
                analysis = json.loads(json_str)
                logger.info(f"Follow-up analysis complete: {analysis.get('reasoning', '')}")
                return analysis
            except json.JSONDecodeError:
                logger.error(f"Failed to parse follow-up analysis response as JSON: {response.text}")
                # Return default analysis
                return {
                    "context_sufficient": True,
                    "follow_up_queries": [],
                    "filters": {},
                    "reasoning": "Default analysis due to parsing error"
                }
        except Exception as e:
            logger.error(f"Follow-up analysis failed: {e}", exc_info=True)
            # Return default analysis on error
            return {
                "context_sufficient": True,
                "follow_up_queries": [],
                "filters": {},
                "reasoning": "Default analysis due to error"
            }
    
    async def generate_response(
        self, 
        teacher_id: str,
        query: str,
        contexts: List[Dict[str, Any]],
        teacher_info: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate a response using retrieved context and teacher personality"""
        try:
            # Format the context
            formatted_context = self.doc_processor.format_retrieved_contexts(
                contexts,
                include_metadata=config.rag.include_metadata
            )
            
            # Extract teacher personality info
            teaching_style = teacher_info.get('teaching_style', 'explanatory')
            personality_traits = teacher_info.get('personality_traits', [])
            formality_level = teacher_info.get('formality_level', 'neutral')
            domain = teacher_info.get('domain', 'general knowledge')
            
            # Format chat history if provided
            chat_history_str = ""
            if chat_history and len(chat_history) > 0:
                chat_history_str = "Previous conversation:\n"
                for msg in chat_history[-5:]:  # Include last 5 messages for context
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    chat_history_str += f"{role.capitalize()}: {content}\n"
            
            # Construct the prompt
            prompt = f"""
            You are an AI teacher named {teacher_info.get('name', 'Teacher')} with expertise in {domain}.
            Your teaching style is {teaching_style} and your personality traits include {', '.join(personality_traits)}.
            You communicate in a {formality_level} manner.
            
            {chat_history_str}
            
            The student has asked: "{query}"
            
            I'll provide you with relevant information to help answer this question:
            
            {formatted_context}
            
            Using the information provided above and your teaching expertise, please respond to the student's question.
            If the information is insufficient, use your general knowledge but make it clear what parts are not from the provided context.
            
            Maintain your {teaching_style} teaching style throughout your response:
            - If you are socratic, use guiding questions to help the student discover the answer
            - If you are explanatory, provide clear and detailed explanations
            - If you are practical, focus on real-world applications and examples
            - If you are theoretical, emphasize concepts and principles
            - If you are adaptive, adjust your response based on the student's needs
            
            Your response should be helpful, accurate, and appropriate for the student's question.
            """
            
            # Generate the response
            response = self.generation_model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            logger.error(f"Failed to generate response: {e}", exc_info=True)
            return f"I apologize, but I'm having trouble generating a response at the moment. Could you please try again or rephrase your question?"
