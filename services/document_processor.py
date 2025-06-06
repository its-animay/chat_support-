from typing import List, Dict, Any, Optional, Tuple
import re
from core.config import config
from core.logger import logger

class DocumentProcessor:
    """Handles document preprocessing, chunking, and formatting"""
    
    def __init__(self):
        self.chunk_size = config.rag.chunk_size
        self.chunk_overlap = config.rag.chunk_overlap
    
    def process_document(
        self, 
        document: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a document into chunks with metadata"""
        try:
            # Extract document fields
            doc_id = document.get('id', '')
            text = document.get('text', '')
            metadata = {
                'title': document.get('title', ''),
                'source': document.get('source', ''),
                'domain': document.get('domain', ''),
                'sub_domains': document.get('sub_domains', []),
                'difficulty_level': document.get('difficulty_level', ''),
                'timestamp': document.get('timestamp', ''),
                'tags': document.get('tags', []),
                'doc_id': doc_id
            }
            
            # Chunk the document
            chunks = self._chunk_text(text)
            
            # Create processed chunks with metadata
            processed_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}" if doc_id else f"chunk_{i}"
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = chunk_id
                chunk_metadata['chunk_index'] = i
                chunk_metadata['total_chunks'] = len(chunks)
                
                processed_chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
            
            return processed_chunks
        except Exception as e:
            logger.error(f"Failed to process document: {e}", exc_info=True)
            return []
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is shorter than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size
            
            # If we're at the end of the text, just add the last chunk
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find a good breaking point (end of sentence or paragraph)
            # Look for period, question mark, or exclamation followed by space or newline
            last_period = text.rfind('. ', start, end)
            last_question = text.rfind('? ', start, end)
            last_exclamation = text.rfind('! ', start, end)
            last_newline = text.rfind('\n', start, end)
            
            # Find the latest good breaking point
            break_points = [p for p in [last_period, last_question, last_exclamation, last_newline] if p != -1]
            
            if break_points:
                # Add 2 to include the punctuation and space
                chunk_end = max(break_points) + 2
                
                # For newline, only add 1
                if chunk_end - 2 == last_newline:
                    chunk_end = last_newline + 1
            else:
                # If no good breaking point, break at a space
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    chunk_end = last_space + 1
                else:
                    # If no space, just break at the chunk size
                    chunk_end = end
            
            chunks.append(text[start:chunk_end])
            
            # Start the next chunk with overlap
            start = chunk_end - self.chunk_overlap
            
            # Ensure we're making progress
            if start <= 0 or start >= len(text):
                break
        
        return chunks
    
    def format_retrieved_contexts(
        self, 
        contexts: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> str:
        """Format retrieved contexts into a single string for the LLM"""
        formatted_contexts = []
        
        for i, ctx in enumerate(contexts):
            text = ctx.get('text', '')
            metadata = ctx.get('metadata', {})
            
            if include_metadata:
                title = metadata.get('title', 'Untitled')
                source = metadata.get('source', 'Unknown source')
                
                formatted_ctx = f"[DOCUMENT {i+1}]\nTitle: {title}\nSource: {source}\n\n{text}\n"
            else:
                formatted_ctx = f"[DOCUMENT {i+1}]\n{text}\n"
            
            formatted_contexts.append(formatted_ctx)
        
        return "\n".join(formatted_contexts)
