from typing import List, Dict, Any, Optional, Generator, Tuple
import uuid
import re
from core.config import Settings
from core.logger import logger

settings = Settings()

class DocumentProcessor:
    """Service for processing and chunking documents for RAG"""
    
    @staticmethod
    def chunk_document(
        document: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None,
        max_content_size: int = 65000  # Just below Milvus limit of 65535
    ) -> List[Dict[str, Any]]:
        """
        Split a document into smaller chunks with overlap
        
        Args:
            document: Document dict with 'content' and optional 'metadata'
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            max_content_size: Maximum content size for each chunk
            
        Returns:
            List of document chunks
        """
        content = document.get('content', '')
        if not content:
            return []
            
        # Use settings if not specified
        chunk_size = chunk_size or settings.rag.chunk_size
        chunk_overlap = chunk_overlap or settings.rag.chunk_overlap
        
        # Ensure chunk_size doesn't exceed max_content_size
        chunk_size = min(chunk_size, max_content_size)
        
        # Get document metadata
        metadata = document.get('metadata', {}).copy()
        
        # Ensure document ID is compatible with Milvus (max 36 chars)
        original_doc_id = document.get('id', str(uuid.uuid4()))
        if len(original_doc_id) > 36:
            # Save original ID in metadata if needed
            metadata['original_id'] = original_doc_id
            # Use a truncated or new ID that fits the 36-char limit
            doc_id = original_doc_id[:36]
        else:
            doc_id = original_doc_id
        
        # Split content into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If content is small enough, return as single chunk
        if len(content) <= chunk_size:
            # Still need to ensure content doesn't exceed Milvus limits
            if len(content) > max_content_size:
                truncated_content = content[:max_content_size - 50] + "\n\n[Content truncated due to size limits]"
                metadata['original_content_length'] = len(content)
                metadata['content_truncated'] = True
                logger.warning(f"Content truncated for document {doc_id}: {len(content)} chars -> {len(truncated_content)} chars")
                content = truncated_content
                
            return [{
                'id': f"{doc_id[:30]}-chunk-1",  # Ensure ID length constraints
                'content': content,
                'metadata': {
                    **metadata,
                    'parent_id': doc_id,
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            }]
        
        # Process paragraphs into chunks
        chunks = []
        current_chunk = ""
        current_paragraphs = []
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, create a new chunk
            if current_chunk and len(current_chunk) + len(paragraph) + 1 > chunk_size:
                # Save current chunk (ensure it doesn't exceed max_content_size)
                if len(current_chunk) > max_content_size:
                    # This shouldn't happen with reasonable chunk_size settings,
                    # but we'll handle it just in case
                    truncated_chunk = current_chunk[:max_content_size - 50] + "\n\n[Content truncated due to size limits]"
                    chunk_metadata = {
                        **metadata,
                        'parent_id': doc_id,
                        'chunk_index': chunk_index,
                        'paragraph_indices': list(range(len(current_paragraphs))),
                        'original_content_length': len(current_chunk),
                        'content_truncated': True
                    }
                    logger.warning(f"Chunk content truncated: {len(current_chunk)} chars -> {len(truncated_chunk)} chars")
                    current_chunk = truncated_chunk
                else:
                    chunk_metadata = {
                        **metadata,
                        'parent_id': doc_id,
                        'chunk_index': chunk_index,
                        'paragraph_indices': list(range(len(current_paragraphs))),
                    }
                
                chunks.append({
                    'id': f"{doc_id[:28]}-c{chunk_index + 1}",  # Shortened to ensure ID length constraints
                    'content': current_chunk,
                    'metadata': chunk_metadata
                })
                
                # Start a new chunk with overlap
                overlap_start = max(0, len(current_paragraphs) - chunk_overlap)
                current_paragraphs = current_paragraphs[overlap_start:]
                current_chunk = "\n\n".join(current_paragraphs)
                chunk_index += 1
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            
            current_paragraphs.append(paragraph)
        
        # Add the last chunk if there's anything left
        if current_chunk:
            # Ensure the last chunk doesn't exceed max_content_size
            if len(current_chunk) > max_content_size:
                truncated_chunk = current_chunk[:max_content_size - 50] + "\n\n[Content truncated due to size limits]"
                chunk_metadata = {
                    **metadata,
                    'parent_id': doc_id,
                    'chunk_index': chunk_index,
                    'paragraph_indices': list(range(len(current_paragraphs))),
                    'original_content_length': len(current_chunk),
                    'content_truncated': True
                }
                logger.warning(f"Final chunk content truncated: {len(current_chunk)} chars -> {len(truncated_chunk)} chars")
                current_chunk = truncated_chunk
            else:
                chunk_metadata = {
                    **metadata,
                    'parent_id': doc_id,
                    'chunk_index': chunk_index,
                    'paragraph_indices': list(range(len(current_paragraphs))),
                }
                
            chunks.append({
                'id': f"{doc_id[:28]}-c{chunk_index + 1}",  # Shortened to ensure ID length constraints
                'content': current_chunk,
                'metadata': chunk_metadata
            })
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    @staticmethod
    def process_documents(
        documents: List[Dict[str, Any]],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents into chunks
        
        Args:
            documents: List of document dicts
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of document chunks
        """
        all_chunks = []
        
        for doc in documents:
            # Ensure document has an ID
            if 'id' not in doc:
                doc['id'] = str(uuid.uuid4())
            
            # Chunk the document
            chunks = DocumentProcessor.chunk_document(
                document=doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            all_chunks.extend(chunks)
        
        return all_chunks


# Helper functions for specific document formats
def extract_text_from_html(html_content: str) -> str:
    """Extract plain text from HTML content"""
    try:
        # Simple regex-based HTML tag removal
        # For production, consider using a proper HTML parser like BeautifulSoup
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {e}", exc_info=True)
        return html_content


def extract_text_from_markdown(markdown_content: str) -> str:
    """Extract plain text from Markdown content"""
    try:
        # Simple regex-based Markdown removal
        # Remove headers
        text = re.sub(r'#{1,6}\s+', '', markdown_content)
        # Remove emphasis
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        # Remove links
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        # Remove code blocks
        text = re.sub(r'```(.+?)```', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'`(.+?)`', r'\1', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from Markdown: {e}", exc_info=True)
        return markdown_content


# Create singleton instance
document_processor = DocumentProcessor()