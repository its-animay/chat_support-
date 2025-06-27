from typing import List, Dict, Any, Optional, Tuple, BinaryIO, Union
import os
import uuid
import tempfile
import asyncio
import re
from io import BytesIO
from core.config import Settings
from core.logger import logger
from services.document_processor import DocumentProcessor, extract_text_from_html, extract_text_from_markdown

settings = Settings()

class FileProcessor:
    """Service for processing various file formats for RAG"""
    
    @staticmethod
    async def process_file(
        file_content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a file and convert it to documents for RAG
        
        Args:
            file_content: Binary content of the file
            filename: Name of the file with extension
            metadata: Optional metadata to add to documents
            chunk_size: Custom chunk size (or use default from settings)
            chunk_overlap: Custom chunk overlap (or use default from settings)
            
        Returns:
            List of document chunks ready for insertion into vector database
        """
        try:
            # Generate a unique ID for this file
            file_id = str(uuid.uuid4())
            
            # Get file extension (lowercase)
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            # Check if file type is supported
            if ext not in settings.file_processing.allowed_extensions:
                logger.warning(f"Unsupported file type: {ext}")
                error_msg = f"Unsupported file type: {ext}. Please try a supported format: {', '.join(settings.file_processing.allowed_extensions)}"
                return [{
                    "content": error_msg,
                    "metadata": metadata or {}
                }]
            
            # Prepare base metadata
            base_metadata = metadata or {}
            base_metadata.update({
                "source_file": filename,
                "file_id": file_id,
                "file_type": ext
            })
            
            # Extract text based on file type
            if ext in ['.txt', '.text']:
                return await FileProcessor._process_text_file(file_content, base_metadata, chunk_size, chunk_overlap)
            elif ext in ['.md', '.markdown']:
                return await FileProcessor._process_markdown_file(file_content, base_metadata, chunk_size, chunk_overlap)
            elif ext in ['.html', '.htm']:
                return await FileProcessor._process_html_file(file_content, base_metadata, chunk_size, chunk_overlap)
            elif ext in ['.pdf']:
                return await FileProcessor._process_pdf_file(file_content, base_metadata, chunk_size, chunk_overlap)
            elif ext in ['.docx', '.doc']:
                return await FileProcessor._process_docx_file(file_content, base_metadata, chunk_size, chunk_overlap)
            elif ext in ['.csv']:
                return await FileProcessor._process_csv_file(file_content, base_metadata, chunk_size, chunk_overlap)
            elif ext in ['.json']:
                return await FileProcessor._process_json_file(file_content, base_metadata, chunk_size, chunk_overlap)
            elif ext in ['.xlsx', '.xls']:
                return await FileProcessor._process_excel_file(file_content, base_metadata, chunk_size, chunk_overlap)
            else:
                # This shouldn't happen if we've properly checked allowed extensions
                logger.warning(f"Unsupported file type: {ext}")
                return [{
                    "content": f"Unsupported file type: {ext}. Please try a supported format like .txt, .pdf, .docx, .md, .html, .csv, or .json.",
                    "metadata": base_metadata
                }]
                
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}", exc_info=True)
            return [{
                "content": f"Error processing file: {str(e)}",
                "metadata": metadata or {}
            }]
    
    @staticmethod
    async def _process_text_file(
        content: bytes,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process plain text file"""
        try:
            # Decode bytes to string
            text = content.decode('utf-8', errors='replace')
            
            # Create a document
            document = {
                "id": metadata.get("file_id", str(uuid.uuid4())),
                "content": text,
                "metadata": metadata
            }
            
            # Chunk the document
            return DocumentProcessor.chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}", exc_info=True)
            return [{
                "content": f"Error processing text file: {str(e)}",
                "metadata": metadata
            }]
    
    @staticmethod
    async def _process_markdown_file(
        content: bytes,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process Markdown file"""
        try:
            # Decode bytes to string
            md_text = content.decode('utf-8', errors='replace')
            
            # Extract plain text from markdown
            text = extract_text_from_markdown(md_text)
            
            # Create a document
            document = {
                "id": metadata.get("file_id", str(uuid.uuid4())),
                "content": text,
                "metadata": {
                    **metadata,
                    "format": "markdown"
                }
            }
            
            # Chunk the document
            return DocumentProcessor.chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error(f"Error processing markdown file: {e}", exc_info=True)
            return [{
                "content": f"Error processing markdown file: {str(e)}",
                "metadata": metadata
            }]
    
    @staticmethod
    async def _process_html_file(
        content: bytes,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process HTML file"""
        try:
            # Decode bytes to string
            html_text = content.decode('utf-8', errors='replace')
            
            # Extract plain text from HTML
            text = extract_text_from_html(html_text)
            
            # Create a document
            document = {
                "id": metadata.get("file_id", str(uuid.uuid4())),
                "content": text,
                "metadata": {
                    **metadata,
                    "format": "html"
                }
            }
            
            # Chunk the document
            return DocumentProcessor.chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error(f"Error processing HTML file: {e}", exc_info=True)
            return [{
                "content": f"Error processing HTML file: {str(e)}",
                "metadata": metadata
            }]
    
    @staticmethod
    async def _process_pdf_file(
        content: bytes,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process PDF file"""
        try:
            # We'll need PyPDF2 or pdfminer.six for this
            # For this implementation, let's use PyPDF2 which is simpler
            try:
                import PyPDF2
            except ImportError:
                return [{
                    "content": "PDF processing requires PyPDF2 library. Please install it with 'pip install PyPDF2'.",
                    "metadata": metadata
                }]
            
            # Process the PDF
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            
            # Extract text from each page
            all_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                all_text += page.extract_text() + "\n\n"
            
            # Add page count to metadata
            metadata["page_count"] = len(pdf_reader.pages)
            metadata["format"] = "pdf"
            
            # Create a document
            document = {
                "id": metadata.get("file_id", str(uuid.uuid4())),
                "content": all_text,
                "metadata": metadata
            }
            
            # Chunk the document
            return DocumentProcessor.chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}", exc_info=True)
            return [{
                "content": f"Error processing PDF file: {str(e)}",
                "metadata": metadata
            }]
    
    @staticmethod
    async def _process_docx_file(
        content: bytes,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process DOCX file"""
        try:
            # Check if mammoth is available
            try:
                import mammoth
            except ImportError:
                return [{
                    "content": "DOCX processing requires mammoth library. Please install it with 'pip install mammoth'.",
                    "metadata": metadata
                }]
            
            # Convert DOCX to HTML and then extract text
            result = mammoth.convert_to_html(BytesIO(content))
            html = result.value
            text = extract_text_from_html(html)
            
            # Add format to metadata
            metadata["format"] = "docx"
            
            # Create a document
            document = {
                "id": metadata.get("file_id", str(uuid.uuid4())),
                "content": text,
                "metadata": metadata
            }
            
            # Chunk the document
            return DocumentProcessor.chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error(f"Error processing DOCX file: {e}", exc_info=True)
            return [{
                "content": f"Error processing DOCX file: {str(e)}",
                "metadata": metadata
            }]
    
    @staticmethod
    async def _process_csv_file(
        content: bytes,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process CSV file"""
        try:
            # Check if pandas is available
            try:
                import pandas as pd
                import io
            except ImportError:
                return [{
                    "content": "CSV processing requires pandas library. Please install it with 'pip install pandas'.",
                    "metadata": metadata
                }]
            
            # Read CSV with pandas
            df = pd.read_csv(io.BytesIO(content))
            
            # Convert to markdown table format for better readability
            md_table = df.to_markdown(index=False)
            
            # Add CSV stats to metadata
            metadata["format"] = "csv"
            metadata["row_count"] = len(df)
            metadata["column_count"] = len(df.columns)
            metadata["columns"] = df.columns.tolist()
            
            # Create a document
            document = {
                "id": metadata.get("file_id", str(uuid.uuid4())),
                "content": md_table,
                "metadata": metadata
            }
            
            # Chunk the document
            return DocumentProcessor.chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}", exc_info=True)
            return [{
                "content": f"Error processing CSV file: {str(e)}",
                "metadata": metadata
            }]
    
    @staticmethod
    async def _process_json_file(
        content: bytes,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process JSON file"""
        try:
            import json
            
            # Parse JSON
            json_data = json.loads(content)
            
            # Format as string with indentation for readability
            formatted_json = json.dumps(json_data, indent=2)
            
            # Add format to metadata
            metadata["format"] = "json"
            
            # Create a document
            document = {
                "id": metadata.get("file_id", str(uuid.uuid4())),
                "content": formatted_json,
                "metadata": metadata
            }
            
            # Chunk the document
            return DocumentProcessor.chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error(f"Error processing JSON file: {e}", exc_info=True)
            return [{
                "content": f"Error processing JSON file: {str(e)}",
                "metadata": metadata
            }]
    
    @staticmethod
    async def _process_excel_file(
        content: bytes,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process Excel file"""
        try:
            # Check if pandas is available
            try:
                import pandas as pd
                import io
            except ImportError:
                return [{
                    "content": "Excel processing requires pandas and openpyxl libraries. Please install them with 'pip install pandas openpyxl'.",
                    "metadata": metadata
                }]
            
            # Read Excel with pandas
            excel_file = io.BytesIO(content)
            
            # Get sheet names
            xls = pd.ExcelFile(excel_file)
            sheet_names = xls.sheet_names
            
            # Add Excel stats to metadata
            metadata["format"] = "excel"
            metadata["sheet_count"] = len(sheet_names)
            metadata["sheets"] = sheet_names
            
            # Process each sheet as a separate document
            all_chunks = []
            
            for sheet_name in sheet_names:
                # Read sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Convert to markdown table format for better readability
                md_table = f"# Sheet: {sheet_name}\n\n" + df.to_markdown(index=False)
                
                # Create sheet-specific metadata
                sheet_metadata = metadata.copy()
                sheet_metadata["sheet_name"] = sheet_name
                sheet_metadata["row_count"] = len(df)
                sheet_metadata["column_count"] = len(df.columns)
                sheet_metadata["columns"] = df.columns.tolist()
                
                # Create a document
                document = {
                    "id": f"{metadata.get('file_id', str(uuid.uuid4()))}_sheet_{sheet_name}",
                    "content": md_table,
                    "metadata": sheet_metadata
                }
                
                # Chunk the document
                chunks = DocumentProcessor.chunk_document(
                    document=document,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                all_chunks.extend(chunks)
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}", exc_info=True)
            return [{
                "content": f"Error processing Excel file: {str(e)}",
                "metadata": metadata
            }]


# Create singleton instance
file_processor = FileProcessor()