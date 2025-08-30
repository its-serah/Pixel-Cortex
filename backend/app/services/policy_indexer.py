import os
import hashlib
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import markdown
from PyPDF2 import PdfReader
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.models import PolicyDocument, PolicyChunk

class PolicyIndexer:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
    
    def calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def read_markdown_file(self, file_path: Path) -> str:
        """Read and parse markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert markdown to plain text for indexing
        html = markdown.markdown(content)
        # Remove HTML tags (simple approach)
        import re
        clean_text = re.sub(r'<[^>]+>', '', html)
        return clean_text.strip()
    
    def read_pdf_file(self, file_path: Path) -> str:
        """Read and extract text from PDF file"""
        try:
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def read_text_file(self, file_path: Path) -> str:
        """Read plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """Split text into chunks of approximately equal size"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        # Simple sentence-aware chunking
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def index_document(self, file_path: Path, db: Session) -> PolicyDocument:
        """Index a single policy document"""
        filename = file_path.name
        file_type = file_path.suffix.lower().lstrip('.')
        
        # Read file based on type
        if file_type == 'md':
            content = self.read_markdown_file(file_path)
            title = filename.replace('.md', '').replace('_', ' ').replace('-', ' ').title()
        elif file_type == 'pdf':
            content = self.read_pdf_file(file_path)
            title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
        elif file_type in ['txt', 'text']:
            content = self.read_text_file(file_path)
            title = filename.replace('.txt', '').replace('_', ' ').replace('-', ' ').title()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        if not content:
            print(f"No content extracted from {filename}")
            return None
        
        content_hash = self.calculate_hash(content)
        
        # Check if document already exists and is unchanged
        existing_doc = db.query(PolicyDocument).filter(PolicyDocument.filename == filename).first()
        if existing_doc and existing_doc.content_hash == content_hash:
            print(f"Document {filename} is up to date, skipping")
            return existing_doc
        
        # Delete existing document and chunks if updating
        if existing_doc:
            db.query(PolicyChunk).filter(PolicyChunk.document_id == existing_doc.id).delete()
            db.delete(existing_doc)
            db.commit()
        
        # Create new document
        doc = PolicyDocument(
            filename=filename,
            title=title,
            content=content,
            content_hash=content_hash,
            file_type=file_type
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        # Chunk the content
        chunks = self.chunk_text(content)
        
        # Create chunk records
        for i, chunk_content in enumerate(chunks):
            chunk_hash = self.calculate_hash(chunk_content)
            
            chunk = PolicyChunk(
                document_id=doc.id,
                chunk_index=i,
                content=chunk_content,
                content_hash=chunk_hash
            )
            db.add(chunk)
        
        db.commit()
        print(f"Indexed document {filename} with {len(chunks)} chunks")
        return doc
    
    async def index_policies_directory(self, policies_dir: str):
        """Index all policy documents in a directory"""
        policies_path = Path(policies_dir)
        
        if not policies_path.exists():
            print(f"Policies directory {policies_dir} does not exist")
            return
        
        db = SessionLocal()
        try:
            supported_extensions = ['.md', '.pdf', '.txt']
            
            for file_path in policies_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        self.index_document(file_path, db)
                    except Exception as e:
                        print(f"Error indexing {file_path}: {e}")
            
            # Get total count
            total_docs = db.query(PolicyDocument).count()
            total_chunks = db.query(PolicyChunk).count()
            print(f"Indexing complete: {total_docs} documents, {total_chunks} chunks")
            
        finally:
            db.close()
    
    def reindex_all_policies(self, policies_dir: str):
        """Force reindex of all policies (delete existing first)"""
        db = SessionLocal()
        try:
            # Delete all existing chunks and documents
            db.query(PolicyChunk).delete()
            db.query(PolicyDocument).delete()
            db.commit()
            print("Cleared existing policy index")
            
        finally:
            db.close()
        
        # Reindex everything
        import asyncio
        asyncio.run(self.index_policies_directory(policies_dir))
