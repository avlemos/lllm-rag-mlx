from mlx_lm import load, generate
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import faiss
import os
from PyPDF2 import PdfReader
import glob
import sqlite3
import hashlib
from datetime import datetime
import json

class DocumentStore:
    def __init__(self, db_path: str = "rag_cache.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with necessary tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    file_hash TEXT,
                    last_modified TEXT,
                    processed_date TEXT,
                    chunks TEXT,
                    UNIQUE(file_path, file_hash)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    embedding BLOB,
                    chunk_text TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
            ''')
            
    def get_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def is_document_processed(self, file_path: str) -> bool:
        """Check if document is already processed and up to date"""
        try:
            # Normalize the path to handle spaces and special characters
            file_path = os.path.normpath(os.path.abspath(file_path))
            
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                return False
                
            file_hash = self.get_file_hash(file_path)
            last_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            
            print(f"\nChecking cache for: {file_path}")
            print(f"Normalized path: {file_path}")
            print(f"File hash: {file_hash}")
            print(f"Last modified: {last_modified}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    '''
                    SELECT id, file_hash, last_modified 
                    FROM documents 
                    WHERE file_path = ?
                    ''',
                    (file_path,)
                )
                result = cursor.fetchone()
                
                if result:
                    db_id, db_hash, db_modified = result
                    print(f"Found in database with path: {file_path}")
                    print(f"DB hash: {db_hash}")
                    print(f"DB last modified: {db_modified}")
                    
                    is_cached = (file_hash == db_hash)
                    print(f"Cache status: {'Hit' if is_cached else 'Miss (hash mismatch)'}")
                    return is_cached
                else:
                    print(f"Not found in database: {file_path}")
                    return False
                    
        except Exception as e:
            print(f"Error checking document cache: {str(e)}")
            return False
            
    def store_document(self, file_path: str, chunks: List[str]):
        """Store document information and chunks"""
        try:
            # Normalize the path
            file_path = os.path.normpath(os.path.abspath(file_path))
            
            file_hash = self.get_file_hash(file_path)
            last_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            
            print(f"\nStoring document: {file_path}")
            print(f"Normalized path: {file_path}")
            print(f"Hash: {file_hash}")
            print(f"Last modified: {last_modified}")
            print(f"Number of chunks: {len(chunks)}")
            
            with sqlite3.connect(self.db_path) as conn:
                # First, remove any existing entries for this file
                conn.execute('DELETE FROM embeddings WHERE document_id IN (SELECT id FROM documents WHERE file_path = ?)', (file_path,))
                conn.execute('DELETE FROM documents WHERE file_path = ?', (file_path,))
                
                # Then insert the new document
                cursor = conn.execute('''
                    INSERT INTO documents 
                    (file_path, file_hash, last_modified, processed_date, chunks)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    file_path,
                    file_hash,
                    last_modified,
                    datetime.now().isoformat(),
                    json.dumps(chunks)
                ))
                print(f"Document stored with ID: {cursor.lastrowid}")
                
        except Exception as e:
            print(f"Error storing document: {str(e)}")
            raise

            
    def store_embeddings(self, file_path: str, chunk_embeddings: List[Tuple[str, np.ndarray]]):
        """Store embeddings for document chunks"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT id FROM documents WHERE file_path = ?',
                (file_path,)
            )
            doc_id = cursor.fetchone()[0]
            
            for chunk_text, embedding in chunk_embeddings:
                conn.execute('''
                    INSERT INTO embeddings (document_id, embedding, chunk_text)
                    VALUES (?, ?, ?)
                ''', (
                    doc_id,
                    embedding.tobytes(),
                    chunk_text
                ))
                
    def get_all_embeddings(self) -> Tuple[List[np.ndarray], List[str]]:
        """Retrieve all stored embeddings and their corresponding chunks"""
        embeddings = []
        chunks = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT embedding, chunk_text FROM embeddings')
            for row in cursor:
                embedding_bytes, chunk_text = row
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                embeddings.append(embedding)
                chunks.append(chunk_text)
                
        return embeddings, chunks

class RAGSystem:
    def __init__(self, model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit", db_path: str = "rag_cache.db"):
        # Load MLX model and tokenizer
        print("Loading MLX model and tokenizer...")
        self.model, self.tokenizer = load(model_name)
        
        # Initialize sentence transformer for embeddings
        print("Initializing SentenceTransformer...")
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize FAISS index
        self.dimension = 384  # embedding dimension for MiniLM
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store documents and their embeddings
        self.documents: List[str] = []
        
        # Load existing embeddings
        # self._load_existing_embeddings()
    
    def _load_existing_document_cache(self, db_path: str = "rag_cache.db"):
        """Load existing document cache from storage"""
        # Initialize document store
        self.store = DocumentStore(db_path)
        
    def _load_existing_embeddings(self):
        """Load existing embeddings from storage into FAISS index"""
        embeddings, chunks = self.store.get_all_embeddings()
        if embeddings and chunks:
            embeddings_array = np.stack(embeddings)
            self.index.add(embeddings_array)
            self.documents.extend(chunks)
            print(f"Loaded {len(chunks)} existing chunks into the index")
    
    def process_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            print("process_pdf: ", pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return ""
    
    def load_pdfs_from_folder(self, folder_path: str) -> List[Tuple[str, str]]:
        """Load all PDFs from a specified folder, using cache when possible"""
        # Normalize the folder path
        folder_path = os.path.normpath(os.path.abspath(folder_path))
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {folder_path}")
            
        documents = []
        
        print(f"\nFound {len(pdf_files)} PDF files in {folder_path}")
        
        for pdf_file in pdf_files:
            # Normalize the file path to handle spaces and special characters
            normalized_path = os.path.normpath(os.path.abspath(pdf_file))
            print(f"\nProcessing file: {pdf_file}")
            print(f"Normalized path: {normalized_path}")
            
            if self.store.is_document_processed(normalized_path):
                print(f"Using cached version of {normalized_path}")
                continue
                
            print(f"Processing new file: {normalized_path}")
            text = self.process_pdf(normalized_path)
            if text:
                documents.append((normalized_path, text))
                
        return documents

    def get_document_count(self):
        """Retrieve the number of documents in the index"""
        return self.documents.count()


    def add_documents(self, documents: List[Tuple[str, str]], chunk_size: int = 512):
        """Add documents to the RAG system with chunking and storage"""
        if not documents:
            return
            
        for file_path, doc in documents:
            if not doc.strip():
                continue
                
            # Chunk the document
            chunks = []
            words = doc.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                current_chunk.append(word)
                current_length += len(word) + 1
                
                if current_length >= chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                    
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            if not chunks:
                continue
                
            print(f"Created {len(chunks)} chunks from {file_path}")
            
            # Generate embeddings
            embeddings = self.encoder.encode(chunks, convert_to_numpy=True)
            
            # Store document and embeddings
            self.store.store_document(file_path, chunks)
            self.store.store_embeddings(file_path, list(zip(chunks, embeddings)))
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            self.documents.extend(chunks)
            
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents for a query"""
        if not self.documents:
            raise ValueError("No documents in the index")
            
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(k, len(self.documents))
        )
        
        return [self.documents[i] for i in indices[0]]
    
    def generate_response(self, query: str, k: int = 3) -> str:
        """Generate a response using RAG"""
        relevant_docs = self.retrieve(query, k)
        
        context = "\n".join(relevant_docs)
        prompt = f"""Context: {context}

Question: {query}

Based on the context provided, please answer the question:"""
        
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
        return generate(self.model, self.tokenizer, prompt=prompt, verbose=True)

# Example usage
def main():
    try:
        # Initialize RAG system
        rag = RAGSystem(db_path="rag_cache.db")
        rag._load_existing_document_cache()
        rag._load_existing_embeddings
        
        # Load PDFs from folder
        pdf_folder = "../data/"  # Update this path
        print(f"Loading PDFs from {pdf_folder}")
        documents = rag.load_pdfs_from_folder(pdf_folder)
        
        if documents:
            print(f"Processing {len(documents)} new documents")
            rag.add_documents(documents)
        
        # # Generate a response
        # query = "What can you say about iText?"
        # print(f"Generating response for query: {query}")
        # response = rag.generate_response(query)
        # print(f"Response: {response}")
        
        print("\n\nWelcome to the interactive session. Type 'exit' to quit.")
        while True:
            # Prompt the user for a query
            query = input("Enter your query (or exit): ")
            
            # Check if the user wants to exit
            if query.lower() == 'exit':
                print("Exiting the session. Goodbye!")
                break
            
            # Generate a response
            print(f"Generating response for query: {query}")
            response = rag.generate_response(query)
            print("\n##################################################")
            print(f"Response: {response}")
            print("##################################################\n")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()