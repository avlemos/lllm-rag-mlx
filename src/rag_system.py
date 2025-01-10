from mlx_lm import load, generate
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import faiss
import os
from PyPDF2 import PdfReader
import glob

class RAGSystem:
    def __init__(self, model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"):
        # Load MLX model and tokenizer
        self.model, self.tokenizer = load(model_name)
        
        # Initialize sentence transformer for embeddings
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize FAISS index
        self.dimension = 384  # embedding dimension for MiniLM
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store documents and their embeddings
        self.documents: List[str] = []
        
    def process_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return ""
    
    def load_pdfs_from_folder(self, folder_path: str) -> List[str]:
        """
        Load all PDFs from a specified folder
        """
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {folder_path}")
            
        documents = []
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            text = self.process_pdf(pdf_file)
            if text:  # Only add non-empty documents
                documents.append(text)
                
        if not documents:
            raise ValueError("No valid text could be extracted from the PDF files")
                
        return documents

    def add_documents(self, documents: List[str], chunk_size: int = 512):
        """
        Add documents to the RAG system with chunking
        """
        if not documents:
            raise ValueError("No documents provided")
            
        # Simple chunking strategy
        chunks = []
        for doc in documents:
            if not doc.strip():  # Skip empty documents
                continue
                
            # Split document into chunks of roughly chunk_size characters
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
            raise ValueError("No valid chunks created from documents")
            
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create embeddings for chunks
        embeddings = self.encoder.encode(chunks, convert_to_numpy=True)
        
        # Verify embedding shape
        if len(embeddings.shape) != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(f"Invalid embedding shape. Expected (n, {self.dimension}), got {embeddings.shape}")
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(chunks)
        print(f"Added {len(chunks)} chunks to the index")
        
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve relevant documents for a query
        """
        if not self.documents:
            raise ValueError("No documents in the index")
            
        # Get query embedding
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(k, len(self.documents))
        )
        
        # Return relevant documents
        return [self.documents[i] for i in indices[0]]
    
    def generate_response(self, query: str, k: int = 3) -> str:
        """
        Generate a response using RAG
        """
        # Retrieve relevant contexts
        relevant_docs = self.retrieve(query, k)
        
        # Create prompt with context
        context = "\n".join(relevant_docs)
        prompt = f"""Context: {context}

Question: {query}

Based on the context provided, please answer the question:"""
        
        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
        # Generate response
        response = generate(self.model, self.tokenizer, prompt=prompt, verbose=True)
        return response

# Example usage
def main():
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        # Load PDFs from folder
        pdf_folder = "data"  # Update this path
        print(f"Loading PDFs from {pdf_folder}")
        documents = rag.load_pdfs_from_folder(pdf_folder)
        
        print(f"Successfully loaded {len(documents)} documents")
        
        # Add documents to RAG system
        rag.add_documents(documents)
        
        # Generate a response
        query = "Who is André Lemos?"
        print(f"Generating response for query: {query}")
        response = rag.generate_response(query)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()