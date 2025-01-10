from mlx_lm import load, generate
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import faiss
import os

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
        
    def add_documents(self, documents: List[str], chunk_size: int = 512):
        """
        Add documents to the RAG system with chunking
        """
        # Simple chunking strategy
        chunks = []
        for doc in documents:
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
        
        # Create embeddings for chunks
        embeddings = self.encoder.encode(chunks)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(chunks)
        
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve relevant documents for a query
        """
        # Get query embedding
        query_embedding = self.encoder.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
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
    # Initialize RAG system
    rag = RAGSystem()
    
    # Example documents
    documents = [
        "MLX is Apple's machine learning framework designed for efficient training and deployment.",
        "RAG systems combine retrieval and generation for more accurate responses.",
        "The Llama model family was developed by Meta AI and includes various sizes."
    ]
    
    # Add documents to RAG system
    rag.add_documents(documents)
    
    # Generate a response
    query = "What is MLX and how does it relate to machine learning?"
    response = rag.generate_response(query)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
