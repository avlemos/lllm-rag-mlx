Python 3.12.8

Document Storage and Embedding:

Uses sentence-transformers for creating document embeddings
Implements document chunking for handling longer texts
Uses FAISS for efficient similarity search


Retrieval:

Converts user queries into embeddings
Finds most relevant document chunks using FAISS
Returns top-k most similar documents


Generation:

Combines retrieved context with the user query
Uses your existing MLX model for generation
Maintains compatibility with the chat template system
