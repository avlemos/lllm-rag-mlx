# MLX RAG System with PDF Processing

This project implements a Retrieval-Augmented Generation (RAG) system using MLX, designed to process PDF documents and answer questions based on their content. The system combines the power of the Llama model for generation with efficient document retrieval using FAISS.

## Features

- PDF document processing and text extraction
- Automatic document chunking for optimal processing
- Efficient similarity search using FAISS
- Integration with MLX and Llama model for text generation
- Document embedding using Sentence Transformers
- Batch processing of multiple PDF files
- Error handling and validation

## Prerequisites

- Python 3.8+
- MLX (Apple Silicon optimized)
- Sufficient storage for model weights and document embeddings

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   └── rag_system.py
└── data/               # Place your PDF files here
```

## Usage

1. Place your PDF documents in the `data/` directory.

2. Run the system:
```python
from rag_system import RAGSystem

# Initialize the system
rag = RAGSystem()

# Load PDFs from folder
pdf_folder = "data/"
documents = rag.load_pdfs_from_folder(pdf_folder)

# Add documents to the system
rag.add_documents(documents)

# Generate a response
query = "What are the main topics covered in these documents?"
response = rag.generate_response(query)
print(response)
```

The file rag_system.py also contains an example usage (as the following above), with an interactive session instead

## Configuration Options

You can customize the system behavior through several parameters:

- `chunk_size`: Control the size of document chunks (default: 512 characters)
- `k`: Number of relevant documents to retrieve (default: 3)
- `model_name`: Choose different MLX models

Example:
```python
# Initialize with custom parameters
rag = RAGSystem(model_name="mlx-community/Llama-3.2-3B-Instruct-4bit")

# Add documents with custom chunk size
rag.add_documents(documents, chunk_size=1024)

# Generate response with custom retrieval count
response = rag.generate_response(query, k=5)
```

## Error Handling

The system includes comprehensive error handling for common issues:
- Invalid PDF files
- Empty documents
- Invalid embedding shapes
- Missing PDF folder
- Insufficient documents for indexing

Error messages will guide you to the source of any problems.

## Performance Considerations

- **Memory Usage**: The system loads document embeddings into memory. For large document collections, ensure sufficient RAM is available.
- **Processing Time**: Initial document processing and embedding generation may take time depending on the collection size.
- **Model Size**: The MLX model requires significant storage and memory. Ensure your system meets the requirements.

## Limitations

- Only processes text-based PDFs (scanned documents may not work)
- Limited by available system memory for document embeddings
- Requires Apple Silicon for MLX optimization
- Maximum context length depends on the chosen model

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Acknowledgments

- MLX team at Apple
- Sentence Transformers
- FAISS team at Facebook Research
- PyPDF2 contributors
