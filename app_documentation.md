# PDF Viewer with Chat Application Documentation

## Application Overview
This Streamlit application provides an interactive interface for viewing and querying PDF documents using AI. It combines PDF viewing capabilities with a chat interface that can answer questions about the document content using a hybrid retrieval approach.

## Information Flow
1. PDF Upload -> 2. Document Processing -> 3. Vector Store Creation -> 4. User Query -> 5. Hybrid Retrieval -> 6. LLM Response Generation -> 7. Response Display

## Key Functions

### 1. `process_pdf(pdf_bytes: bytes) -> Optional[dict]`
**Purpose:** Processes uploaded PDF and creates hybrid retriever
**Input:** PDF file bytes
**Output:** Dictionary containing FAISS vector store and ensemble retriever
**Flow Position:** Step 2 in information flow
**Details:**
- Extracts text from PDF pages
- Creates document objects with page metadata
- Splits documents into chunks
- Creates FAISS vector store and BM25 retriever
- Combines retrievers into ensemble retriever

### 2. `normalize_text(text: str) -> str`
**Purpose:** Normalizes text for better matching in search and highlighting
**Input:** Raw text string
**Output:** Normalized text string
**Flow Position:** Used in text search and evidence highlighting
**Details:**
- Handles special characters and ligatures
- Normalizes spaces and hyphens
- Processes line breaks and hyphenated words
- Removes extra spaces and special dashes
- Creates multiple search variations for better matching

### 2. `query_document(query: str, k: int = 10) -> List[DocumentResponse]`
**Purpose:** Handles document queries using hybrid RAG
**Input:** User query string, number of results (k)
**Output:** List of DocumentResponse objects
**Flow Position:** Steps 4-6 in information flow
**Details:**
- Uses ensemble retriever to find relevant documents
- Combines relevant chunks into context
- Generates synthesized response using LLM
- Formats response with answer and evidence
- Handles fallback when LLM is unavailable

### 3. DocumentResponse Class
**Purpose:** Response model for document queries
**Attributes:**
- `answer`: The synthesized answer to the query
- `evidence`: Supporting passages from the document
**Flow Position:** Part of Step 6 in information flow
**Details:**
- Standardizes response format
- Ensures consistent data structure for frontend display

## Information Transformation
1. **PDF to Text:** Raw PDF bytes are converted to text with page metadata
2. **Text to Chunks:** Text is split into manageable chunks for processing
3. **Chunks to Vectors:** Text chunks are converted to vector embeddings
4. **Query to Context:** User query is matched with relevant document chunks
5. **Context to Answer:** Relevant context is synthesized into a coherent answer
6. **Answer to Display:** Response is formatted for user-friendly presentation

## Error Handling
- Embeddings initialization failure
- PDF processing errors
- LLM API connectivity issues
- Invalid query formats
- Document search limitations

## Session State Management
The application uses Streamlit's session state to maintain:
- Vector store and retriever
- Chat message history
- LLM responses
- Highlighted PDF state
- Current page number

## Integration Points
- LLM providers (Ollama, Deepseek, OpenRouter)
- Vector store (FAISS)
- Text retrieval (BM25)
- PDF processing (PyPDF2, PyMuPDF)
- Streamlit UI components
