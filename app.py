import streamlit as st
import json
import datetime
import base64
import httpx
import openai
import fitz  # PyMuPDF
import tempfile
import os
from io import BytesIO
from openai import OpenAI
import instructor
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from pydantic import BaseModel, Field
from fuzzywuzzy import fuzz
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def text_similarity(a: str, b: str) -> float:
    """Calculate text similarity between two strings"""
    return fuzz.partial_ratio(a, b) / 100.0  # Convert to 0-1 range

# Page configuration
st.set_page_config(layout="wide")


# Initialize LLM client with timeout and retry settings
client = None
api_available = False

# Define default LLM settings from environment variables
ds_base_url = os.getenv("DS_BASE_URL", "http://mtm-llm-uk.uksouth.cloudapp.azure.com:11435/v1")
ds_llm_model = os.getenv("DS_LLM_MODEL", "qwen2.5:72b-instruct-q8_0")
ds_api = os.getenv("DS_API", "ollama")

deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
deepseek_llm_model = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-chat")
deepseek_api = os.getenv("DEEPSEEK_API", "sk-xxxxxxxxxxx")

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_llm_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
ollama_api = os.getenv("OLLAMA_API", "Ollama")

# Initialize variables that will be set in the sidebar
base_url = None
llm_provider = None
api_key = None
llm_model = None



# Initialize embeddings with error handling
try:
    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )
    # Test embeddings with a simple query
    test_embedding = embeddings.embed_query("test")
    if not test_embedding or not isinstance(test_embedding, list):
        raise ValueError("Embeddings returned invalid format")
except Exception as e:
    st.error(f"Failed to initialize embeddings: {str(e)}")
    st.write("Error details:")
    import traceback
    st.code(traceback.format_exc())
    st.warning("Document search functionality will be limited")
    embeddings = None

# Initialize LLM client after sidebar selection
if 'llm_provider' in st.session_state and 'base_url' in st.session_state and 'llm_model' in st.session_state and 'api_key' in st.session_state:
    llm_provider = st.session_state.llm_provider
    base_url = st.session_state.base_url
    llm_model = st.session_state.llm_model
    api_key = st.session_state.api_key

    try:
        # Test API connection first
        test_client = OpenAI(
            base_url=f"{base_url}/v1" if llm_provider in ["Deepseek", "Local (Ollama)"] else base_url,
            api_key=api_key,
            timeout=10.0,  # Shorter timeout for connection test
        )

        # Simple ping to check if API is reachable
        try:
            test_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            api_available = True
        except Exception as e:
            st.warning(f"LLM API is not responding: {str(e)}")
            api_available = False

        # Initialize the actual client if API is available
        if api_available:
            # Initialize client with provider-specific settings
            if llm_provider == "Data Science Machine":
                client = instructor.patch(
                    OpenAI(
                        base_url=ds_base_url,  # Use the full URL with /v1
                        api_key=ds_api,
                        timeout=30.0,
                        max_retries=3,
                    ),
                    mode=instructor.Mode.JSON,
                )
            else:
                client = instructor.patch(
                    OpenAI(
                        base_url=f"{base_url}/v1" if llm_provider != "OpenRouter" else base_url,
                        api_key=api_key,
                        timeout=30.0,  # 30 second timeout
                        max_retries=3,  # Retry up to 3 times
                    ),
                    mode=instructor.Mode.JSON,
                )
            # Set temperature=0 for deterministic responses
            client.temperature = 0
            st.write("LLM client initialized successfully")
        else:
            st.info("Running in local mode - using document search only")

    except Exception as e:
        st.error(f"Failed to initialize LLM client: {str(e)}")
        st.write(f"Error type: {type(e)}")
        st.write("Traceback:")
        import traceback
        st.code(traceback.format_exc())
        st.info("Running in local mode - using document search only")
elif api_available:
    st.sidebar.success("LLM API is available and responsive")
else:
    st.info("LLM provider not configured - running in local mode")

# Initialize vector store
vector_store = None

class DocumentResponse(BaseModel):
    """
    A response model for document queries.

    Attributes:
        answer (str): The answer to the user's question based on the context.
        evidence (List[str]): Passages extracted as is and used as evidence to support the answer.
    """
    answer: str = Field(default="No answer found", description="The answer to the user's question based on the context")
    evidence: List[str] = Field(default_factory=list, description="passages extracted AS IS and used as evidence to support the answer")

def extract_paragraphs_with_metadata(pdf_bytes: bytes, file_name: str) -> List[Document]:
    """Extract paragraphs with metadata using PyMuPDF"""
    import fitz  # PyMuPDF

    # Open PDF from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    documents = []

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        blocks = page.get_text("blocks")

        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            if block_type == 0:  # Text block
                # Create document with metadata
                doc = Document(
                    page_content=text.strip(),
                    metadata={
                        "page_number": page_num + 1,
                        "file_name": file_name,
                        "bbox": [x0, y0, x1, y1]
                    }
                )
                documents.append(doc)

    return documents

def process_pdf(pdf_bytes: bytes) -> Optional[dict]:
    """Process PDF and create hybrid retriever"""
    if not embeddings:
        st.warning("Embeddings not available - cannot create vector store")
        return None

    try:
        # Extract paragraphs with metadata
        documents = extract_paragraphs_with_metadata(pdf_bytes, uploaded_file.name)

        if not documents:
            st.warning("No text content found in PDF")
            return None

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store
        faiss_store = FAISS.from_documents(splits, embeddings)
        faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 10})

        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 10

        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

        return {
            "faiss_store": faiss_store,
            "ensemble_retriever": ensemble_retriever
        }
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        st.write("Error details:")
        import traceback
        st.code(traceback.format_exc())
        return None

def rank_chunks(evidence_text: str, chunks: List[Document]) -> List[Document]:
    """Rank chunks based on partial ratio similarity"""
    # Normalize evidence text
    normalized_evidence = normalize_text(evidence_text)

    # Score each chunk
    scored_chunks = []
    for chunk in chunks:
        # Normalize chunk text
        normalized_chunk = normalize_text(chunk.page_content)

        # Calculate partial ratio score
        score = fuzz.partial_ratio(normalized_evidence, normalized_chunk)

        scored_chunks.append({
            'chunk': chunk,
            'score': score
        })

    # Sort by score descending
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)

    return scored_chunks

def normalize_text(text: str) -> str:
    """Normalize text for better matching by:
    1. Replacing special dashes with regular hyphens
    2. Removing ligatures
    3. Normalizing whitespace and line breaks
    4. Handling hyphenated words across line breaks
    5. Removing all non-alphanumeric characters except basic punctuation
    """
    import re

    # Replace different dash types with regular hyphen
    text = text.replace('—', '-').replace('–', '-')

    # Replace common ligatures
    ligatures = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl'
    }
    for lig, replacement in ligatures.items():
        text = text.replace(lig, replacement)

    # Handle line breaks and hyphenated words more aggressively
    text = re.sub(r'-\s*\n\s*', '', text)  # Remove hyphen+newline combinations with any whitespace
    text = re.sub(r'\s*\n\s*', ' ', text)  # Replace remaining newlines with single space

    # Normalize multiple spaces and strip leading/trailing spaces
    text = ' '.join(text.split())

    # Remove all non-alphanumeric characters except basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,;:!?()\[\]{}'\"-]", '', text)

    # Normalize quotes
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Convert to lowercase for case-insensitive matching
    text = text.lower()

    return text

def query_document(query: str, k: int = 10) -> dict:
    """Query the document using hybrid RAG"""
    if vector_store is None or "ensemble_retriever" not in vector_store:
        st.warning("Vector store not initialized")
        return {"response": [], "docs": []}

    if not query or not isinstance(query, str):
        st.warning("Invalid query format")
        return {"response": [], "docs": []}

    try:
        # Retrieve relevant documents using hybrid approach
        docs = vector_store["ensemble_retriever"].get_relevant_documents(query)

        # Combine the relevant chunks into context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate a synthesized response using the LLM
        if client and api_available:
            try:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": """You are a helpful assistant that answers questions based on the provided context.
                        For each response, provide:
                        1. A clear answer to the question
                        2. The exact passages from the context that support your answer
                        Format your response as JSON with 'answer' and 'evidence' fields."""},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ],
                    temperature=0,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )

                if response and response.choices:
                    try:
                        llm_response = json.loads(response.choices[0].message.content)
                        answer = llm_response.get('answer', 'No answer found')
                        # Ensure answer is always a string
                        if isinstance(answer, list):
                            answer = " ".join(str(item) for item in answer)
                        elif not isinstance(answer, str):
                            answer = str(answer)

                        evidence = llm_response.get('evidence', [])
                        # Ensure evidence is always a list
                        if isinstance(evidence, str):
                            evidence = [evidence]
                        elif not isinstance(evidence, list):
                            evidence = []
                    except json.JSONDecodeError:
                        answer = response.choices[0].message.content
                        evidence = []
                else:
                    # Fallback response if no valid response from LLM
                    answer = "No response from LLM"
                    evidence = ["No supporting evidence provided"]
            except Exception as e:
                st.error(f"Error getting LLM response: {str(e)}")
                answer = "Error getting response from LLM"
                evidence = ["No supporting evidence provided"]
        else:
            # Fallback to just returning the first chunk if LLM is not available
            answer = docs[0][0].page_content if docs else "No relevant information found"

        # Initialize evidence if not already set
        if 'evidence' not in locals():
            evidence = ["No supporting evidence provided"]

        # Format response with the required fields
        return {
            "response": [
                {
                    "response": DocumentResponse(
                        answer=answer,
                        evidence=evidence
                    ),
                    "metadata": docs[0].metadata if docs else {},
                    "context": [doc.page_content for doc in docs]  # Add context for reference
                }
            ],
            "docs": docs
        }
    except Exception as e:
        st.error(f"Error querying document: {str(e)}")
        st.write("Error details:")
        import traceback
        st.code(traceback.format_exc())
        return []

# Sidebar for settings
with st.sidebar:
    # LLM Provider Selector
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["Local (Ollama)", "Data Science Machine", "Deepseek"],
        index=0,
        help="Select which LLM provider to use"
    )

    # Initialize based on selection and store in session state
    if llm_provider == "Local (Ollama)":
        st.session_state.base_url = ollama_base_url
        st.session_state.llm_model = ollama_llm_model
        st.session_state.api_key = ollama_api
    elif llm_provider == "Data Science Machine":  # Fix the exact match
        st.session_state.base_url = ds_base_url
        st.session_state.llm_model = ds_llm_model
        st.session_state.api_key = ds_api
    elif llm_provider == "Deepseek":
        st.session_state.base_url = deepseek_base_url
        st.session_state.llm_model = deepseek_llm_model
        st.session_state.api_key = deepseek_api

    # Set the provider in session state
    st.session_state.llm_provider = llm_provider

    # PDF Upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# PDF Viewer
if uploaded_file is not None:
    # Initialize search text
    search_text = ""

    # Read file
    pdf_bytes = uploaded_file.read()
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

    # Process PDF and create vector store
    if "vector_store" not in st.session_state:
        with st.spinner("Processing document..."):
            vector_store = process_pdf(pdf_bytes)
            st.session_state.vector_store = vector_store
    else:
        vector_store = st.session_state.vector_store

    # Display PDF with search functionality
    if search_text:
        # Create a temporary file for the highlighted PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            # Write original PDF to temp file
            tmp_file.write(pdf_bytes)
            tmp_file_path = tmp_file.name

            # Highlight text in PDF
            pdf_document = fitz.open(tmp_file_path)
            found_pages = set()

            # Normalize search text for better matching
            normalized_search = normalize_text(search_text)

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Search for normalized text with different variations
                search_variations = [
                    normalized_search,
                    normalized_search.replace(' ', ''),     # Try without spaces
                    normalized_search.replace('-', ' '),    # Try with hyphens as spaces
                    ' '.join(normalized_search.split())     # Try with single spaces
                ]

                # Add fuzzy matching with slight variations
                if len(normalized_search) > 10:
                    # Try matching 90% of the text
                    partial_match = normalized_search[:int(len(normalized_search)*0.9)]
                    search_variations.append(partial_match)

                    # Try matching with one word removed
                    words = normalized_search.split()
                    if len(words) > 2:
                        search_variations.append(' '.join(words[:-1]))  # Remove last word
                        search_variations.append(' '.join(words[1:]))   # Remove first word

                text_instances = []
                for variation in search_variations:
                    instances = page.search_for(variation)
                    if instances:
                        text_instances.extend(instances)

                if text_instances:
                    found_pages.add(page_num + 1)
                    for inst in text_instances:
                        try:
                            # Create highlight annotation with custom color
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
                            highlight.update()
                        except Exception as e:
                            st.warning(f"Could not create highlight: {str(e)}")
                            # Fallback to simple rectangle annotation
                            rect = page.add_rect_annot(inst)
                            rect.set_colors(stroke=(1, 1, 0))
                            rect.update()

            # Save highlighted PDF
            highlighted_pdf_path = tmp_file_path + "_highlighted.pdf"
            pdf_document.save(highlighted_pdf_path)
            pdf_document.close()

            # Read highlighted PDF
            with open(highlighted_pdf_path, "rb") as f:
                highlighted_pdf_bytes = f.read()
            base64_pdf = base64.b64encode(highlighted_pdf_bytes).decode('utf-8')

            # Update session state with highlighted PDF
            st.session_state['highlighted_pdf'] = base64_pdf
            # No timestamp needed

            # Clean up temporary files
            os.unlink(tmp_file_path)
            os.unlink(highlighted_pdf_path)

        if found_pages:
            st.success(f"Text found on pages: {', '.join(map(str, sorted(found_pages)))}")
            # Add page navigation buttons
            cols = st.columns(len(found_pages))
            for idx, page_num in enumerate(sorted(found_pages)):
                with cols[idx]:
                    if st.button(f"Go to page {page_num}"):
                        st.session_state['page_num'] = page_num
                        st.rerun()
        else:
            st.warning("Text not found in the document")

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Chat with Document", "PDF Viewer", "LLM Response History"])

    with tab1:
        # Create a container for chat history
        chat_container = st.container()

        # Display chat messages in the container
        with chat_container:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages with a fixed height container
            chat_history = st.container(height=500, border=False)  # Reduced from 600 to 500 (halfway reduction)
            with chat_history:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

        # Create a fixed position container for the chat input
        chat_input_container = st.container()
        with chat_input_container:
            # Chat input with a fixed position
            if prompt := st.chat_input("Ask a question about the document", key="chat_input"):
                # Only process if prompt is not empty
                if prompt.strip():
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # Display user message in the chat history
                    with chat_history:
                        with st.chat_message("user"):
                            st.markdown(prompt)

                    # Initialize response with a default value
                    response = DocumentResponse(answer="No response available", evidence=[])

                    # Get response from document
                    if vector_store:
                        responses = query_document(prompt)
                        if responses:
                            # Get the response content
                            if responses and responses['response']:
                                response = responses['response'][0]['response']

                                # Format the message with response and evidence
                                message_content = f"{response.answer}\n\n**Evidence:**\n"
                                message_content += "\n".join(f"- {evidence}" for evidence in response.evidence)
                            else:
                                message_content = "No response available"
                                response = DocumentResponse(answer=message_content, evidence=[])

                            # Display assistant message in the chat history
                            with chat_history:
                                with st.chat_message("assistant"):
                                    st.markdown(message_content)

                            # Store the raw JSON response
                            if 'llm_responses' not in st.session_state:
                                st.session_state.llm_responses = []

                            # Format context as JSON object with chunk numbers as keys
                            context_json = {}
                            if responses and responses['response']:
                                context_json = {
                                    f"chunk_{i+1}": {
                                        "content": chunk,
                                        "metadata": responses['docs'][i].metadata if i < len(responses['docs']) else {}
                                    }
                                    for i, chunk in enumerate(responses['response'][0]['context'])
                                }

                            st.session_state.llm_responses.append({
                                "timestamp": datetime.datetime.now().isoformat(),
                                "query": prompt,
                                "response": response.dict(),
                                "context": context_json
                            })

                            # Add assistant message to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": message_content
                            })

                    # Highlight evidence in PDF
                    if responses and responses['response'] and responses['response'][0]['response'].evidence:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            # Write original PDF to temp file
                            tmp_file.write(pdf_bytes)
                            tmp_file_path = tmp_file.name

                            # Highlight evidence in PDF
                            pdf_document = fitz.open(tmp_file_path)
                            found_pages = set()
                            evidence_without_matches = []

                            # First pass: Try direct highlighting for all evidence
                            for evidence_text in responses['response'][0]['response'].evidence:
                                found_match = False
                                normalized_evidence = normalize_text(evidence_text)

                                for page_num in range(len(pdf_document)):
                                    page = pdf_document[page_num]
                                    text_instances = page.search_for(normalized_evidence)

                                    if text_instances:
                                        found_match = True
                                        found_pages.add(page_num + 1)
                                        for inst in text_instances:
                                            try:
                                                # Create highlight annotation with custom color
                                                highlight = page.add_highlight_annot(inst)
                                                highlight.set_colors(stroke=(0, 1, 0))  # Green highlight for direct matches
                                                highlight.update()
                                            except Exception as e:
                                                st.warning(f"Could not create highlight: {str(e)}")
                                                # Fallback to simple rectangle annotation
                                                rect = page.add_rect_annot(inst)
                                                rect.set_colors(stroke=(0, 1, 0))
                                                rect.update()

                                # Track evidence that didn't get any direct matches
                                if not found_match:
                                    evidence_without_matches.append(evidence_text)

                            # Second pass: Fallback highlighting only for evidence without direct matches
                            if evidence_without_matches:
                                for evidence_text in evidence_without_matches:
                                    # Rank all chunks for this evidence
                                    ranked_chunks = rank_chunks(evidence_text, responses['docs'])

                                    # Take the top ranked chunk if above confidence threshold
                                    if ranked_chunks and ranked_chunks[0]['score'] >= 90:
                                        best_match = ranked_chunks[0]['chunk']
                                        page_num = best_match.metadata['page_number'] - 1
                                        bbox = best_match.metadata['bbox']

                                        # Get the page
                                        page = pdf_document[page_num]

                                        # Add padding to the bbox coordinates
                                        horizontal_padding = 5
                                        vertical_padding = 2
                                        padded_bbox = [
                                            bbox[0] - horizontal_padding,
                                            bbox[1] - vertical_padding,
                                            bbox[2] + horizontal_padding,
                                            bbox[3] + vertical_padding
                                        ]

                                        # Create rectangle with padding
                                        rect = page.add_rect_annot(fitz.Rect(*padded_bbox))
                                        rect.set_colors(stroke=(1, 0, 0))  # Red color for fallback highlights
                                        rect.set_border(width=2)
                                        rect.update()

                                        st.info(f"Used fallback highlighting for evidence on page {page_num + 1}")

                            # Save highlighted PDF
                            highlighted_pdf_path = tmp_file_path + "_highlighted.pdf"
                            pdf_document.save(highlighted_pdf_path)
                            pdf_document.close()

                            # Read highlighted PDF
                            with open(highlighted_pdf_path, "rb") as f:
                                highlighted_pdf_bytes = f.read()
                            base64_pdf = base64.b64encode(highlighted_pdf_bytes).decode('utf-8')

                            # Update session state with highlighted PDF
                            st.session_state['highlighted_pdf'] = base64_pdf

                            # Clean up temporary files
                            os.unlink(tmp_file_path)
                            os.unlink(highlighted_pdf_path)

                        if found_pages:
                            st.success(f"Evidence found on pages: {', '.join(map(str, sorted(found_pages)))}")
                            # Update page number to first evidence page
                            st.session_state['page_num'] = sorted(found_pages)[0]
                            st.rerun()

    with tab2:
        st.markdown("""
            <style>
                .pdf-container {
                    height: 800px;
                    margin-bottom: 20px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Display PDF with page parameter
        page_num = st.session_state.get('page_num', 1)
        display_pdf = st.session_state.get('highlighted_pdf', base64_pdf)
        pdf_display = f'<div class="pdf-container"><iframe src="data:application/pdf;base64,{display_pdf}#page={page_num}" width="100%" height="100%" type="application/pdf"></iframe></div>'
        st.markdown(pdf_display, unsafe_allow_html=True)

        # Search text input
        search_text = st.text_input("Search text in PDF")

    with tab3:
        if 'llm_responses' in st.session_state and st.session_state.llm_responses:
            for response in st.session_state.llm_responses:
                with st.expander(f"Response at {response['timestamp']}"):
                    st.write("**Query:**")
                    st.code(response['query'], language='text')

                    st.write("**Response:**")
                    st.json(response['response'])

                    st.write("**Context:**")
                    st.json(response['context'])
        else:
            st.info("No LLM responses yet")

        # Add page navigation controls
        if "highlighted_pdf" in st.session_state:
            st.write("Navigate to page:")
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    if st.button(f"Page {i+1}"):
                        st.session_state['page_num'] = i+1
                        st.experimental_rerun()
else:
    st.info("Please upload a PDF file to view its content.")
