# PDF-RAG-GEMMA
"An end-to-end Retrieval-Augmented Generation (RAG) system that processes PDFs, builds vector embeddings with SentenceTransformers, performs similarity search using FAISS, and generates accurate, context-aware answers using Gemma-2B-IT."

📄 PDF Question-Answering RAG System using Gemma 2B

This project implements a Retrieval-Augmented Generation (RAG) pipeline to query information from a PDF using:

Google Gemma-2B-IT (LLaMA-style instruction-tuned model)

SentenceTransformer (all-MiniLM-L6-v2) for embeddings

FAISS for vector similarity search

PyPDF2 for text extraction

Google Colab userdata API for securely loading Hugging Face tokens

The system supports:

PDF ingestion

Text chunking

Embedding generation

Vector indexing

Relevant chunk retrieval

Answer generation using the Gemma model

🚀 Features

✔ Extracts text from PDF files
✔ Splits content into manageable text chunks
✔ Generates embeddings using SentenceTransformer
✔ Stores embeddings in a FAISS similarity index
✔ Retrieves top-K relevant chunks for any query
✔ Generates LLM-based answers using context
✔ Fully runs on Google Colab GPU

🗂 Project Workflow
1. Install Dependencies

Installs transformers, sentence-transformers, FAISS CPU, PyPDF2, and NLTK.

2. Load Model

Gemma-2-2B-IT is loaded using your HuggingFace token stored in Colab's userdata.

3. PDF Processing

Reads the PDF

Extracts text

Splits text into chunked sentences

4. Embedding + Indexing

Embeddings are generated

Stored in FAISS index

5. Querying

Takes a natural language query

Retrieves top-K similar chunks

Generates a combined context

Sends context + question to Gemma for answer generation

📁 Folder / File Setup

Place your PDF in the project root:

/content/Enterprise RAG.pdf

🧠 How It Works
Embedding Step
document_embeddings = encoder.encode(chunks)

FAISS Index
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings)

Querying
similar_chunks = find_most_similar_chunks(query)
response = generate_response(query, context)

🔍 Example Query
query = "summarize this PDF?"
answer, relevant_chunks = query_documents(query)

📌 Key Functions
extract_text_from_pdf(path)

Reads and returns raw PDF text.

split_text_into_chunks(text)

Splits PDF text into smaller segments.

find_most_similar_chunks(query)

FAISS-based nearest-neighbor lookup.

generate_response(query, context)

Uses Gemma model to generate an answer.

query_documents(query)

Full RAG pipeline wrapper.

🧪 Technologies Used
Technology	Purpose
PyTorch	Model inference
Gemma 2B IT	LLM for final responses
SentenceTransformer	Embeddings
FAISS	Vector similarity search
PyPDF2	PDF text extraction
NLTK	Sentence tokenization
Colab userdata	Secure token storage
📌 Notes

GPU runtime is required to run Gemma efficiently.

Ensure Hugging Face token is stored using:
userdata.put("HF_TOKEN", "your_token_here")
