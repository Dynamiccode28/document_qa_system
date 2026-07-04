---
title: Document QA RAG System
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---


# ContextQuest: Local RAG Document QA System

## Overview

ContextQuest is a fully containerized, open-source Retrieval-Augmented Generation (RAG) application that enables users to upload a PDF document and interact with it through a conversational interface.

The system generates answers strictly from the uploaded document and provides source citations for every response, ensuring transparency and verifiability.

The entire pipeline operates using open-source models without relying on paid APIs such as OpenAI, allowing the application to run locally or in a free cloud environment while maintaining complete data privacy.

---

## Live Demo

Application:

https://huggingface.co/spaces/SVM28/document-qa-system

> **Note:** If the application is in sleep mode, click **Wake Up** and wait approximately 30 seconds for the container to start.

---

# Features

- Retrieval-Augmented Generation (RAG) pipeline built from scratch
- Upload and query PDF documents
- Source-grounded responses with citations
- Semantic document chunking
- FAISS-based vector similarity search
- Fully open-source technology stack
- No dependency on paid LLM APIs
- Dockerized deployment
- Local execution with Ollama support
- Cloud deployment on Hugging Face Spaces

---

# How to Use

## Step 1: Upload a Document

Click **Browse files** from the sidebar and upload a text-based PDF document (maximum size: 10 MB).

## Step 2: Process the Document

Click **Process Document**.

The application will:

- Extract text
- Clean formatting artifacts
- Create semantic chunks
- Generate embeddings
- Build a FAISS vector index

Processing typically takes **5–10 seconds**.

## Step 3: Ask Questions

Enter your question in the chat input and press **Enter**.

The system retrieves relevant document sections before generating an answer.

## Step 4: Verify the Response

Expand **View Source Chunks** beneath any response to inspect the exact passages retrieved from the uploaded document.

---

# System Architecture

The application follows a two-stage Retrieval-Augmented Generation pipeline.

## Phase 1: Document Ingestion

```text
PDF Document
      │
      ▼
Text Extraction (PyPDF2)
      │
      ▼
Text Preprocessing
(Removes PDF formatting artifacts)
      │
      ▼
Semantic Chunking
(~500-character chunks)
      │
      ▼
Sentence Embeddings
(all-MiniLM-L6-v2)
      │
      ▼
384-Dimensional Vector Embeddings
      │
      ▼
FAISS Vector Index
```

---

## Phase 2: Query Processing

```text
User Question
      │
      ▼
Question Embedding
      │
      ▼
FAISS Similarity Search
(Top-K Relevant Chunks)
      │
      ▼
Retrieved Context
+
User Question
      │
      ▼
Prompt Construction
      │
      ▼
Large Language Model
      │
      ▼
Grounded Response
+
Source Citations
```

---

# Technology Stack

| Component | Technology | Purpose |
|------------|------------|---------|
| Frontend | Streamlit | Interactive chat interface |
| Backend | FastAPI | REST API |
| PDF Processing | PyPDF2, Regex | Text extraction and preprocessing |
| Embedding Model | all-MiniLM-L6-v2 | Semantic text embeddings |
| Vector Database | FAISS (CPU) | Similarity search |
| Cloud LLM | Qwen2.5-0.5B-Instruct | Lightweight inference |
| Local LLM | Ollama (Llama 3.1 8B) | Local inference |
| Deployment | Docker, Hugging Face Spaces | Containerized deployment |

---

# Project Structure

```text
document_qa_system/
│
├── app.py                 # Streamlit frontend
├── api.py                 # FastAPI backend
├── requirements.txt
├── Dockerfile
├── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── prompt.py
│
├── models/
│
└── assets/
```

---

# Design Decisions

## 1. Semantic Chunking Strategy

The embedding model (`all-MiniLM-L6-v2`) accepts a maximum of 512 tokens.

Instead of splitting exactly at the token limit, the application creates chunks of approximately **500 characters**, ending at natural sentence boundaries whenever possible.

This approach:

- Preserves semantic meaning
- Avoids cutting sentences in half
- Maintains a comfortable safety margin below the token limit

---

## 2. PDF Text Normalization

PDF extraction frequently introduces unwanted newline characters.

Before chunking, the extracted text is normalized using regular expressions to restore proper sentence flow.

```python
re.sub(r'\n+', '\n', text)
```

This preprocessing significantly improves embedding quality and retrieval accuracy.

---

## 3. Grounded Prompt Engineering

The language model receives a strict instruction:

> Answer only using the provided context. If the answer cannot be found in the context, respond that there is insufficient information.

This minimizes hallucinations and ensures answers remain grounded in the uploaded document.

---

## 4. Cloud Deployment Strategy

The application is deployed on the free tier of Hugging Face Spaces, which provides limited memory resources.

Because larger models such as Llama 3.1 8B exceed the available memory, the cloud deployment uses **Qwen2.5-0.5B-Instruct**, which offers an effective balance between inference quality and resource efficiency.

For local execution, the application supports **Ollama** with **Llama 3.1 8B** for improved response quality.

---

# Limitations

### Scanned PDFs

The application performs text extraction rather than Optical Character Recognition (OCR).

Image-only or scanned PDFs cannot be processed unless OCR is added.

---

### File Size

Uploads are restricted to **10 MB** to prevent memory exhaustion on free cloud infrastructure.

---

### Lightweight Cloud Model

The cloud deployment uses a compact language model.

While efficient, it may not perform as well as larger models when answering highly complex or multi-step reasoning questions.

---

# Running Locally

## Clone the Repository

```bash
git clone https://github.com/Dynamiccode28/document_qa_system.git

cd document_qa_system
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Pull the Local LLM

Install Ollama and download the model.

```bash
ollama pull llama3.1:8b
```

---

## Start the Backend

```bash
python -m uvicorn api:app --reload
```

---

## Launch the Frontend

Open a new terminal.

```bash
streamlit run app.py
```

---

# Future Improvements

- OCR support for scanned documents
- Hybrid search (BM25 + Dense Retrieval)
- Metadata-aware retrieval
- Persistent vector database
- Multi-document retrieval
- Conversation memory
- Streaming responses
- Evaluation pipeline
- Reranking models
- GPU acceleration

---

# Learning Objectives

This project was developed from scratch to gain a practical understanding of:

- Retrieval-Augmented Generation (RAG)
- Semantic embeddings
- Vector databases
- FAISS similarity search
- Prompt engineering
- Large Language Models
- End-to-end AI system architecture
- Docker containerization
- Local and cloud deployment

No high-level orchestration frameworks (such as LangChain) were used in the core implementation, allowing every stage of the RAG pipeline to be understood and implemented directly.

---

