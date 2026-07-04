---
title: Document QA RAG System
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---


title: RAG Document QA Systememoji: 📄colorFrom: bluecolorTo: indigosdk: dockerapp_port: 8501pinned: false
📄 ContextQuest: Local RAG Document QA System
A fully containerized, open-source Retrieval-Augmented Generation (RAG) application. Upload a PDF document and chat with it. The AI answers strictly based on the content of your document, complete with verifiable source citations.

No paid APIs (like OpenAI) are used. The entire pipeline runs locally or in a free cloud container, ensuring complete data privacy.

🌐 Live Application
Access the app directly in your browser here:
👉 https://dynamiccode28-document-qa-system.hf.space

(Note: If the app is asleep, click "Wake Up" and wait ~30 seconds for the server to spin up).

🚀 How to Use (End-User Guide)
Interacting with the system is as simple as using a chat app:

Upload: In the left sidebar, click "Browse files" and select any text-based PDF (max 10MB).
Process: Click the "Process Document" button. The system will extract the text, break it down, and index it (takes ~5-10 seconds).
Chat: Type your question in the bottom text box and press Enter.
Verify: Click "📄 View Source Chunks" under any answer to see the exact paragraphs from your PDF that the AI used to formulate its response.
🧠 How It Works (System Architecture)
This project implements a custom-built RAG pipeline. It is separated into two distinct phases: Ingestion and Querying.

Phase 1: Document Ingestion (Preparation)
PDF File → [Text Extraction] → Raw Text
↓
Raw Text → [Text Preprocessing] → Cleaned Text (Fixes PDF line-break bugs)
↓
Cleaned Text → [Semantic Chunking] → List of ~500 character chunks
↓
Chunks → [Embedding Model] → 384-Dimensional Vectors
↓
Vectors → [FAISS Index] → In-Memory Vector Database

text


### Phase 2: Querying (Question Answering)
User Question → [Embedding Model] → Question Vector
↓
Question Vector → [FAISS Similarity Search] → Top 3 Relevant Chunks
↓
Chunks + Question → [Prompt Engineering] -> Structured LLM Prompt
↓
Prompt → [Local LLM] -> Final Generated Answer + Citations

text


## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Chat-based user interface |
| **Text Processing** | PyPDF2, Regex | PDF parsing & whitespace normalization |
| **Embeddings** | `all-MiniLM-L6-v2` | Converts text to 384-dimension vectors |
| **Vector DB** | FAISS (CPU) | Blazing-fast similarity search |
| **LLM (Cloud)** | Qwen2.5-0.5B-Instruct | Lightweight model fitting in free-tier RAM |
| **LLM (Local)** | Ollama (Llama 3) | For local, high-power testing |
| **Deployment** | Docker, Hugging Face Spaces | Containerized cloud hosting |

---

## 💡 Important Architectural Insights & Design Decisions

As an educational project built from scratch (without LangChain), several deliberate engineering decisions were made:

*   **Why 500-Character Chunks?** 
    The embedding model (`all-MiniLM-L6-v2`) has a strict limit of 512 tokens. Instead of cutting at exactly 512 tokens (which might slice a word in half), we cut at 500 *characters* (~125 tokens) at natural sentence boundaries (`.` `!` `?`). This leaves a massive safety buffer and ensures semantic completeness.
*   **The PDF Newline Bug:** 
    PyPDF2 often injects `\n` at the end of every line in a PDF. If fed directly to an embedding model, it destroys the horizontal semantic flow. A crucial preprocessing step (`re.sub(r'\n+', '\n', text)`) was added to flatten the text before chunking.
*   **Strict Prompting:** 
    The LLM is given a strict system prompt: *"Answer ONLY using the context. If the answer is not present, say 'I do not have enough information.'"* This prevents hallucinations and ensures strict RAG behavior.
*   **Cloud vs. Local Architecture Pivot:** 
    To keep the project 100% free, the app is deployed on Hugging Face Spaces (limited to 1GB RAM). Because Llama-3-8B requires ~5GB RAM, the cloud version uses a highly efficient `Qwen-0.5B` model inside the Docker container. The local version uses Ollama for maximum quality.

## ⚠️ Limitations & Constraints

- **Scanned PDFs:** The system uses text extraction, not OCR. If the PDF is a flat image of text (like a scanned book), PyPDF2 will return empty strings. 
- **Document Size:** To prevent Out-Of-Memory (OOM) errors on free cloud tiers, file uploads are restricted to 10MB.
- **Small LLM Context:** The cloud LLM (0.5B parameters) is incredibly fast and lightweight, but may struggle with highly complex, multi-hop reasoning across 5+ chunks compared to massive models like GPT-4.

## 🏃‍♂️ Running Locally (For Developers)

If you want to run the high-power version using Ollama and Llama 3:

```bash
# 1. Clone the repository
git clone https://github.com/Dynamiccode28/document_qa_system.git
cd document_qa_system

# 2. Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama and pull the model (requires Ollama installed on your system)
ollama pull llama3.1:8b

# 5. Run the backend API
python -m uvicorn api:app --reload

# 6. In a new terminal, run the UI
streamlit run app.py
📜 License
This project is open source and available under the MIT License.

Built from scratch to understand the mathematics of embeddings, the mechanics of vector search, and the architecture of Large Language Models.

text


### Why this README is professional:
1. **The YAML Block:** It's at the very top, which is required by Hugging Face to know it's a Docker app.
2. **Immediate Value:** The "Live Application" link is right at the top. Recruiters don't want to read a manual to see your work; they want to click a link.
3. **End-User vs. Developer Separation:** It tells a normal person how to use it in 3 steps, but has a whole "Running Locally" section for technical interviewers.
4. **The "Architectural Insights" Section:** This is the secret sauce. It proves you didn't just copy a tutorial—you understand *why* chunk sizes matter, *why* you preprocess text, and *how* memory limits affect cloud architecture. 

**Push this to GitHub one last time, and your project is perfectly wrapped up!**
