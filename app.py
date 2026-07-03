import streamlit as st
import numpy as np
import faiss
import requests
import os
import sys

# --- IMPORT RAG COMPONENTS DIRECTLY ---
from src.pdf_extractor import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeder import model
from src.prompt_builder import build_rag_prompt

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Document QA", page_icon="📄")

# --- SESSION STATE (Replaces FastAPI Global Variables) ---
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BACKEND LOGIC (Runs directly in Streamlit) ---

def process_pdf_locally(file):
    """Does what our FastAPI /upload endpoint used to do."""
    save_path = os.path.join("data", file.name)
    with open(save_path, "wb") as f:
        f.write(file.getvalue())
        
    raw_text = extract_text_from_pdf(save_path)
    chunks = chunk_text(raw_text)
    
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save to session state so it survives reruns
    st.session_state.vector_index = index
    st.session_state.text_chunks = chunks
    
    return f"Processed {len(chunks)} chunks successfully."

def ask_question_locally(question):
    """Does what our FastAPI /ask endpoint used to do."""
    if st.session_state.vector_index is None:
        return {"error": "Please upload a document first."}
        
    query_vector = model.encode([question]).astype('float32')
    distances, indices = st.session_state.vector_index.search(query_vector, k=3)
    retrieved_chunks = [st.session_state.text_chunks[idx] for idx in indices[0]]
    
    # Call Hugging Face API for the LLM
    final_messages = build_rag_prompt(question, retrieved_chunks)
    hf_token = os.getenv("HF_TOKEN")
    hf_url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    data = {"messages": final_messages, "max_tokens": 500}
    
    try:
        response = requests.post(hf_url, headers=headers, json=data)
        response.raise_for_status()
        result_json = response.json()
        answer = result_json[0]["generated_text"]
        if "assistant" in answer:
            answer = answer.split("assistant")[-1].strip()
        return {"answer": answer, "sources": retrieved_chunks}
    except Exception as e:
        return {"error": str(e)}

# --- USER INTERFACE ---

with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Extracting text, generating embeddings..."):
                result = process_pdf_locally(uploaded_file)
                st.success(result)
                st.session_state.messages = [] # Clear chat history

st.title("💬 Chat with your Document")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        response = ask_question_locally(prompt)
    
    if "error" in response:
        answer = f"❌ Error: {response['error']}"
    else:
        answer = response.get("answer", "No answer received.")
    
    with st.chat_message("assistant"):
        st.markdown(answer)
        if "sources" in response and response["sources"]:
            with st.expander("📄 View Source Chunks"):
                for i, source in enumerate(response["sources"]):
                    st.text_area(f"Chunk {i+1}", value=source, height=100, key=f"src_{i}_{prompt}")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})