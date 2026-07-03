import streamlit as st
import numpy as np
import faiss
import os
import sys

# --- IMPORT RAG COMPONENTS ---
from src.pdf_extractor import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeder import model

# --- IMPORT TINY LOCAL LLM ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Document QA", page_icon="📄")

# --- LOAD TINY LLM (Happens once when the app starts) ---
@st.cache_resource
def load_tiny_llm():
    print("Loading local Qwen 0.5B LLM...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm_model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=lm_model, tokenizer=tokenizer, max_new_tokens=200, do_sample=False)

tiny_llm = load_tiny_llm()

# --- SESSION STATE ---
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BACKEND LOGIC ---
def process_pdf_locally(file):
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
    
    st.session_state.vector_index = index
    st.session_state.text_chunks = chunks
    return f"Processed {len(chunks)} chunks successfully."

def ask_question_locally(question):
    if st.session_state.vector_index is None:
        return {"error": "Please upload a document first."}
        
    query_vector = model.encode([question]).astype('float32')
    distances, indices = st.session_state.vector_index.search(query_vector, k=3)
    retrieved_chunks = [st.session_state.text_chunks[idx] for idx in indices[0]]
    
    # Build a simple prompt for the tiny LLM
    context_text = "\n".join(retrieved_chunks)
    prompt = f"Context: {context_text}\n\nQuestion: {question}\nAnswer based only on the context:"
    
    # Generate answer locally!
    result = tiny_llm(prompt)
    answer = result[0]['generated_text']
    
    # Clean up the output (remove the prompt from the response)
    if "Answer based only on the context:" in answer:
        answer = answer.split("Answer based only on the context:")[-1].strip()
        
    return {"answer": answer, "sources": retrieved_chunks}

# --- USER INTERFACE ---
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Extracting text, generating embeddings..."):
                result = process_pdf_locally(uploaded_file)
                st.success(result)
                st.session_state.messages = []

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