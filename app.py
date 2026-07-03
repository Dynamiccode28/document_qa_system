import streamlit as st
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Document QA", page_icon="📄")

# --- SESSION STATE INITIALIZATION ---
# This list survives screen reruns!
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- HELPER FUNCTIONS TO TALK TO FASTAPI ---

def upload_file_to_api(file):
    """Sends the uploaded PDF to the FastAPI /upload endpoint."""
    url = "http://localhost:8000/upload"
    # FastAPI expects files in a specific dictionary format
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    try:
        response = requests.post(url, files=files, timeout=60)
        response.raise_for_status() # Raise error if API returns 400/500
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def ask_question_to_api(question):
    """Sends the question to the FastAPI /ask endpoint."""
    url = "http://localhost:8000/ask"
    data = {"text": question}
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# --- USER INTERFACE LAYOUT ---

# Put the upload feature in a sidebar so it doesn't clutter the chat
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Extracting text, generating embeddings..."):
                result = upload_file_to_api(uploaded_file)
                if "error" in result:
                    st.error(f"Failed: {result['error']}")
                else:
                    st.success(result["message"])
                    # Clear old chat history when a new document is loaded
                    st.session_state.messages = []

# Main chat area
st.title("💬 Chat with your Document")

# 1. Redraw all previous messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Wait for new user input (this creates the chat box at the bottom)
if prompt := st.chat_input("Ask a question about your document..."):
    
    # A. Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # B. Save user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # C. Send to FastAPI and get response
    with st.spinner("Thinking..."):
        response = ask_question_to_api(prompt)
    
    # D. Format the answer
    if "error" in response:
        answer = f"❌ Error: {response['error']}"
    else:
        answer = response.get("answer", "No answer received.")
    
    # E. Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)
        
        # Add a cool dropdown to show the sources used!
        if "sources" in response and response["sources"]:
            with st.expander("📄 View Source Chunks"):
                for i, source in enumerate(response["sources"]):
                    st.text_area(f"Chunk {i+1}", value=source, height=100, key=f"src_{i}_{prompt}")
    
    # F. Save assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})