import numpy as np
import faiss
import requests
import os
import sys

# Import all our previously built modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pdf_extractor import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeder import model
from src.prompt_builder import build_rag_prompt

def ask_question(pdf_path, question, k=3):
    """
    Takes a PDF path and a question, returns an AI-generated answer.
    """
    print("1. Extracting and chunking text...")
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    
    print("2. Generating embeddings and searching...")
    # Embed all chunks
    chunk_embeddings = model.encode(chunks)
    chunk_embeddings = np.array(chunk_embeddings).astype('float32')
    
    # Build FAISS index
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    
    # Embed the question and search
    query_vector = model.encode([question]).astype('float32')
    distances, indices = index.search(query_vector, k)
    
    # Step 3: Retrieve the actual text of the top k chunks
    # indices[0] is an array like [5, 12, 2]. We use those numbers to grab the text.
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    
    print("3. Building prompt...")
   
    # Step 4: Build the RAG prompt
    final_prompt = build_rag_prompt(question, retrieved_chunks)
    
    print("4. Asking Ollama for the answer...")
    # Step 5: Send to Ollama
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.1:8b", # Change to llama3.2:3b if you used that
        "prompt": final_prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    answer = response.json().get("response", "Error: Ollama returned an empty response.")

    # NEW: Package the answer and the sources together
    result = {
        "answer": answer,
        "sources": retrieved_chunks
    }
    
    return result

# --- Test the Full Pipeline ---
if __name__ == "__main__":
    pdf_path = os.path.join("data", "test.pdf")
    
    # Use your simple test PDF question
    my_question = "which are twin satellites were launch together?" 
    
    if os.path.exists(pdf_path):
        print(f"\nAsking: '{my_question}'\n")
        
        # Now we receive a dictionary, not just a string
        result = ask_question(pdf_path, my_question)
        
        # Print the answer
        print("\n" + "="*50)
        print("FINAL AI ANSWER:")
        print("="*50)
        print(result["answer"])
        
        # Print the sources
        print("\n" + "="*50)
        print("SOURCES USED BY THE LLM:")
        print("="*50)
        for i, source in enumerate(result["sources"]):
            print(f"[Source {i+1}]")
            print(source)
            
    else:
        print(f"Error: Could not find {pdf_path}")