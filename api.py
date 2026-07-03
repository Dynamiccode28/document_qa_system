import os
import shutil
import numpy as np
import faiss
import requests
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

#import implemented custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.pdf_extractor import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeder import model
from src.prompt_builder import build_rag_prompt

#-----global state------
#This variables stay alive in RAM between different reuests from user
vector_index=None
text_chunks=None
is_ready=False

#------ FASTAPI APP ------
app=FastAPI(title="RAG Document QA API")

#----- DATA MODELS -----
class QuestionRequest(BaseModel):
    text:str

#ENDPOINT

@app.get("/")
def health_check():
    return {"status":"API is running!","document_loaded":is_ready}

@app.post("/upload")
async def upload_pdf(file:UploadFile=File(...)):
    global vector_index,text_chunks,is_ready

    #1. enforcing 10MB size limits
    max_size=10*1024*1025
    if file.size>max_size:
        raise HTTPException(status_code=413,detail="File too large,Max size is 10 MB")
    
    #2.save the uploded file to data/folder
    save_path=os.path.join("data",file.filename)
    with open(save_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
    try:
        #3.run the rag preparation pipeline
        raw_text=extract_text_from_pdf(save_path)
        text_chunks=chunk_text(raw_text)
        embeddings=model.encode(text_chunks)
        embeddings=np.array(embeddings).astype('float32')
        dimension=embeddings.shape[1]
        vector_index=faiss.IndexFlatL2(dimension)
        vector_index.add(embeddings)
        is_ready=True
        return {"message": f"Successfully processed {file.filename}. Ready for questions!"}
    except Exception as e:
        is_ready=False
        raise HTTPException(status_code=500,detail=f"Failed to process PDF:{str(e)}")
    
@app.post("/ask")
def ask_question(request: QuestionRequest):
    global vector_index, text_chunks, is_ready
    
    # 1. Check if a document has been uploaded
    if not is_ready or vector_index is None:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a PDF first.")
        
    # 2. Retrieve relevant chunks using FAISS
    query_vector = model.encode([request.text]).astype('float32')
    distances, indices = vector_index.search(query_vector, k=3)
    retrieved_chunks = [text_chunks[idx] for idx in indices[0]]
    
    # 3. Build the prompt and ask Hugging Face Serverless API
    import os
    final_messages = build_rag_prompt(request.text, retrieved_chunks)
    
    # We will set this 'HF_TOKEN' in the cloud dashboard later
    hf_token = os.getenv("HF_TOKEN") 
    hf_url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": final_messages,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(hf_url, headers=headers, json=data)
        response.raise_for_status()
        
        # HF API returns the answer inside a list with a 'generated_text' key
        result_json = response.json()
        answer = result_json[0]["generated_text"]
        
        # The HF API sometimes echoes the prompt, we just want the assistant's reply
        if "assistant" in answer:
            answer = answer.split("assistant")[-1].strip()
        
        return {
            "answer": answer,
            "sources": retrieved_chunks
        }
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama is not running. Please start Ollama.")
