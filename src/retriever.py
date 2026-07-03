import numpy as np
import faiss
import os
import sys

#adding parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pdf_extractor import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeder import model

def build_retriever(pdf_path,chunk_size=500):
    #getting text and chunks
    print("Extracting  & chunking...")
    raw_text=extract_text_from_pdf(pdf_path)
    chunks=chunk_text(raw_text,chunk_size=chunk_size)

    #generating embeddings
    print("Generating embeddings...")
    embeddings=model.encode(chunks)

    #preparing data for FAISS
    #FAISS requires data to be 2d numpy array of float32
    embeddings=np.array(embeddings).astype('float32')

    #getting dimensions of vector
    dimension=embeddings.shape[1]
    print(f"vector dimensio:{dimension}")

    #builsing FAISS indexing
    print("Building FAISS index...")
    index=faiss.IndexFlatL2(dimension)

    #add vectors to the index
    index.add(embeddings)
    print(f"stored {index.ntotal} vectors in FAISS")
    return index,chunks

def search_query(index,chunks,query_text,k=3):
    #query embedding
    print(f"\nSearching for: '{query_text}'")
    query_vector=model.encode([query_text])
    query_vector=np.array(query_vector).astype('float32')
    
    #search the index k=how many teop results we want back
    distances,indices=index.search(query_vector,k)

    #displaying results
    print("\n--- TOP RESULTS ---")
    for i in range(k):
        chunk_ind=indices[0][i]
        distance=distances[0][i]
        print(f"\n Results {i+1} (Distance:{distance:4f})")
        print('-'*40)
        print(chunks[chunk_ind])
        print('-'*40)

# --- Test the system ---
if __name__ == "__main__":
    pdf_path = os.path.join("data", "test.pdf")
    
    if os.path.exists(pdf_path):
        # Build the index
        index, chunks = build_retriever(pdf_path)
        
        # Ask a question (Change this to match what your test.pdf is about!)
        my_question = "What is the main topic of this document?"
        
        # Search
        search_query(index, chunks, my_question, k=2)
    else:
        print(f"Error: Could not find {pdf_path}")