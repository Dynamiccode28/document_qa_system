from sentence_transformers import SentenceTransformer
import os
import sys

# Add parent directory to path so we can import our previous files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.text_chunker import chunk_text
from src.pdf_extractor import extract_text_from_pdf

# Step 1: Load the model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Get our chunks
pdf_path = os.path.join("data", "test.pdf")
print(f"Extracting and chunking {pdf_path}...")
raw_text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(raw_text, chunk_size=500)

print(f"Generated {len(chunks)} chunks.")

# Step 3: Convert chunks to embeddings (vectors)
print("Generating embeddings (this might take a few seconds)...")
embeddings = model.encode(chunks)

# Step 4: Look at the results
print("\n--- RESULTS ---")
print(f"Type of embeddings object: {type(embeddings)}")
print(f"Shape of embeddings: {embeddings.shape}")

print("\n--- FIRST CHUNK ---")
print(chunks[0])

print("\n--- FIRST EMBEDDING (First 10 numbers out of 384) ---")
print(embeddings[0][:10])