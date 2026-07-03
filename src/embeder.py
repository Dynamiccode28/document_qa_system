from sentence_transformers import SentenceTransformer
import os
import sys

# Ensure the parent directory is in the path (just in case)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load the model once when the app starts
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')