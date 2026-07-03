# 1. Specify the base image with Python 3.10 pre-installed
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system-level dependencies required by sentence-transformers and faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY the requirements file first (to leverage Docker layer caching)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code into the container
COPY . .

# 7. Create a directory for user uploads so the container doesn't crash
RUN mkdir -p /app/data

# 8. Expose the port our Streamlit app will run on
EXPOSE 8501

# 9. The command to start the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]