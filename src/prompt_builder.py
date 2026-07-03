def build_rag_prompt(question, context_chunks):
    """
    Formats the retrieved chunks into a Chat message format for HF API.
    """
    context_text = "\n\n---\n\n".join(context_chunks)
    
    # Hugging Face free API expects a list of message dictionaries
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's question using ONLY the information provided in the context. If the context does not contain the answer, say 'I do not have enough information to answer this.' Keep your answer concise."},
        {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}"}
    ]
    
    return messages