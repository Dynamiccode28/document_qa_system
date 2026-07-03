def build_rag_prompt(question,context_chunks):
    #system rules
    system_prompt= """You are a helpful assistant. 
You must answer the user's question using ONLY the information provided in the context below.
Do not use any outside knowledge.
If the context does not contain the answer, simply say "I do not have enough information to answer this."
Keep your answer concise and to the point.
"""
    #context:joining all the retrieved chunka with double newlines
    context_text= "\n\n---\n\n".join(context_chunks)

    #assembling final prompt
    final_prompt=f"""{system_prompt}

    CONTEXT:
    {context_text}
    QUESTION:
    {question}
    ANSWER:
    """
    
    return final_prompt

# --- Test the function ---
if __name__ == "__main__":
    # Fake data to simulate what FAISS will give us
    fake_question = "What is the refund policy?"
    fake_chunks = [
        "Our company offers a 30-day return policy on all electronics.",
        "Refunds are processed within 5-7 business days after we receive the item.",
        "The cafeteria serves lunch from 12 PM to 2 PM." # Irrelevant chunk!
    ]
    
    prompt = build_rag_prompt(fake_question, fake_chunks)
    
    print("--- FINAL PROMPT THAT WILL BE SENT TO LLM ---")
    print(prompt)