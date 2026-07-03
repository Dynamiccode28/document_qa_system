import re # Add this to your imports at the very top of the file

def chunk_text(text, chunk_size=500):
    """
    Splits a large string of text into smaller chunks.
    Tries to split at natural sentence boundaries.
    """
    
    # --- NEW: CLEAN THE TEXT ---
    # 1. Collapse multiple newlines down to a single newline
    text = re.sub(r'\n+', '\n', text)
    # 2. Replace all remaining newlines with a normal space
    text = text.replace('\n', ' ')
    # 3. Collapse multiple spaces down to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    # ---------------------------
    chunks=[]
    start_index=0
    text_length=len(text)

    while start_index < text_length:
        #1.guess the end of chunk
        end_index=start_index+chunk_size

        # 2: If our guess goes past the end of the document, 
        # just grab whatever is left and stop
        if end_index >= text_length:
            final_chunk=text[start_index:].strip()
            if final_chunk:
                chunks.append(final_chunk)
            break
        #3. Look back from our guess to find sentence ender
        break_point =-1
        for punctuation in ['.','!','?','\n']:
            # rfind searches BACKWARDS from 'end_index' down to 'start_index'
            found_pos=text.rfind(punctuation,start_index,end_index)
             # If we found one, and it's further along than our previous best find
            if found_pos > start_index and found_pos > break_point:
                break_point = found_pos
            
            #4.decide where to cut
            if break_point!=-1:
                chunk=text[start_index:break_point+1].strip()
                #next loop start after the punctuation
                start_index=break_point+1
            else:
                #No punctuation found. force hard cut at 500th char
                chunk=text[start_index:end_index].strip()
                #next lop starts exactly where this one is ended
                start_index=end_index
            
            #5.add the to our list 
            if chunk:
                chunks.append(chunk)
    return chunks

if __name__=="__main__":
    import os
    import sys
    #add parent directory to path so we can import pdf_extractor
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.pdf_extractor import extract_text_from_pdf

    pdf_path=os.path.join("data","test.pdf")

    if os.path.exists(pdf_path):
        print("Extracting length...")
        full_text=extract_text_from_pdf(pdf_path)

        print("chunking text...")
        chunks=chunk_text(full_text,chunk_size=500)
        print(f"\n Total chunks created : {len(chunks)}\n")

        # Print the first 3 chunks to see how it worked
        for i, chunk in enumerate(chunks[:3]):
            print(f"--- CHUNK {i+1} (Length: {len(chunk)} chars) ---")
            print(chunk)
            print()
    else:
        print(f"Error: Could not find {pdf_path}")