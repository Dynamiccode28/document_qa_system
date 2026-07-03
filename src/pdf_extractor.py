import PyPDF2
import os

def extract_text_from_pdf(pdf_path):
    """
    Reads a PDF file and returns all its text as a single string.
    """
    # Initialize an empty string to hold all the text
    text = ""
    
    # Step 1: Open the PDF file in "read binary" mode ('rb')
    with open(pdf_path, 'rb') as file:
        
        # Step 2: Create a PDF reader object
        reader = PyPDF2.PdfReader(file)
        
        # Step 3: Find out how many pages the PDF has
        num_pages = len(reader.pages)
        print(f"Found {num_pages} pages in the PDF.")
        
        # Step 4: Loop through every page
        for page_num in range(num_pages):
            
            # Get the specific page
            page = reader.pages[page_num]
            
            # Extract text from this page
            page_text = page.extract_text()
            
            # Add it to our total text, with a newline separator
            text += page_text + "\n"
            
    return text

# --- Test the function ---
if __name__ == "__main__":
    # Path to the PDF you put in the data folder
    pdf_file_path = os.path.join("data", "test.pdf")
    
    # Check if file exists before trying to read it
    if os.path.exists(pdf_file_path):
        extracted_text = extract_text_from_pdf(pdf_file_path)
        print("\n--- EXTRACTED TEXT ---")
        print(extracted_text[:500]) # Print only the first 500 characters to keep it short
        print("...(truncated for display)...")
    else:
        print(f"Error: Could not find file at {pdf_file_path}")