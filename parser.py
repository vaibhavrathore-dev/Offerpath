import fitz  # this is PyMuPDF

def extract_text_from_pdf(pdf_file):
    """
    Takes a PDF file object (from Streamlit uploader)
    and returns all the text as a single string.
    """
    # fitz.open reads the PDF from raw bytes
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    full_text = ""
    
    # loop through every page in the PDF
    for page_number in range(len(doc)):
        page = doc[page_number]
        full_text += page.get_text()  # extract text from this page
    
    return full_text.strip()  # strip removes leading/trailing whitespace