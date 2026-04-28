from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(uploaded_file):
    """
    Extract text content from an uploaded PDF file.
    """
    text = ""
    try:
        pdf_reader = PdfReader(uploaded_file)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i + 1} ---\n"
                text += page_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        
    return text

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split the extracted text into manageable chunks for the Vector Store.
    We use Langchain's RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks
