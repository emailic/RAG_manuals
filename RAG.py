import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
from PyPDF2 import PdfReader
import pytesseract
import pdf2image


def get_text_from_pdf_page(source_doc, page_number, pdfs):
    """Extracts text from a specific PDF page, using OCR if necessary."""
    pdf_path = pdfs.get(source_doc)
    page_number = int(page_number)
    
    if not pdf_path:
        raise ValueError(f"PDF for {source_doc} not found.")
    
    reader = PdfReader(pdf_path)
    if page_number < 1 or page_number > len(reader.pages):
        raise ValueError(f"Invalid page number {page_number}. The document has {len(reader.pages)} pages.")

    text = reader.pages[page_number - 1].extract_text()
    if text and text.strip():
        return text.strip()
    
    # If no text found, try OCR
    images = pdf2image.convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    return pytesseract.image_to_string(images[0]).strip() if images else ""

def query_pinecone(question, index):
    """Query Pinecone with an embedding from OpenAI."""
    embedding = openai.embeddings.create(input=question, model="text-embedding-3-small")
    vector = embedding.data[0].embedding
    results = index.query(vector=vector, top_k=8, include_metadata=True)
    return results['matches']

# def query_pinecone(question: str, indexw):
#     """Mock function for querying Pinecone to avoid API costs."""
#     return [
#         {'id': 'doc_Caterpillar 3500_page_125', 'metadata': {'page': 125, 'source': 'Caterpillar 3500'}, 'score': 0.65},
#         {'id': 'doc_Caterpillar 3500_page_124', 'metadata': {'page': 124, 'source': 'Caterpillar 3500'}, 'score': 0.62},
#         {'id': 'doc_Caterpillar 3500_page_122', 'metadata': {'page': 122, 'source': 'Caterpillar 3500'}, 'score': 0.60},
#         {'id': 'doc_Waukesha VGF_page_248', 'metadata': {'page': 248, 'source': 'Waukesha VGF'}, 'score': 0.59},
#         {'id': 'doc_Caterpillar 3500_page_123', 'metadata': {'page': 123, 'source': 'Caterpillar 3500'}, 'score': 0.58}
#     ]

def choose_manual(manuals_retrieved):
    while True:
        selected_manual = input("Looks like we found an answer to your question in two manuals. Type W for Waukesha or C for Caterpillar: ").strip().lower()
        if selected_manual == "w":
            return 'Waukesha VGF'
        if selected_manual == "c":
            return 'Caterpillar 3500'
        print("That is not an option. Please type W for Waukesha or C for Caterpillar.")

def main():
    """Main function to run the RAG pipeline."""
    # Load environment variables
    load_dotenv()  

    # Fetch API keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "rag-engine-manual-test"

    # Ensure both API keys are available
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        raise ValueError("Missing required API keys. Check your .env file.")

    # Initialize OpenAI and Pinecone clients
    openai.api_key = OPENAI_API_KEY
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    index = pinecone.Index(INDEX_NAME)

    # Paths to manuals
    pdfs = {
        "Caterpillar 3500": "manuals/Caterpillar-3500-generator-sets-operation-and-maintenance-manual.pdf",
        "Waukesha VGF": "manuals/Waukesha_VGF_f18g.pdf"
    }

    print("Starting RAG pipeline...")

    question = "How to clean and maintain the air filter?"
    results = query_pinecone(question, index)

    # Get list of unique manuals from query results
    manuals_retrieved = list({result["metadata"]["source"] for result in results})
    print(f"Manual(s) retrieved: {manuals_retrieved}")

    # Let user select the relevant manual
    relevant_manual = choose_manual(manuals_retrieved)

    # Extract and print relevant text from the chosen manual
    for match in results:
        source, page = match["metadata"]["source"], match["metadata"]["page"]
        if source == relevant_manual:
            print("--------------------------------------------------------")
            print(f"Manual: {source}, Page: {page}")
            page_text = get_text_from_pdf_page(source, page, pdfs)
            print(f"Text: {page_text}\n")

if __name__ == "__main__":
    main()