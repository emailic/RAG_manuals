import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
from PyPDF2 import PdfReader
import pytesseract
import pdf2image

client = openai.OpenAI()

print ("RUNNING RAG.py")

# Initialize API keys
load_dotenv()  # Load environment variables from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-engine-manual-test"

if not PINECONE_API_KEY:
    raise ValueError("Missing required API keys. Check your .env file.")

pdfs = {
    "Caterpillar 3500": "manuals/Caterpillar-3500-generator-sets-operation-and-maintenance-manual.pdf",
    "Waukesha VGF": "manuals/Waukesha_VGF_f18g.pdf"
}

# Function to load the PDF and extract text from a specific page
def get_text_from_pdf_page(source_doc, page_number):
    print("Using regular pdf2text...")
    pdf_path = pdfs.get(source_doc)
    page_number = int(page_number)
    if not pdf_path:
        raise ValueError(f"PDF for {source_doc} not found.")
    
    reader = PdfReader(pdf_path)

    # Check if the page number is valid
    if page_number < 1 or page_number > len(reader.pages):
        raise ValueError(f"Invalid page number {page_number}. The document has {len(reader.pages)} pages.")

    # Try extracting text normally
    page = reader.pages[page_number - 1]
    text = page.extract_text()

    if text and text.strip():
        return text.strip()

    # If no text was extracted, assume it's a scanned PDF and use OCR
    print(f"Page {page_number} appears to be scanned, regular pdf2text failed. Using OCR...")
    images = pdf2image.convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    
    if images:
        text = pytesseract.image_to_string(images[0])
        return text.strip()

    return ""

# Initialize OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index = pinecone.Index(INDEX_NAME)
stats = index.describe_index_stats()
print("STATS...................\n", stats)

# Query function
def query_pinecone(question):
    embedding = client.embeddings.create(input=question, model="text-embedding-3-small")
    vector = embedding.data[0].embedding
    results = index.query(vector=vector, top_k=5, include_metadata=True)
    return results["matches"]

# Example query
question = "How to clean and maintain the air filter?"
results = query_pinecone(question)
print("RESULTS", results)

#lets see if we retrieved chunks from more than one manual
manuals_retrieved= list()

for result in results:
    source_manual = result["metadata"]["source"]
    if source_manual not in manuals_retrieved:
        manuals_retrieved.append(source_manual)

print ("number of manuals retrieved: ", len(manuals_retrieved))

#introduce the logic for making the user choose the relevant manual

def choose_manual(manuals_retrieved):
    while True:
        selected_manual = input("Looks like we found an answer to your question in two manuals. Type W for Waukesha or C for Caterpillar: ").strip().lower()
        if selected_manual == "w":
            return 'Waukesha VGF'
        if selected_manual == "c":
            return 'Caterpillar 3500'
        print("That is not an option. Please type W for Waukesha or C for Caterpillar.")



#user chooses the relevant manual and gets results only relevant to his manual.

relevant_manual = choose_manual(manuals_retrieved)

for match in results:
    source = match['metadata']['source']
    page = match['metadata']['page']
    if source == relevant_manual:
        print(f"Manual: {source}, Page: {page}")
        page_text = get_text_from_pdf_page(source, page)
        #print(f"Text: {page_text}\n")

