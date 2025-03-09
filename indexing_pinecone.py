import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai
from PyPDF2 import PdfReader
import pytesseract
import pdf2image
from tqdm import tqdm

load_dotenv()  # Load environment variables from .env
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "rag-engine-manual-test"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

if not PINECONE_API_KEY:
    raise ValueError("Missing required API keys. Check your .env file.")

client = openai.OpenAI(api_key = OPENAI_API_KEY)
pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def initialize_pinecone():
    """Create Pinecone index."""
    if INDEX_NAME not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=INDEX_NAME, dimension=1536, metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        print(f"Creating index {INDEX_NAME}...")

    while INDEX_NAME not in pinecone.list_indexes().names():
        print("Waiting for the index to be ready...")
        time.sleep(5)
    
    return pinecone.Index(INDEX_NAME)


def clear_pinecone_index(index):
    """Clear the Pinecone index before inserting new data."""
    stats = index.describe_index_stats()
    total_vectors = stats["total_vector_count"]

    if total_vectors > 0:
        print(f"Clearing {total_vectors} vectors from index {index}...")
        index.delete(delete_all=True)
        print("Pinecone index cleared.")
    else:
        print("Pinecone index is already empty. Nothing to delete.")



def extract_text_from_pdf(pdf_path):
    """Extract text from PDF, handling both searchable and scanned documents."""
    reader = PdfReader(pdf_path)
    text_chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
             text_chunks.append((text, i + 1))

    # Caterpillar manual cant be read, likely because it is scanned. Thus we extract text from images:
    if len(text_chunks)==0:
        print(f"PDF Manual needs to be processed using OCR. Path: {pdf_path}")
        images = pdf2image.convert_from_path(pdf_path)
        for i, img in tqdm(enumerate(images), desc = "Extracting text from images"):
            text = pytesseract.image_to_string(img)
            if text:
                text_chunks.append((text, i + 1)) 
    return text_chunks

if __name__ == "__main__":
    index = initialize_pinecone()
    clear_pinecone_index(index)

    pdfs = {
        "Caterpillar 3500": "manuals/Caterpillar-3500-generator-sets-operation-and-maintenance-manual.pdf",
        "Waukesha VGF": "manuals/Waukesha_VGF_f18g.pdf"
    }

    print(f"Using index: {INDEX_NAME}")
    docs = []
    
    for manual, pdf_path in pdfs.items():
        text_chunks = extract_text_from_pdf(pdf_path)
        
        for text, page in tqdm(text_chunks, desc=f"Indexing {manual}"):
            embedding = client.embeddings.create(input=text, model="text-embedding-3-small")
            vector = embedding.data[0].embedding
            metadata = {"source": manual, "page": page}
            index.upsert(vectors=[(f"doc_{manual}_page_{page}", vector, metadata)])
            docs.append((text, manual, page))
    
    print("Manuals chunked, vectorized, and upserted into Pinecone database.")




