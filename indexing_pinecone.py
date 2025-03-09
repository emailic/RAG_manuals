import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import pdf2image

from tqdm import tqdm
import time
client = openai.OpenAI()
#from sentence_transformers import SentenceTransformer

print("RUNNIG INDEXING PINECONE SCRIPT")
# Initialize API keys
load_dotenv()  # Load environment variables from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "rag-engine-manual-test"

if not PINECONE_API_KEY:
    raise ValueError("Missing required API keys. Check your .env file.")

# Initialize OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create a Pinecone index if it doesn't exist
if INDEX_NAME not in pinecone.list_indexes().names():
    pinecone.create_index(name=INDEX_NAME, dimension=1536, metric="cosine", spec = ServerlessSpec(cloud = "aws", region= 'us-east-1'))  # OpenAI embeddings are 1536-d
    print(f"Creating index {INDEX_NAME}...")

# Wait for the index to become available
while INDEX_NAME not in pinecone.list_indexes().names():
    print("Waiting for the index to be ready...")
    time.sleep(5)

index = pinecone.Index(INDEX_NAME)
stats = index.describe_index_stats()
print(stats)


def clear_pinecone_index(index):
    stats = index.describe_index_stats()
    total_vectors = stats["total_vector_count"]

    if total_vectors > 0:
        print(f"Clearing {total_vectors} vectors from index {index}...")
        index.delete(delete_all=True)
        print("✅ Pinecone index cleared.")
    else:
        print("⚠️ Pinecone index is already empty. No deletion performed.")


print("Existing indexes:", pinecone.list_indexes().names())

# Run this before inserting new data
clear_pinecone_index(index)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text_chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
             text_chunks.append((text, i + 1))  # Store text along with page number
    # Caterpillar manual cant be read, likely because it is scanned. Thus we extract text from images:
    if len(text_chunks)==0:
        print(f"PDF Manual was scanned. Manual path: {pdf_path}")
        images = pdf2image.convert_from_path(pdf_path)
        for i, img in tqdm(enumerate(images), desc = "Extracting text from images"):
            text = pytesseract.image_to_string(img)
            if text:
                text_chunks.append((text, i + 1))  # Store text along with page number
    return text_chunks


# Process both PDFs
pdfs = {
    "Caterpillar 3500": "manuals/Caterpillar-3500-generator-sets-operation-and-maintenance-manual.pdf",
    "Waukesha VGF": "manuals/Waukesha_VGF_f18g.pdf"
}

print(f"Using index: {index}")

# Store document embeddings
docs = []
for manual, pdf_path in pdfs.items():
    text_chunks = extract_text_from_pdf(pdf_path)
    for text, page in tqdm(text_chunks, desc=f"Indexing {manual}"):
        embedding = client.embeddings.create(input=text, model="text-embedding-3-small")
        vector = embedding.data[0].embedding
        metadata = {"source": manual, "page": page}
        index.upsert(vectors=[("doc_{}_page_{}".format(manual, page), vector, metadata)])
        docs.append((text, manual, page))
print("Manuals chunked, vectorized, and upserted into Pinecone database.")