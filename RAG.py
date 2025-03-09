import os
from dotenv import load_dotenv
import pinecone
import openai
from PyPDF2 import PdfReader
from tqdm import tqdm
client = openai.OpenAI()
#from sentence_transformers import SentenceTransformer

# Initialize API keys
load_dotenv()  # Load environment variables from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "rag-engine-manuals"

if not PINECONE_API_KEY:
    raise ValueError("Missing required API keys. Check your .env file.")

# Initialize OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create a Pinecone index if it doesn't exist
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(name=INDEX_NAME, metric="cosine", dimension=1536)  # OpenAI embeddings are 1536-d
index = pinecone.Index(INDEX_NAME)

# Function to clear the Pinecone index
def clear_pinecone_index(index):
    index.delete(delete_all=True)
    print("✅ Pinecone index cleared.")

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
    return text_chunks

# Process both PDFs
pdfs = {
    "Caterpillar 3500": "manuals/Caterpillar-3500-generator-sets-operation-and-maintenance-manual.pdf",
    "Waukesha VGF": "manuals/Waukesha_VGF_f18g.pdf"
}

# Store document embeddings
docs = []
for manual, pdf_path in pdfs.items():
    text_chunks = extract_text_from_pdf(pdf_path)
    for text, page in tqdm(text_chunks, desc=f"Indexing {manual}"):
        embedding = client.embeddings.create(input=text, model="text-embedding-ada-002")
        vector = embedding["data"][0]["embedding"]
        metadata = {"source": manual, "page": page}
        index.upsert(vectors=[("doc_{}_{}".format(manual, page), vector, metadata)])
        docs.append((text, manual, page))

# Query function
def query_pinecone(question):
    embedding = openai.Embedding.create(input=question, model="text-embedding-ada-002")
    vector = embedding["data"][0]["embedding"]
    results = index.query(vector=vector, top_k=5, include_metadata=True)
    return results["matches"]

# Example query
question = "How to clean and maintain the air filter?"
results = query_pinecone(question)
for match in results:
    print(f"Manual: {match['metadata']['source']}, Page: {match['metadata']['page']}")
    print(f"Text: {docs[int(match['id'].split('_')[-1])][0]}\n")
