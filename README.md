

# RAG Engine with Pinecone and OpenAI

This project implements a Retrieval-Augmented Generation (RAG) system that indexes manuals using Pinecone and OpenAI, processes PDF documents (both searchable and scanned), and provides answers based on the indexed content. It utilizes a combination of Pinecone for vector storage and search, OpenAI's embeddings for querying, and Optical Character Recognition (OCR) via Tesseract to process non-searchable PDFs.

## Resources  

- **[Miro Board](https://miro.com/app/board/uXjVIE2Elfg=/?share_link_id=974868060599)**  
- **[Medium Article](https://medium.com/@ema.ilic9/building-a-rag-system-for-heavy-machinery-engine-manuals-5aacf9043892)**  

## Prerequisites

- Python 3.12
- Poetry (for managing dependencies)
- Pinecone API Key
- OpenAI API Key

## Setup

### 1. Clone the repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Install dependencies

This project uses Poetry for dependency management. Install Poetry and then install the project dependencies:

```bash
pip install poetry
poetry install
```

### 3. Create `.env` file

Create a `.env` file in the project root directory with the following content:

```bash
OPENAI_API_KEY=<your_openai_api_key>
PINECONE_API_KEY=<your_pinecone_api_key>
```

Ensure that the `.env` file is listed in `.gitignore` to prevent it from being uploaded to version control.

### 4. Install Tesseract and other dependencies

For OCR functionality, you will need Tesseract installed on your system.

- **On Ubuntu:**

```bash
sudo apt-get install tesseract-ocr
```

- **On macOS (using Homebrew):**

```bash
brew install tesseract
```

Additionally, you will need to install `pdf2image`:

- **On Ubuntu:**

```bash
sudo apt-get install poppler-utils  # Required for pdf2image
```
- **On macOS (using Homebrew):**

```bash
brew install poppler
```

## Running the Project

### 1. Indexing PDFs in Pinecone

To index PDFs into Pinecone, run the `indexing_pinecone.py` script. This script extracts text from the provided PDFs (handling both searchable and scanned PDFs), generates embeddings using OpenAI, and upserts the vectors into Pinecone.

```bash
python indexing_pinecone.py
```

Ensure you have the necessary PDFs (e.g., `Caterpillar-3500-generator-sets-operation-and-maintenance-manual.pdf` and `Waukesha_VGF_f18g.pdf`) in the `manuals/` directory.

### 2. Running the RAG Pipeline

Once your PDFs are indexed, you can run the `RAG.py` script to query the indexed manuals. The script allows you to ask a question, and it retrieves the most relevant pages from the manuals using Pinecone and displays the text.

```bash
python RAG.py
```

You will be prompted to choose between the retrieved manuals (e.g., `Caterpillar` or `Waukesha`) to get the relevant information.

## Functions

### `indexing_pinecone.py`

- **initialize_pinecone()**: Initializes Pinecone, creating an index if necessary.
- **clear_pinecone_index(index)**: Clears the Pinecone index before inserting new data.
- **extract_text_from_pdf(pdf_path)**: Extracts text from a PDF, using OCR if necessary for scanned documents.
- **main**: Upserts each page with the relevant metadata to Pinecone.

### `RAG.py`

- **get_text_from_pdf_page(source_doc, page_number)**: Extracts text from a specific page of a PDF, using OCR if necessary.
- **query_pinecone(question)**: Queries Pinecone for the most relevant pages based on a question.
- **choose_manual(manuals_retrieved)**: Allows the user to select the relevant manual from the results.
- **main**: Retrieves chunks (pages) relevant to the selected manual.


