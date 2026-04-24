# Constitution RAG

A retrieval-augmented generation (RAG) prototype for the Indian Constitution. This project loads constitutional articles into a Qdrant vector store, encodes them with Sentence Transformers, and uses Google Gemini via LangChain to answer user questions based on retrieved constitutional text.

## Project Overview

- Loads the `Sharathhebbar24/Indian-Constitution` dataset.
- Creates or connects to a Qdrant collection named `constitution`.
- Encodes articles using `sentence-transformers` (`all-MiniLM-L6-v2`).
- Inserts article embeddings and payload into Qdrant.
- Runs semantic search on user queries.
- Uses `ChatGoogleGenerativeAI` via LangChain to generate responses from retrieved context.

## Features

- Dataset ingestion from Hugging Face `datasets`.
- Vector store creation and management in Qdrant.
- Sentence embedding generation with `SentenceTransformer`.
- Semantic search for relevant constitutional articles.
- Context-aware question answering using Google Gemini.
- Interactive command-line interface for insertion, search, and cleanup.

## Workflow

1. Load constitutional dataset from `data.py`.
2. Initialize the Qdrant client using environment variables.
3. Create the `constitution` collection if it does not exist.
4. Convert dataset examples into text and generate embeddings.
5. Upsert points into Qdrant with article metadata and text.
6. Accept a user query and encode it to a query vector.
7. Retrieve the top matching articles from Qdrant.
8. Build a context prompt from retrieved results.
9. Send the context and query to the Gemini model.
10. Print the generated response.

## Setup

### Prerequisites

- Python 3.10+ recommended
- Qdrant instance accessible via URL and API key
- Google Generative AI access configured in environment

### Installation

1. Clone the repository or open the workspace.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the following variables:

```env
QDRANT_URL=https://your-qdrant-instance-url
QDRANT_API_KEY=your-qdrant-api-key
```

4. Ensure you have valid Google Gemini credentials configured in your environment if required by `langchain_google_genai`.

## Usage

Run the main script:

```bash
python main.py
```

The command-line menu supports:

- `1` Insert all Data
- `2` Search by Query
- `3` Delete Collection
- `4` Exit

### Insert Data

Choose option `1` to load the dataset, encode it, and insert points into the `constitution` collection.

### Search

Choose option `2` and enter a question. The tool will:

- embed the query
- search Qdrant for the top 3 most relevant constitutional articles
- pass the retrieved context into Google Gemini
- print the generated answer

Enter `exit` inside the query prompt to return to the main menu.

### Delete Collection

Choose option `3` to remove the `constitution` collection from Qdrant.

## Tech Stack

- Python
- Qdrant (`qdrant-client`)
- Sentence Transformers (`sentence-transformers`)
- Hugging Face Datasets (`datasets`)
- LangChain and LangChain Core
- Google Gemini via `langchain_google_genai`
- dotenv for environment configuration

## Important Files

- `main.py` - main application logic, collection management, indexing, search, and LLM invocation
- `data.py` - dataset loader for the Indian Constitution dataset
- `requirements.txt` - Python dependencies

## Notes

- The Qdrant collection is configured with cosine distance and an HNSW index.
- The system prompt is set for a legal assistant and instructs the model not to hallucinate.
- Make sure the dataset is accessible and that credentials are valid before running the script.

