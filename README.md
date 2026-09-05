# Document Q&A — Retrieval-Augmented Generation

A document question-answering application that allows users to upload a PDF and ask natural-language questions about its contents.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant sections from the uploaded document and provide grounded answers using Cohere's language model.

---

## Overview

Traditional chatbots may generate answers from their general training data, which can lead to incorrect or unsupported information.

This project uses a RAG pipeline:

1. Extract text from a PDF.
2. Split the text into meaningful chunks.
3. Convert the chunks into vector embeddings.
4. Store the embeddings in Pinecone.
5. Convert the user's question into an embedding.
6. Retrieve the most relevant document sections.
7. Provide those sections to the language model.
8. Generate an answer based only on the retrieved document context.

This allows the chatbot to answer questions about documents without requiring the entire document to be included in every prompt.

---

## Features

- Upload PDF documents through a Streamlit interface
- Extract text from PDFs using PyMuPDF
- Sentence-aware and paragraph-aware text chunking
- Generate embeddings using Cohere `embed-v4.0`
- Store and search document embeddings using Pinecone
- Retrieve the most relevant document sections for each question
- Generate grounded answers using Cohere Command A
- Display retrieved context for transparency
- Show page numbers for retrieved document sections
- Maintain conversation history during the session
- Process a document once and ask multiple questions
- Isolate different uploaded documents using document-specific Pinecone namespaces
- API keys are entered through the application interface rather than stored in source code

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Application development |
| Streamlit | Web interface |
| PyMuPDF | PDF text extraction |
| Cohere | Embeddings and answer generation |
| Pinecone | Vector database and similarity search |

---

## Architecture

```text
                         PDF Document
                              │
                              ▼
                    ┌──────────────────┐
                    │   PyMuPDF        │
                    │  Text Extraction │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Text Chunking   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Cohere Embedding │
                    │   embed-v4.0     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Pinecone      │
                    │   Vector Store   │
                    └────────┬─────────┘
                             │
                             │
User Question ───────────────┘
       │
       ▼
┌──────────────────┐
│ Query Embedding  │
│  search_query    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Similarity Search│
│   Top Results    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Retrieved Context│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Cohere         │
│   Command A      │
└────────┬─────────┘
         │
         ▼
      Answer