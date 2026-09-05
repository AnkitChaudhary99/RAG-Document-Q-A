import re

import cohere
import fitz
from pinecone import Pinecone, ServerlessSpec


class VectorStore:
    def __init__(
        self,
        pdf_path: str,
        cohere_api_key: str,
        pinecone_api_key: str,
        document_id: str,
    ):
        self.pdf_path = pdf_path
        self.document_id = document_id

        self.co = cohere.ClientV2(api_key=cohere_api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)

        self.index_name = "rag-qa-bot-v2"
        self.namespace = f"document-{document_id}"

        self.embedding_model = "embed-v4.0"
        self.embedding_dimension = 1024
        self.retrieve_top_k = 5

        self.pages = []
        self.chunks = []
        self.embeddings = []

        self.load_pdf()
        self.split_text()

        if not self.chunks:
            raise ValueError("No readable text was found in the PDF.")

        self.embed_chunks()
        self.index_chunks()

    def load_pdf(self):
        self.pages = []

        with fitz.open(self.pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()

                if page_text:
                    self.pages.append(
                        {
                            "page": page_number,
                            "text": page_text,
                        }
                    )

    def split_text(self, chunk_size: int = 1000):
        self.chunks = []

        for page_data in self.pages:
            page_number = page_data["page"]
            page_text = page_data["text"]

            paragraphs = re.split(r"\n\s*\n", page_text)
            current_chunk = ""

            for paragraph in paragraphs:
                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                sentences = re.split(
                    r"(?<=[.!?])\s+",
                    paragraph,
                )

                for sentence in sentences:
                    sentence = sentence.strip()

                    if not sentence:
                        continue

                    # Handle extremely long sentences
                    if len(sentence) > chunk_size:
                        if current_chunk:
                            self.chunks.append(
                                {
                                    "text": current_chunk.strip(),
                                    "page": page_number,
                                }
                            )
                            current_chunk = ""

                        words = sentence.split()
                        temp = ""

                        for word in words:
                            candidate = f"{temp} {word}".strip()

                            if len(candidate) <= chunk_size:
                                temp = candidate
                            else:
                                if temp:
                                    self.chunks.append(
                                        {
                                            "text": temp,
                                            "page": page_number,
                                        }
                                    )

                                temp = word

                        if temp:
                            current_chunk = temp

                        continue

                    candidate = f"{current_chunk} {sentence}".strip()

                    if len(candidate) <= chunk_size:
                        current_chunk = candidate
                    else:
                        if current_chunk:
                            self.chunks.append(
                                {
                                    "text": current_chunk.strip(),
                                    "page": page_number,
                                }
                            )

                        current_chunk = sentence

            if current_chunk:
                self.chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "page": page_number,
                    }
                )

    def embed_chunks(self, batch_size: int = 90):
        texts = [chunk["text"] for chunk in self.chunks]

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.co.embed(
                texts=batch,
                model=self.embedding_model,
                input_type="search_document",
                output_dimension=self.embedding_dimension,
                embedding_types=["float"],
            )

            self.embeddings.extend(
                response.embeddings.float
            )

    def index_chunks(self):
        existing_indexes = self.pc.list_indexes().names()

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
                deletion_protection="disabled",
            )

        self.index = self.pc.Index(self.index_name)

        # Remove any previous vectors belonging to this document.
        try:
            self.index.delete(
                delete_all=True,
                namespace=self.namespace,
            )
        except Exception:
            pass

        vectors = []

        for i, (chunk, embedding) in enumerate(
            zip(self.chunks, self.embeddings)
        ):
            vectors.append(
                {
                    "id": str(i),
                    "values": embedding,
                    "metadata": {
                        "text": chunk["text"],
                        "page": chunk["page"],
                    },
                }
            )

        self.index.upsert(
            vectors=vectors,
            namespace=self.namespace,
        )

    def retrieve(self, query: str) -> list:
        response = self.co.embed(
            texts=[query],
            model=self.embedding_model,
            input_type="search_query",
            output_dimension=self.embedding_dimension,
            embedding_types=["float"],
        )

        query_embedding = response.embeddings.float[0]

        search_results = self.index.query(
            vector=query_embedding,
            top_k=self.retrieve_top_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        documents = []

        for match in search_results.matches:
            if match.metadata and "text" in match.metadata:
                documents.append(
                    {
                        "data": {
                            "text": match.metadata["text"]
                        },
                        "page": match.metadata.get("page"),
                    }
                )

        return documents