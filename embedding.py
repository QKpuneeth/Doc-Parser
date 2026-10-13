import os

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

load_dotenv()

# ─── Defaults (overridable via .env) ─────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL_ENDPOINT = "https://ai.questkart.cloud/embeddings"
COLLECTION_NAME = "dummy_emb"
BATCH_SIZE = 32


class QuestkartEmbeddings(Embeddings):
    """LangChain wrapper around our custom embedding endpoint (POST {"texts": [...]})."""

    def __init__(self, url: str | None = None, batch_size: int = BATCH_SIZE, timeout: int = 30):
        self.url = url or os.getenv("EMBEDDING_MODEL_ENDPOINT", DEFAULT_EMBEDDING_MODEL_ENDPOINT)
        self.batch_size = batch_size
        self.timeout = timeout

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(self.url, json={"texts": texts},
                             headers={"content-type": "application/json"}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed(texts[i:i + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def get_vector_store() -> PGVector:
    # Must use the psycopg3 driver: postgresql+psycopg://user:pass@host:port/db
    connection = os.environ["PGVECTOR_CONNECTION"]
    return PGVector(
        embeddings=QuestkartEmbeddings(),
        collection_name=COLLECTION_NAME,
        connection=connection,
        use_jsonb=True,
    )


def main():
    vector_store = get_vector_store()

    docs = [
        Document(page_content="there are cats in the pond",
                 metadata={"id": 1, "location": "pond", "topic": "animals"}),
        Document(page_content="ducks are also found in the pond",
                 metadata={"id": 2, "location": "pond", "topic": "animals"}),
        Document(page_content="fresh apples are available at the market",
                 metadata={"id": 3, "location": "market", "topic": "food"}),
        Document(page_content="the market also sells fresh oranges",
                 metadata={"id": 4, "location": "market", "topic": "food"}),
        Document(page_content="the new art exhibit is fascinating",
                 metadata={"id": 5, "location": "museum", "topic": "art"}),
        Document(page_content="a sculpture exhibit is also at the museum",
                 metadata={"id": 6, "location": "museum", "topic": "art"}),
        Document(page_content="a new coffee shop opened on Main Street",
                 metadata={"id": 7, "location": "Main Street", "topic": "food"}),
        Document(page_content="the book club meets at the library",
                 metadata={"id": 8, "location": "library", "topic": "reading"}),
        Document(page_content="the library hosts a weekly story time for kids",
                 metadata={"id": 9, "location": "library", "topic": "reading"}),
        Document(page_content="a cooking class for beginners is offered at the community center",
                 metadata={"id": 10, "location": "community center", "topic": "classes"}),
    ]

    # Same ids on re-run → rows are upserted, not duplicated
    vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
    print(f"Inserted/updated {len(docs)} documents into collection '{COLLECTION_NAME}'")

    query = "kitty"
    print(f"\nSimilarity search for '{query}':")
    for doc, score in vector_store.similarity_search_with_score(query, k=3):
        print(f"  [{score:.4f}] {doc.page_content}  {doc.metadata}")

    query = "where can I buy fruit?"
    print(f"\nFiltered search for '{query}' (topic = food):")
    for doc in vector_store.similarity_search(query, k=3, filter={"topic": {"$eq": "food"}}):
        print(f"  {doc.page_content}  {doc.metadata}")


if __name__ == "__main__":
    main()
 
 