# Vector store for the user's to-read list with search and save functionality
import os
from typing import Iterable, Optional, cast

from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document

from src.core.book import Book
from src.vectorstore.chroma import EMBEDDING

TO_READ_COLLECTION = "to_read_list"
TO_READ_PERSIST_DIR = "data/chroma_to_read"


def book_to_to_read_text(book: Book, reason: str) -> str:
    """Serialize a Book object plus reason into a text block for embedding."""
    parts = [book.title]
    if book.subtitle:
        parts.append(book.subtitle)
    if book.authors:
        parts.append("Authors: " + ", ".join(book.authors))
    if book.description:
        parts.append("Description: " + book.description)
    if book.subjects:
        parts.append("Subjects: " + ", ".join(book.subjects))
    if reason:
        parts.append("Reason: " + reason)
    return "\n".join(parts)


def create_to_read_document(book: Book, reason: str) -> Document:
    """Create a LangChain Document for to-read list entries."""
    metadata = {
        "title": book.title,
        "authors": book.authors,
        "subjects": book.subjects,
        "description": book.description,
        "isbn13": book.isbn13,
        "isbn10": book.isbn10,
        "google_id": book.google_id,
        "openlib_key": book.openlib_key,
        "openlib_edition_key": book.openlib_edition_key,
        "published_year": book.published_year,
        "published_date": book.published_date,
        "page_count": book.page_count,
        "google_average_rating": book.google_average_rating,
        "google_ratings_count": book.google_ratings_count,
        "openlib_average_rating": book.openlib_average_rating,
        "openlib_ratings_count": book.openlib_ratings_count,
        "url": book.url,
        "reason": reason,
    }
    return Document(page_content=book_to_to_read_text(book, reason), metadata=metadata)


def get_to_read_store(
    *,
    persist_directory: str = TO_READ_PERSIST_DIR,
    collection_name: str = TO_READ_COLLECTION,
) -> Chroma:
    """Get or create the Chroma vector store for the to-read list."""
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDING,
        persist_directory=persist_directory,
    )


def _persist_store(store: Chroma) -> None:
    persist = getattr(cast(object, store), "persist", None)
    if callable(persist):
        persist()


def add_to_read_list(books_with_reasons: Iterable[tuple[Book, str]]) -> int:
    """Add books with reasons to the to-read list collection."""
    documents = [
        create_to_read_document(book, reason) for book, reason in books_with_reasons
    ]
    if not documents:
        return 0
    store = get_to_read_store()
    store.add_documents(documents)
    _persist_store(store)
    return len(documents)


def search_to_read_list(query: str, k: int = 5) -> list[tuple[Document, float]]:
    """Search the to-read list with semantic similarity."""
    store = get_to_read_store()
    return store.similarity_search_with_score(query, k=k)


def to_read_list_exists(*, persist_directory: str = TO_READ_PERSIST_DIR) -> bool:
    if not os.path.isdir(persist_directory):
        return False
    with os.scandir(persist_directory) as entries:
        return any(entry.is_file() or entry.is_dir() for entry in entries)


def _keys_from_metadata(metadata: dict) -> set:
    keys = set()
    for field in ["isbn13", "isbn10", "google_id", "openlib_key"]:
        value = metadata.get(field)
        if value:
            keys.add(str(value))
    title = metadata.get("title") or ""
    authors = metadata.get("authors") or []
    if isinstance(authors, list):
        authors_text = ",".join(authors)
    else:
        authors_text = str(authors)
    keys.add(f"{title}::{authors_text}")
    return keys


def get_all_to_read_keys(to_read_store: Optional[Chroma] = None) -> set[str]:
    """Get all known keys from the to-read list collection."""
    if to_read_store is None:
        if not to_read_list_exists():
            return set()
        to_read_store = get_to_read_store()

    results = to_read_store.get()
    if not results or "metadatas" not in results:
        return set()

    keys = set()
    for metadata in results["metadatas"]:
        if metadata:
            keys.update(_keys_from_metadata(metadata))
    return keys
