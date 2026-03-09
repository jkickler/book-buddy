# Chroma vector store for book embeddings with similarity search
import os
from typing import Iterable, Optional, cast

from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from ..core.book import Book
from .state import get_cached_vector_store, set_cached_vector_store

DEFAULT_COLLECTION_NAME = "books"
DEFAULT_PERSIST_DIRECTORY = "data/chroma"
EMBEDDING = OpenAIEmbeddings(model="text-embedding-3-large")


def book_to_text(book: Book) -> str:
    """Serialize a Book object into a single text block for embedding."""
    parts = [book.title]
    if book.subtitle:
        parts.append(book.subtitle)
    if book.authors:
        parts.append("Authors: " + ", ".join(book.authors))
    if book.description:
        parts.append("Description: " + book.description)
    if book.subjects:
        parts.append("Subjects: " + ", ".join(book.subjects))
    if book.published_year:
        parts.append(f"Published year: {book.published_year}")
    return "\n".join(parts)


def books_to_documents(
    books: Iterable[Book],
    extra_metadata: Optional[list[dict]] = None,
) -> list[Document]:
    """Convert books to LangChain Documents with metadata.

    Args:
        books: An iterable of Book instances to convert.
        extra_metadata: Optional list of additional metadata dictionaries for each book.

    Returns:
        A list of Document objects with book content and metadata.
    """
    documents: list[Document] = []
    if extra_metadata is not None:
        extra_items = list(extra_metadata)
        book_items = list(books)
        if len(extra_items) != len(book_items):
            raise ValueError("extra_metadata length must match books length")
        items = zip(book_items, extra_items)
    else:
        items = ((book, {}) for book in books)

    for book, extra in items:
        metadata = {
            "title": book.title,
            "authors": book.authors,
            "subjects": book.subjects,
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
        }
        metadata.update(extra)
        documents.append(
            Document(
                page_content=book_to_text(book),
                metadata=metadata,
            )
        )
    return documents


def create_book_vector_store(
    books: Iterable[Book],
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None,
    extra_metadata: Optional[list[dict]] = None,
) -> Chroma:
    """Create a Chroma vector store from books.

    Args:
        books: An iterable of Book instances to add to the vector store.
        collection_name: The name of the Chroma collection.
        persist_directory: Directory to persist the vector store; if None, in-memory.
        extra_metadata: Optional list of additional metadata for each book.

    Returns:
        A Chroma vector store instance.
    """
    logger.info("START")
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
    documents = books_to_documents(books, extra_metadata=extra_metadata)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=EMBEDDING,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    if persist_directory:
        persist = getattr(cast(object, vector_store), "persist", None)
        if callable(persist):
            persist()
    logger.info(f"Created vector store with {len(documents)} documents")
    return vector_store


def load_book_vector_store(
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """Load an existing Chroma vector store for books."""
    logger.info("START")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDING,
        persist_directory=persist_directory,
    )
    count = vector_store._collection.count()
    logger.info(f"Loaded vector store with {count} documents")
    return vector_store


def get_vector_store_cached(
    *,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Get cached vector store or load from disk."""
    cached = get_cached_vector_store()
    if cached is None:
        cached = load_book_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        set_cached_vector_store(cached)
    return cached


def vector_store_exists(
    *,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
) -> bool:
    if not os.path.isdir(persist_directory):
        return False
    with os.scandir(persist_directory) as entries:
        return any(entry.is_file() or entry.is_dir() for entry in entries)


def similarity_search_books(
    query: str,
    vector_store: Chroma,
    *,
    k: int = 5,
    metadata_filter: Optional[dict] = None,
) -> list[tuple[Document, float]]:
    """Run similarity search with vector store with distance.
    Lower score represents more similarity.

    Args:
        query: The search query string.
        vector_store: The Chroma vector store to search in.
        k: Number of top similar documents to return.
        metadata_filter: Optional filter for metadata.

    Returns:
        A list of tuples, each containing a Document and its similarity score.
        Lower score represents more similarity.
    """
    logger.info(f"START search='{query}', k={k}")
    results = vector_store.similarity_search_with_score(
        query, k=k, filter=metadata_filter
    )
    logger.info(f"Search returned {len(results)} results")
    return results
