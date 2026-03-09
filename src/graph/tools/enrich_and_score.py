# Tool for enriching books with metadata and scoring against user's library
import json

from langchain.tools import tool
from langchain_chroma import Chroma
from loguru import logger

from src.api.book_service import BookDataService
from src.graph.schemas import BookQuery, EnrichAndScoreInput
from src.graph.tools.book_utils import (
    book_summary,
    normalize_text,
    score_books_against_library,
)
from src.vectorstore.chroma import get_vector_store_cached


def _is_empty_query(book_query: BookQuery) -> bool:
    """Check if the book query is empty."""
    return not (book_query.title or book_query.author or book_query.isbn)


def fetch_single_book(book_query: BookQuery, service: BookDataService):
    """Fetch and enrich a single book based on the query."""
    title = normalize_text(book_query.title)
    author = normalize_text(book_query.author)
    isbn = normalize_text(book_query.isbn)

    try:
        # Try Google Books search (works with or without ISBN)
        google_books = service.google_client.fetch_books(
            title=title,
            author=author,
            isbn=isbn if isbn else None,
        )

        if google_books:
            book = google_books[0]
            return service.enrich_book_data(book)

        # Fallback: try title/author search via BookDataService (includes OpenLibrary)
        if title or author:
            book = service.search_books(title, author)
            if book:
                return book

    except Exception:
        pass

    raise ValueError("Missing title, author, or ISBN")


def book_query_to_dict(book_query: BookQuery) -> dict:
    """Convert BookQuery to a dictionary."""
    return book_query.model_dump()


def process_book_query(
    book_query: BookQuery,
    service: BookDataService,
    vector_store: Chroma,
) -> dict:
    """Process a single book query to fetch, enrich, and score against the user's library."""
    query_str = book_query.title or book_query.isbn or book_query.author or "empty"

    query = {"query": book_query_to_dict(book_query)}
    if _is_empty_query(book_query):
        return {
            **query,
            "status": "clarify",
            "message": "Please provide a title, author, or ISBN.",
        }

    try:
        book = fetch_single_book(book_query, service)
    except ValueError:
        return {
            **query,
            "status": "not_found",
            "message": "No confident match found.",
        }

    similarity_scores = score_books_against_library([book], vector_store)

    return {
        **query,
        "status": "ok",
        "enriched_book": book_summary(book),
        "similarity_scores": similarity_scores,
    }


@tool("enrich_and_score", args_schema=EnrichAndScoreInput)
def enrich_and_score_tool(
    books: list[BookQuery],
) -> str:
    """Enrich books and score similarity against the user's library.

    Use this tool when the user asks whether specific books (or multiple specific books) fit their taste/library.
    Provide structured inputs per book: title, author, and/or ISBN. Use ISBN when available.
    Do not call if not enough information to identify books.
    """
    logger.info(f"ENRICH: START - {len(books)} books")
    vector_store = get_vector_store_cached()
    service = BookDataService()

    results = []
    success_count = 0
    not_found_count = 0
    clarify_count = 0

    for book_query in books:
        result = process_book_query(book_query, service, vector_store)
        results.append(result)

        status = result.get("status", "unknown")
        if status == "ok":
            success_count += 1
        elif status == "not_found":
            not_found_count += 1
        elif status == "clarify":
            clarify_count += 1

    logger.info(f"ENRICH: Result - ok={success_count}, not_found={not_found_count}")

    payload = {"status": "ok", "results": results}
    return json.dumps(payload)
