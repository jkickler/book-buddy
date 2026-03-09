# CSV ingestion for Goodreads library export with book enrichment
import os
import re
import shutil
from typing import Callable, Iterable, Optional

import pandas as pd
from loguru import logger

from ..api.book_service import BookDataService
from ..core.book import Book
from .chroma import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    create_book_vector_store,
)
from .state import clear_cached_vector_store

ProgressCallback = Callable[[int, int, str], None]


def _find_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    """Find a matching column name from candidates."""
    lower_map = {name.lower(): name for name in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    for name in columns:
        lower_name = name.lower()
        if any(candidate in lower_name for candidate in candidates):
            return name
    return None


def _row_value(row, column: Optional[str]) -> str:
    """Extract and clean a value from a CSV row."""
    if not column:
        return ""
    value = row.get(column)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _normalize_isbn(value: str) -> str:
    """Normalize an ISBN string by removing non-digits."""
    if not value:
        return ""
    cleaned = value.strip()
    if cleaned.startswith("="):
        cleaned = cleaned.lstrip("=")
    cleaned = cleaned.strip().strip('"')
    digits = re.sub(r"[^0-9Xx]", "", cleaned)
    return digits


def _to_float(value: str) -> Optional[float]:
    """Convert a string to float, returning None on failure."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str) -> Optional[int]:
    """Convert a string to int, returning None on failure."""
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_csv(file_obj) -> list[dict[str, object]]:
    """Parse a CSV file into a list of book dictionaries."""
    df = pd.read_csv(file_obj)
    exclusive_shelf_col = _find_column(df.columns, ["exclusive shelf"])
    if exclusive_shelf_col:
        df = df.sort_values(by=exclusive_shelf_col, key=lambda x: x != "read")
    title_col = _find_column(df.columns, ["title"])
    author_col = _find_column(df.columns, ["author"])
    isbn_col = _find_column(df.columns, ["isbn13", "isbn"])
    my_rating_col = _find_column(df.columns, ["my rating"])
    user_rating_col = _find_column(df.columns, ["user rating"])
    year_col = _find_column(df.columns, ["year published"])

    if not title_col or not author_col:
        raise ValueError("CSV must include title and author columns.")

    rows = []
    for _, row in df.iterrows():
        title = _row_value(row, title_col)
        author = _row_value(row, author_col)
        isbn = _normalize_isbn(_row_value(row, isbn_col))
        my_rating = _to_float(_row_value(row, my_rating_col))
        user_rating = _to_float(_row_value(row, user_rating_col))
        year_published = _to_int(_row_value(row, year_col))
        rows.append(
            {
                "title": title,
                "author": author,
                "isbn13": isbn or None,
                "goodreads_user_rating": my_rating,
                "goodreads_community_rating": user_rating,
                "published_year": year_published,
            }
        )
    return rows


def fill_missing_data(book: Book, candidate: Book) -> None:
    """Fill missing fields in a book using data from a Google Books metadata.

    Args:
        book: The Book instance to update with missing data.
        candidate: The candidate Book from Google Books API.
    """
    string_fields = [
        "subtitle",
        "description",
        "isbn13",
        "isbn10",
        "published_date",
        "url",
        "google_id",
    ]
    for field in string_fields:
        if not getattr(book, field) and getattr(candidate, field):
            setattr(book, field, getattr(candidate, field))

    if not book.subjects and candidate.subjects:
        book.subjects = candidate.subjects

    if book.published_year is None and candidate.published_year is not None:
        book.published_year = candidate.published_year

    if book.page_count is None and candidate.page_count is not None:
        book.page_count = candidate.page_count

    if (
        book.google_average_rating is None
        and candidate.google_average_rating is not None
    ):
        book.google_average_rating = candidate.google_average_rating

    if book.google_ratings_count is None and candidate.google_ratings_count is not None:
        book.google_ratings_count = candidate.google_ratings_count


def ingest_csv(
    file_obj,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[object, int]:
    """Ingest CSV file, enrich books, and create a vector store.

    Args:
        file_obj: The CSV file object to parse.
        collection_name: Name of the Chroma collection to create.
        persist_directory: Directory path to persist the vector store.
        progress_callback: Optional callback function for progress updates.

    Returns:
        A tuple containing the created vector store and the total number of books ingested.
    """
    logger.info("START")
    # Clear cached vector store to prevent database lock issues
    clear_cached_vector_store()

    # Remove existing database files if they exist
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
        except Exception:
            pass

    # Load Goodreads CSV file
    rows = parse_csv(file_obj)
    total = len(rows)
    logger.info(f"Parsed {total} books from CSV")
    service = BookDataService()
    enriched_books = []
    extra_metadata = []

    # Create Book object for each row and add metadata from Google Books and
    # Openlibrary if available
    for index, row in enumerate(rows, start=1):
        title_value = row.get("title")
        author_value = row.get("author")
        isbn_value = row.get("isbn13")

        title = str(title_value or "")
        author = str(author_value or "")
        isbn = str(isbn_value or "")
        published_year = row.get("published_year")

        book = Book(
            google_id=None,
            openlib_key=None,
            title=title,
            subtitle=None,
            authors=[author] if author else [],
            subjects=[],
            description=None,
            isbn13=isbn or None,
            isbn10=None,
            published_date=None,
            published_year=published_year,
            url=None,
            page_count=None,
            google_average_rating=None,
            google_ratings_count=None,
            openlib_average_rating=None,
            openlib_ratings_count=None,
            openlib_edition_key=None,
        )

        google_books = service.google_client.fetch_books(
            title=title,
            author=author,
            isbn=isbn if isbn else None,
        )

        # Fill missing data from Google Books if available
        if google_books:
            fill_missing_data(book, google_books[0])

        # Enrich book with OpenLibrary data, primarly subjects and ratings
        book = service.enrich_book_data(
            book,
            book_title=title,
            book_author=author,
        )
        enriched_books.append(book)
        # Add Goodreads ratings as metadata from CSV
        extra_metadata.append(
            {
                "goodreads_user_rating": row.get("goodreads_user_rating"),
                "goodreads_community_rating": row.get("goodreads_community_rating"),
            }
        )

        # For progress updates in UI
        if progress_callback:
            progress_callback(index, total, title)

    vector_store = create_book_vector_store(
        enriched_books,
        collection_name=collection_name,
        persist_directory=persist_directory,
        extra_metadata=extra_metadata,
    )
    logger.info(f"Ingested {total} books")
    return vector_store, total
