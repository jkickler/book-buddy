"""Utility functions for working with books in the graph tools."""

import re
from typing import List, Optional

from langchain_chroma import Chroma
from loguru import logger

from src.core.book import Book
from src.vectorstore.chroma import book_to_text, similarity_search_books
from src.vectorstore.to_read_list import get_all_to_read_keys


def normalize_text(value: str | None) -> str:
    """Normalize text by stripping whitespace and handling None."""
    return (value or "").strip()


def generate_key_from_book(book: Book) -> str:
    """Generate a unique key for a Book instance.

    If no key is found, returns a string with the title and authors.
    """
    for key in [
        book.isbn13,
        book.isbn10,
        book.google_id,
        book.openlib_key,
    ]:
        if key:
            return str(key)
    authors = ",".join(book.authors or [])
    return f"{book.title}::{authors}"


def generate_all_keys_from_book(book: Book) -> set:
    """Generate ALL possible keys for a Book instance as a set."""
    keys = set()
    for key in [
        book.isbn13,
        book.isbn10,
        book.google_id,
        book.openlib_key,
    ]:
        if key:
            keys.add(str(key))
    authors = ",".join(book.authors or [])
    keys.add(f"{book.title}::{authors}")
    return keys


def generate_key_from_doc(metadata: dict) -> str:
    """Generate a unique key from metadata from a document in the vector store.

    If no key is found, returns a string with the title and authors.
    """
    for field in [
        "isbn13",
        "isbn10",
        "google_id",
        "openlib_key",
    ]:
        value = metadata.get(field)
        if value:
            return str(value)
    title = metadata.get("title") or ""
    authors = metadata.get("authors") or []
    if isinstance(authors, list):
        authors_text = ",".join(authors)
    else:
        authors_text = str(authors)
    return f"{title}::{authors_text}"


def generate_all_keys_from_doc(metadata: dict) -> set:
    """Generate ALL possible keys from metadata as a set."""
    keys = set()
    for field in [
        "isbn13",
        "isbn10",
        "google_id",
        "openlib_key",
    ]:
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


def normalized_tokens(value: str | None) -> set[str]:
    """Extract normalized alphanumeric tokens from text."""
    words = re.findall(r"[a-z0-9]+", (value or "").lower())
    return set(words)


def extract_title_author_tokens(book: Book) -> tuple[set[str], set[str]]:
    """Extract normalized title and author tokens from a Book."""
    title_tokens = normalized_tokens(book.title)
    authors = book.authors or []
    author_tokens = normalized_tokens(" ".join(authors)) if authors else set()
    return title_tokens, author_tokens


def matches_already_read(
    metadata: dict,
    candidate_keys: set,
    candidate_title_tokens: set[str],
    candidate_author_tokens: set[str],
) -> tuple[bool, set[str], set[str]]:
    """Check if vector store metadata matches a candidate book.

    Checks if ANY of the candidate's keys match the metadata keys.
    Also checks if key words from candidate title/author appear in metadata.

    Args:
        metadata: Metadata dict from a vector store document.
        candidate_keys: Set of keys for the candidate book.
        candidate_patterns: List of compiled regex patterns for the candidate.

    Returns:
        True if the metadata matches the candidate book, False otherwise.
    """
    user_keys = generate_all_keys_from_doc(metadata)
    if candidate_keys & user_keys:
        return True, set(), set()

    title = metadata.get("title") or ""
    authors = metadata.get("authors") or []
    if not isinstance(authors, list):
        authors = [str(authors)]
    title_tokens = normalized_tokens(title)
    author_tokens = normalized_tokens(" ".join(authors)) if authors else set()

    if not candidate_title_tokens:
        return False, set(), set()

    if not title_tokens:
        return False, set(), set()

    title_match = candidate_title_tokens.issubset(
        title_tokens
    ) or title_tokens.issubset(candidate_title_tokens)
    if not title_match:
        return False, set(), set()

    if candidate_author_tokens and author_tokens:
        if not (candidate_author_tokens & author_tokens):
            return False, set(), set()

    matched_title_tokens = candidate_title_tokens & title_tokens
    matched_author_tokens = candidate_author_tokens & author_tokens
    return True, matched_title_tokens, matched_author_tokens

    return False


def book_summary(book: Book) -> dict:
    """Create a summary dictionary for a Book instance.

    Dict includes title, authors, description, subjects, isbn13, isbn10, published_year, url.
    """
    return {
        "title": book.title,
        "authors": book.authors,
        "description": book.description,
        "subjects": book.subjects,
        "isbn13": book.isbn13,
        "isbn10": book.isbn10,
        "published_year": book.published_year,
        "url": book.url,
    }


def filter_already_read_books(
    candidates: List[Book],
    vector_store: Chroma,
    to_read_store: Optional[Chroma] = None,
) -> List[Book]:
    """Filter out books that the user has already read.

    Compares candidate books against user's library using:
    1. Exact key match (ISBN, Google ID, OpenLibrary key)
    2. Fuzzy text matching on title + author combinations

    Args:
        candidates: List of Book candidates to filter.
        vector_store: Chroma vector store containing user's book library.
        to_read_store: Optional Chroma store containing the to-read list.

    Returns:
        Filtered list of candidates with already-read books removed.
    """
    user_books = vector_store.get()
    if not user_books or "metadatas" not in user_books:
        return candidates

    already_read_keys = set()
    user_metadatas = []
    for metadata in user_books["metadatas"]:
        if metadata:
            for key in generate_all_keys_from_doc(metadata):
                already_read_keys.add(key)
            user_metadatas.append(metadata)

    to_read_keys = set()
    to_read_metadatas = []
    if to_read_store is not None:
        to_read_keys = get_all_to_read_keys(to_read_store)
        to_read_results = to_read_store.get()
        if to_read_results and "metadatas" in to_read_results:
            for metadata in to_read_results["metadatas"]:
                if metadata:
                    to_read_metadatas.append(metadata)

    filtered = []
    filtered_count = 0

    for book in candidates:
        book_keys = generate_all_keys_from_book(book)

        matched_key = book_keys & already_read_keys

        if matched_key:
            filtered_count += 1
            continue

        matched_to_read_key = book_keys & to_read_keys
        if matched_to_read_key:
            filtered_count += 1
            continue

        is_already_read = False
        candidate_title_tokens, candidate_author_tokens = extract_title_author_tokens(
            book
        )
        for metadata in user_metadatas:
            matched, title_tokens, author_tokens = matches_already_read(
                metadata,
                book_keys,
                candidate_title_tokens,
                candidate_author_tokens,
            )
            if matched:
                is_already_read = True
                break

        if not is_already_read and to_read_metadatas:
            for metadata in to_read_metadatas:
                matched, title_tokens, author_tokens = matches_already_read(
                    metadata,
                    book_keys,
                    candidate_title_tokens,
                    candidate_author_tokens,
                )
                if matched:
                    is_already_read = True
                    break

        if is_already_read:
            filtered_count += 1
            continue

        filtered.append(book)

    logger.info(f"FILTER: {filtered_count} filtered, {len(filtered)} remaining")
    return filtered


def score_books_against_library(
    books: List[Book],
    vector_store: Chroma,
) -> dict:
    """Score books against the user's library using similarity search.

    Performs similarity search for each book and aggregates scores to find best matches.

    Args:
        books: List of Book instances to score.
        vector_store: Chroma vector store containing user's library.

    Returns:
        Dictionary with:
        - "best_overall": List of best matches across all candidates, sorted by score
        - "by_candidate": List of per-book scoring results with matches and summaries. Includes:
            - "source_key": Key of the book being scored
            - "candidate": Summary of the book being scored
            - "matches": List of best matches for the book, sorted by score, with metadata and summary
            - "scores": List of scores for the book's matches

    """
    best_scores = {}
    per_candidate = []

    for book in books:
        source_key = generate_key_from_book(book)
        candidate = book_summary(book)

        try:
            results = similarity_search_books(
                query=book_to_text(book),
                vector_store=vector_store,
                k=10,
            )
        except Exception:
            continue

        matches = []
        for doc, score in results:
            key = generate_key_from_doc(doc.metadata)
            match = {
                "score": score,
                "metadata": doc.metadata,
            }
            matches.append(match)

            current = best_scores.get(key)
            if current is None or score < current["score"]:
                best_scores[key] = {
                    "score": score,
                    "metadata": doc.metadata,
                    "matched_from": source_key,
                }

        scores = [item["score"] for item in matches]
        score_summary = {
            "min": min(scores) if scores else None,
            "avg": sum(scores) / len(scores) if scores else None,
            "count": len(scores),
        }

        per_candidate.append(
            {
                "candidate": candidate,
                "source_key": source_key,
                "score_summary": score_summary,
                "matches": sorted(matches, key=lambda item: item["score"]),
            }
        )

    scores = sorted(best_scores.values(), key=lambda item: item["score"])

    return {
        "best_overall": scores,
        "by_candidate": per_candidate,
    }
