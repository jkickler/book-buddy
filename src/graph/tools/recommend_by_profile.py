"""Recommend by profile module for generating book recommendations based on user profile."""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool
from langchain_chroma.vectorstores import Chroma
from loguru import logger
from rapidfuzz import fuzz

from src.api.book_service import BookDataService
from src.api.client import OpenLibraryClient
from src.core.book import Book
from src.graph.tools.book_utils import (
    filter_already_read_books,
    generate_key_from_book,
    score_books_against_library,
)
from src.vectorstore.chroma import get_vector_store_cached
from src.vectorstore.to_read_list import get_to_read_store, to_read_list_exists

TOP_SUBJECTS = 3
MIN_CANDIDATES_PER_SUBJECT = 10
MAX_RECOMMENDATIONS = 10


def extract_subjects_from_store(vector_store: Chroma) -> Dict[str, List[float]]:
    """Extract all subjects from documents in the vector store with their associated Goodreads ratings.

    Args:
        vector_store: Chroma vector store

    Returns:
        Dictionary mapping subject (str) to list of ratings (float). Each book can contribute
        multiple subjects with their corresponding user rating. Books without ratings will have
        None in the list.
    """
    # Gets all documents from the vector store
    results = vector_store.get()
    subject_ratings = defaultdict(list)

    if not results or "metadatas" not in results:
        return dict(subject_ratings)

    metadatas = results["metadatas"]

    for metadata in metadatas:
        if not metadata:
            continue

        subjects = metadata.get("subjects", [])
        if not subjects:
            continue

        rating = metadata.get("goodreads_user_rating")
        for subject in subjects:
            if subject:
                subject_ratings[subject].append(rating)

    return dict(subject_ratings)


def calculate_weighted_subject_scores(
    subject_ratings: Dict[str, List[float]],
) -> Dict[str, float]:
    """Calculate weighted scores for subjects based on count and ratings.

    Uses a weighted scoring system where higher user ratings contribute more to the subject score.
    The weights are: 5-star = 1.0, 4-star = 0.6, 3-star = 0.2, 1-2 stars = 0.0.
    Books without ratings (None) contribute 0.0.

    Compute each subject's score as (count of each star rating) * (its weight), summed:
    score = (#5★)*1.0 + (#4★)*0.6 + (#3★)*0.2 + (#1-2★)*0.0

    Args:
        subject_ratings: Dictionary mapping subject to list of user ratings (floats or None).

    Returns:
        Dictionary mapping subject to weighted score (float). Higher scores indicate
        stronger user preference for that subject based on their highly-rated books.
    """
    weights = {5: 1.0, 4: 0.6, 3: 0.2, 2: 0.0, 1: 0.0}
    scores = {}

    for subject, ratings in subject_ratings.items():
        total_score = 0.0
        for rating in ratings:
            if rating is None:
                continue
            weight = weights.get(int(rating), 0.0)
            total_score += weight

        scores[subject] = total_score
    return scores


def get_top_subjects(
    subject_scores: Dict[str, float],
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """Get the top N subjects sorted by weighted score.

    Returns the subjects with the highest weighted scores, which represent the user's
    strongest reading preferences based on their highly-rated books.
    """
    sorted_subjects = sorted(
        subject_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return sorted_subjects[:top_n]


def expand_candidates_from_related_subjects(
    client: OpenLibraryClient,
    subject: str,
    existing_candidates: List[Dict[str, Any]],
    min_candidates: int = MIN_CANDIDATES_PER_SUBJECT,
) -> List[Dict[str, Any]]:
    """Expand candidate search by querying related subjects to increase candidate diversity.

    Args:
        client: OpenLibraryClient instance
        subject: Original subject slug
        existing_candidates: Current list of candidate works.
        min_candidates: Minimum number of candidates desired (default from MIN_CANDIDATES_PER_SUBJECT).

    Returns:
        Combined list of existing candidates plus new candidates from related subjects.
    """
    _, related_subjects = client.fetch_books_by_subjects(
        subject, related_subs_flag=True
    )

    all_candidates = existing_candidates.copy()
    seen_keys = {c["key"] for c in existing_candidates if c.get("key")}

    for related_subject in related_subjects:
        if len(all_candidates) >= min_candidates:
            break

        try:
            related_works, _ = client.fetch_books_by_subjects(
                related_subject, related_subs_flag=False
            )
            for work in related_works:
                key = work.get("key")
                if key and key not in seen_keys:
                    all_candidates.append(work)
                    seen_keys.add(key)
        except Exception:
            continue

    return all_candidates


def deduplicate_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deduplicate candidates by OpenLibrary work key, because when fetching candidates
    from multiple subjects, the same book may appear multiple times.
    If the same key appears multiple times, keeps the entry with the most complete metadata
    (prefers entries with title, authors, and isbn).

    Scoring is based on presence of title, authors, and isbn fields (higher count wins).

    Args:
        candidates: List of candidate work dictionaries from OpenLibrary.

    Returns:
        Deduplicated list of candidates with unique work keys.
    """
    seen_keys = {}

    for candidate in candidates:
        key = candidate.get("key")
        if not key:
            continue

        if key in seen_keys:
            existing = seen_keys[key]
            # Calculate completeness score: count of present title, authors, isbn fields
            existing_score = sum(
                1 for field in ["title", "authors", "isbn"] if existing.get(field)
            )
            new_score = sum(
                1 for field in ["title", "authors", "isbn"] if candidate.get(field)
            )
            if new_score > existing_score:
                seen_keys[key] = candidate
        else:
            seen_keys[key] = candidate

    return list(seen_keys.values())


def enrich_candidate(
    service: BookDataService,
    candidate: Dict[str, Any],
) -> Optional[Book]:
    """Enrich a candidate with metadata from Google Books API.

    Takes a candidate work from OpenLibrary and fetches additional metadata from
    Google Books API to get more complete book information including description,
    ratings, page count, etc. First attempts to find the book by ISBN, then falls
    back to title/author search with fuzzy matching to find the best match.

    Args:
        service: BookDataService instance for API interactions.
        candidate: Candidate work dictionary with key, title, authors, and optionally isbn.

    Returns:
        Enriched Book object with Google Books metadata, or None if enrichment fails.
    """
    title = candidate.get("title", "")
    authors = candidate.get("authors", [])
    author = authors[0] if authors else ""

    isbns = candidate.get("isbn", [])
    for isbn in isbns:
        try:
            books = service.google_client.fetch_books(
                title="",
                author="",
                isbn=isbn,
            )
            if books:
                book = books[0]
                if not book.openlib_key:
                    book.openlib_key = candidate.get("key")
                return book
        except Exception:
            continue

    try:
        books = service.google_client.fetch_books(
            title=title,
            author=author,
            max_results=5,
        )
        if books:
            best_book = None
            best_score = 0.0
            for book in books:
                title_score = (
                    fuzz.token_set_ratio(title.lower(), book.title.lower()) / 100.0
                )
                author_score = 0.0
                if book.authors and authors:
                    author_score = (
                        fuzz.token_set_ratio(
                            authors[0].lower(), book.authors[0].lower()
                        )
                        / 100.0
                    )
                score = 0.75 * title_score + 0.25 * author_score
                if score > best_score:
                    best_score = score
                    best_book = book

            if best_book and best_score >= 0.7:
                if not best_book.openlib_key:
                    best_book.openlib_key = candidate.get("key")
                return best_book
    except Exception:
        pass

    return None


def generate_recommendations(
    vector_store: Optional[Chroma] = None,
    top_n_subjects: int = TOP_SUBJECTS,
    min_candidates: int = MIN_CANDIDATES_PER_SUBJECT,
    max_recommendations: int = MAX_RECOMMENDATIONS,
) -> Dict[str, Any]:
    """Generate book recommendations based on user's reading profile.

    Main orchestration function that generates personalized book recommendations by:
    1. Extracting subjects from user's library
    2. Calculating weighted subject scores and selecting top subjects
    3. Fetching candidate books from OpenLibrary for top subjects
    4. Deduplicating candidates
    5. Enriching candidates with Google Books metadata
    6. Filtering out already-read books
    7. Scoring candidates by similarity to user's library

    Args:
        vector_store: User's book vector store (uses cached store if None).
        top_n_subjects: Number of top subjects to use for candidate generation (default 3).
        min_candidates: Minimum candidates to fetch per subject (default 10).
        max_recommendations: Maximum recommendations to return (default 5).

    Returns:
        Dictionary with:
        - status: "ok" or "error"
        - seed_subjects: List of top subjects with scores
        - candidates: List of recommended books with metadata and similarity scores
        - message: Error message if status is "error"
    """
    logger.info("RECOMMEND: START")

    if vector_store is None:
        vector_store = get_vector_store_cached()

    # Phase 1: Extract subjects
    subject_ratings = extract_subjects_from_store(vector_store)
    if not subject_ratings:
        logger.error("RECOMMEND: FAILED - No subjects found in library")
        return {
            "status": "error",
            "message": "No subjects found in your library. Please ensure your books have subject metadata.",
            "seed_subjects": [],
            "candidates": [],
        }

    # Phase 2: Calculate scores and get top subjects
    subject_scores = calculate_weighted_subject_scores(subject_ratings)
    top_subjects = get_top_subjects(subject_scores, top_n=top_n_subjects)
    seed_subjects = [sub for sub, _ in top_subjects]

    if not top_subjects:
        logger.error("RECOMMEND: FAILED - Could not determine top genres")
        return {
            "status": "error",
            "message": "Could not determine your top genres. Please rate more books.",
            "seed_subjects": [],
            "candidates": [],
        }

    logger.info(f"RECOMMEND: Subjects: {[(s, round(sc, 2)) for s, sc in top_subjects]}")

    # Phase 3: Fetch candidates from OpenLibrary
    service = BookDataService()

    all_candidates = []

    for subject, score in top_subjects:
        try:
            candidates, _ = service.openlib_client.fetch_books_by_subjects(
                subject=subject
            )

            if len(candidates) < min_candidates:
                candidates = expand_candidates_from_related_subjects(
                    service.openlib_client, subject, candidates, min_candidates
                )

            for c in candidates:
                c["source_subject"] = subject

            all_candidates.extend(candidates)
        except Exception:
            continue

    if not all_candidates:
        logger.error("RECOMMEND: FAILED - No candidates fetched from OpenLibrary")
        return {
            "status": "error",
            "message": "Could not fetch candidates from OpenLibrary.",
            "seed_subjects": seed_subjects,
            "candidates": [],
        }

    # Phase 4: Deduplicate candidates
    all_candidates = deduplicate_candidates(all_candidates)

    # Phase 5: Enrich candidates with Google Books data
    enriched_candidates: List[Book] = []
    enrich_success = 0
    enrich_fail = 0

    for candidate in all_candidates:
        try:
            book = enrich_candidate(service, candidate)
            if book:
                book.openlib_key = candidate.get("key")
                enriched_candidates.append(book)
                enrich_success += 1
            else:
                enrich_fail += 1
        except Exception:
            enrich_fail += 1
            continue

    if not enriched_candidates:
        logger.error("RECOMMEND: FAILED - Could not enrich candidates")
        return {
            "status": "error",
            "message": "Could not enrich candidates with book metadata.",
            "seed_subjects": seed_subjects,
            "candidates": [],
        }

    # Phase 6: Filter out already-read books
    enriched_candidates_before_filter = len(enriched_candidates)
    to_read_store = None
    if to_read_list_exists():
        to_read_store = get_to_read_store()
    enriched_candidates = filter_already_read_books(
        enriched_candidates,
        vector_store,
        to_read_store=to_read_store,
    )
    filtered_out = enriched_candidates_before_filter - len(enriched_candidates)
    logger.info(f"RECOMMEND: Filtered {filtered_out} already-read books")

    if not enriched_candidates:
        logger.error("RECOMMEND: FAILED - All candidates already read")
        return {
            "status": "error",
            "message": "All candidates appear to be books you've already read.",
            "seed_subjects": seed_subjects,
            "candidates": [],
        }

    # Phase 7: Score candidates by similarity
    score_result = score_books_against_library(enriched_candidates, vector_store)

    # Map similarity scores back to Book objects. For each scored candidate,
    # find the matching book and attach the top 3 library matches that explain
    # the similarity. Sort candidates by best similarity score (lower = more similar).
    scored_candidates = []
    for item in score_result["by_candidate"]:
        book = next(
            (
                book
                for book in enriched_candidates
                if generate_key_from_book(book) == item["source_key"]
            ),
            None,
        )
        if book:
            score_data = item["score_summary"]
            score_data["matches"] = item["matches"][:3]
            scored_candidates.append((book, score_data))
    scored_candidates.sort(key=lambda x: x[1]["min"])

    candidate_list = []
    # Gets top 10 scored_candidates
    for book, score_data in scored_candidates[:max_recommendations]:
        candidate_list.append(
            {
                "title": book.title,
                "authors": book.authors,
                "description": book.description,
                "url": book.url,
                "subjects": book.subjects,
                "isbn13": book.isbn13,
                "isbn10": book.isbn10,
                "published_year": book.published_year,
                "google_average_rating": book.google_average_rating,
                "openlib_key": book.openlib_key,
                "similarity_score": score_data,
            }
        )

    logger.info(f"RECOMMEND: Result - {len(candidate_list)} recommendations")
    return {
        "status": "ok",
        "seed_subjects": [
            {"subject": sub, "score": round(score, 2)} for sub, score in top_subjects
        ],
        "candidates": candidate_list,
    }


@tool("recommend_by_profile")
def recommend_by_profile_tool() -> str:
    """Generate book recommendations based on the user's reading profile and taste.

    Use this tool when the user asks for general recommendations like:
    - 'Recommend books for me'
    - 'What books fit my taste?'
    - 'Suggest books based on what I've read'
    - 'What should I read next?'
    This tool analyzes the user's library, finds top genres/subjects, fetches candidates from OpenLibrary, scores them, and returns a ranked list.
    """
    result = generate_recommendations()
    return json.dumps(result)
