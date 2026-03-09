# Tool for saving book recommendations to user's to-read list with fuzzy matching
import json
from typing import Annotated, Optional

from langchain.tools import tool
from langgraph.prebuilt import InjectedState
from rapidfuzz import fuzz

from src.core.book import Book
from src.vectorstore.to_read_list import add_to_read_list

MATCH_THRESHOLD = 70
REASON_TEMPLATE = "Recommended because it matches your taste in: {titles}"
REASON_FALLBACK = "Recommended because it matches your taste."


def _normalize(value: str) -> str:
    """Normalize a string by stripping and lowercasing."""
    return value.strip().lower()


def _authors_text(authors: Optional[list[str]]) -> str:
    """Convert list of authors to a space-separated string."""
    if not authors:
        return ""
    return " ".join(authors)


def _listify(value) -> list:
    """Ensure value is a list, wrapping if necessary."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def match_user_title_to_item(
    user_title: str,
    items: list[dict],
    *,
    title_key: str = "title",
    authors_key: str = "authors",
) -> tuple[Optional[dict], float]:
    """
    Match user-provided title to items using fuzzy matching on title and authors.

    Args:
        user_title (str): The title provided by the user.
        items (list[dict]): List of item dictionaries to match against.
        title_key (str, optional): Key for title in item dict. Defaults to "title".
        authors_key (str, optional): Key for authors in item dict. Defaults to "authors".

    Returns:
        tuple[Optional[dict], float]: Best matching item and its score, or (None, 0.0).
    """
    best_score = 0.0
    best_match = None
    user_value = _normalize(user_title)

    for item in items:
        title = _normalize(str(item.get(title_key) or ""))
        if not title:
            continue
        author_value = _normalize(_authors_text(item.get(authors_key)))
        title_score = float(fuzz.token_set_ratio(user_value, title))
        author_score = float(fuzz.token_set_ratio(user_value, author_value))
        score = 0.75 * title_score + 0.25 * author_score
        if score > best_score:
            best_score = score
            best_match = item

    return best_match, best_score


def normalize_checked_items(checked_books: list[dict]) -> list[dict]:
    """Normalize checked books into enriched book format."""
    normalized = []
    for item in checked_books:
        enriched = item.get("enriched_book") or {}
        if not enriched:
            continue
        entry = dict(enriched)
        entry["_source"] = item
        normalized.append(entry)
    return normalized


def reason_from_matches(matches: list[dict]) -> str:
    """Generate recommendation reason from similarity matches."""
    match_titles = []
    for match in matches:
        metadata = match.get("metadata") or {}
        title = metadata.get("title")
        if title:
            match_titles.append(str(title))
        if len(match_titles) == 3:
            break
    if match_titles:
        joined = ", ".join(match_titles)
        return REASON_TEMPLATE.format(titles=joined)
    return REASON_FALLBACK


def reason_from_recommendation(candidate: dict) -> str:
    """Extract reason from a recommendation candidate."""
    similarity = candidate.get("similarity_score") or {}
    matches = similarity.get("matches") or []
    return reason_from_matches(matches)


def book_from_payload(payload: dict) -> Book:
    """Create a Book object from a given dictionary."""
    authors = [str(item) for item in _listify(payload.get("authors"))]
    subjects = [str(item) for item in _listify(payload.get("subjects"))]
    return Book(
        google_id=payload.get("google_id"),
        openlib_key=payload.get("openlib_key"),
        title=str(payload.get("title") or ""),
        subtitle=payload.get("subtitle"),
        authors=authors,
        subjects=subjects,
        description=payload.get("description"),
        isbn13=payload.get("isbn13"),
        isbn10=payload.get("isbn10"),
        published_date=payload.get("published_date"),
        published_year=payload.get("published_year"),
        url=payload.get("url"),
        page_count=payload.get("page_count"),
        google_average_rating=payload.get("google_average_rating"),
        google_ratings_count=payload.get("google_ratings_count"),
        openlib_average_rating=payload.get("openlib_average_rating"),
        openlib_ratings_count=payload.get("openlib_ratings_count"),
        openlib_edition_key=payload.get("openlib_edition_key"),
    )


def reason_from_checked_book(item: dict) -> str:
    """Generate reason from a checked book item."""
    similarity = item.get("similarity_scores") or {}
    by_candidate = similarity.get("by_candidate") or []
    matches = by_candidate[0].get("matches", []) if by_candidate else []
    return reason_from_matches(matches)


def match_and_build(
    user_title: str,
    items: list[dict],
    *,
    reason_from_item,
) -> Optional[tuple[Book, str]]:
    """
    Match user title to items and build Book with reason, if the threshold is met.

    Args:
        user_title (str): The title to match.
        items (list[dict]): Items to search in.
        reason_from_item: Callable to extract reason from matched item.

    Returns:
        Optional[tuple[Book, str]]: Book and reason if matched, else None.
    """
    match, score = match_user_title_to_item(user_title, items)
    if match and score >= MATCH_THRESHOLD:
        return book_from_payload(match), reason_from_item(match)
    return None


@tool("save_to_read_list")
def save_to_read_list_tool(
    titles: list[str],
    last_recommendations: Annotated[
        Optional[list[dict]], InjectedState("last_recommendations")
    ],
    last_checked_books: Annotated[
        Optional[list[dict]], InjectedState("last_checked_books")
    ],
) -> str:
    """
    Save recommended books to the user's to-read list using recent recommendations or checked books.

    Use this tool after calling recommend_by_profile or enrich_and_score when the user wants to save one or more recommendations.
    The user may provide partial or approximate titles; the tool will match them against recent recommendations
    or checked books using fuzzy matching. For each match, a reason is generated explaining why it fits their taste.
    """
    if not last_recommendations and not last_checked_books:
        return json.dumps(
            {
                "status": "error",
                "message": "No recent recommendations or checked books available.",
                "saved": 0,
                "errors": ["no_recommendations_or_checked_books"],
            }
        )

    if not titles:
        return json.dumps(
            {
                "status": "error",
                "message": "No titles provided.",
                "saved": 0,
                "errors": ["no_titles"],
            }
        )

    books_with_reasons = []
    errors = []

    for title in titles:
        # Try to match the title against recent recommendations from the recommend_by_profile tool
        if last_recommendations:
            result = match_and_build(
                user_title=title,
                items=last_recommendations,
                reason_from_item=reason_from_recommendation,
            )
            if result:
                books_with_reasons.append(result)
                continue

        # If not found in recommendations, try against recent checked books from the enrich_and_score tool
        if last_checked_books:
            checked_items = normalize_checked_items(last_checked_books)
            result = match_and_build(
                user_title=title,
                items=checked_items,
                reason_from_item=lambda item: reason_from_checked_book(
                    item.get("_source") or {}
                ),
            )
            if result:
                books_with_reasons.append(result)
                continue

        errors.append({"title": title, "error": "no_match"})

    saved = add_to_read_list(books_with_reasons)
    status = "ok" if saved else "error"
    return json.dumps(
        {
            "status": status,
            "saved": saved,
            "errors": errors,
        }
    )
