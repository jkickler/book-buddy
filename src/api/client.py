# API clients for Google Books and OpenLibrary with retry logic
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from dateutil import parser
from rapidfuzz import fuzz

from src.core.book import Book

OPENLIB_BASE_URL = "https://openlibrary.org/search.json"
GOOGLE_BOOKS_BASE_URL = "https://www.googleapis.com/books/v1/volumes"
USER_AGENT = "book-recommendation-app/0.1"
HTTP_TIMEOUT_SECONDS = 15
HTTP_RETRIES = 3

OPENLIB_FIELDS = [
    "key",
    "title",
    "subtitle",
    "publish_year",
    "edition_count",
    "first_publish_year",
    ### ---All fields below are returned as arrays
    "publish_date",
    "isbn",
    "author_name",
    "subject",
    "language",
]


class BaseApiClient:
    """Base class providing HTTP session management and retry logic for API clients."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _get_metadata(self, base_url: str, params: Optional[dict] = None):
        """Fetch metadata from base URL with retry logic.

        Open Library = "https://openlibrary.org/search.json"
        Google Books = "https://www.googleapis.com/books/v1/volumes"

        Args:
            base_url: The base URL to fetch metadata from.
            params: Dict of parameters (title, author etc.)

        Returns:
            JSON with metadata.
        """
        api_name = "OpenLibrary" if "openlibrary" in base_url else "Google Books"
        for attempt in range(HTTP_RETRIES):
            try:
                response = self.session.get(
                    base_url,
                    params=params,
                    timeout=HTTP_TIMEOUT_SECONDS,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data

            except requests.RequestException as e:
                continue
            time.sleep(0.5 * (attempt + 1))
        return None

    def _extract_year(self, date_string: Optional[str]) -> Optional[int]:
        """Extract year from various date formats."""
        if not date_string:
            return None
        try:
            parsed_date = parser.parse(date_string)
            return parsed_date.year
        except (ValueError, TypeError):
            # Fallback: extract 4-digit number from string
            year_match = re.search(r"\b(19|20)\d{2}\b", str(date_string))
            return int(year_match.group()) if year_match else None


class GoogleBooksClient(BaseApiClient):
    """Client for interacting with Google Books API to fetch book metadata."""

    def fetch_books(
        self,
        title: str,
        author: str = "",
        isbn: Optional[str] = None,
        max_results: int = 1,
        start_index: int = 0,
        lang_restrict: str = "en",
        print_type: str = "books",
    ) -> List[Book]:
        """
        Fetches book information from Google Books API based on search criteria.

        Queries the Google Books API using title, author, or ISBN. If ISBN is provided,
        it first attempts a direct ISBN search; if no results, falls back to title/author search.

        Args:
            title: The book title to search for.
            author: The author name to search for (optional, defaults to empty string).
            isbn: The ISBN to search for (optional, takes precedence if provided).
            max_results: Maximum number of results to return (default 1).
            start_index: Starting index for pagination (default 0).
            lang_restrict: Language restriction (default "en").
            print_type: Type of print media to search (default "books").

        Returns:
            List of Book objects containing metadata for matching books.
        """
        params = {
            "maxResults": max_results,
            "startIndex": start_index,
            "langRestrict": lang_restrict,
            "printType": print_type,
            "q": "",
        }

        # Build query: prioritize ISBN if provided, otherwise use title+author
        params["q"] = f"isbn:{isbn}" if isbn else f"intitle:{title}+inauthor:{author}"
        data = self._get_metadata(GOOGLE_BOOKS_BASE_URL, params) or {}

        # Fallback: if ISBN query failed, try title+author
        if isbn and not data.get("items"):
            params["q"] = f"intitle:{title}+inauthor:{author}"
            data = self._get_metadata(GOOGLE_BOOKS_BASE_URL, params) or {}

        items = data.get("items", []) or []

        results = []
        for item in items:
            vol_info = item.get("volumeInfo", {}) or {}
            isbn_13 = None
            isbn_10 = None
            for identifier in vol_info.get("industryIdentifiers", []):
                if identifier.get("type") == "ISBN_13":
                    isbn_13 = identifier.get("identifier")
                elif identifier.get("type") == "ISBN_10":
                    isbn_10 = identifier.get("identifier")

            book_title = vol_info.get("title") or ""
            book_authors = vol_info.get("authors") or []

            results.append(
                Book(
                    google_id=item.get("id") or "",
                    openlib_key=None,
                    title=book_title,
                    subtitle=vol_info.get("subtitle"),
                    authors=book_authors,
                    subjects=(vol_info.get("categories") or []),
                    published_date=vol_info.get("publishedDate"),
                    published_year=self._extract_year(vol_info.get("publishedDate")),
                    url=vol_info.get("infoLink"),
                    isbn13=isbn_13,
                    isbn10=isbn_10,
                    page_count=vol_info.get("pageCount"),
                    description=vol_info.get("description"),
                    google_average_rating=vol_info.get("averageRating"),
                    google_ratings_count=vol_info.get("ratingsCount"),
                    openlib_average_rating=None,
                    openlib_ratings_count=None,
                    openlib_edition_key=None,
                )
            )
        return results


class OpenLibraryClient(BaseApiClient):
    """Client for interacting with OpenLibrary API to fetch book data and metadata."""

    def _normalize_title(self, title: str) -> str:
        cleaned = re.sub(r"\s+", " ", title).strip()
        split_title = re.split(r"[:/\[(\-]", cleaned, maxsplit=1)[0]
        return split_title.strip()

    def _author_last_name(self, author: str) -> str:
        """Regex pattern to get author last name."""
        parts = [part for part in re.split(r"\s+", author.strip()) if part]
        return parts[-1] if parts else ""

    def _lookup_by_isbn(
        self, isbn13: Optional[str], isbn10: Optional[str]
    ) -> Optional[str]:
        """Try to lookup openlibrary work key by ISBN."""
        for isbn in [isbn13, isbn10]:
            if not isbn:
                continue
            books = self.fetch_books(query=f"isbn:{isbn}", limit=10)
            if books:
                return books[0]["key"]
        return None

    def _build_title_author_queries(
        self,
        title: str,
        author: str,
    ) -> List[Tuple[str, str]]:
        """
        Build different query specs for title+author search.

        Creates multiple query variations to improve search results, including:
        - Full author name + title
        - Author last name + title (if available)
        - Title only

        All queries use "en" as the language code.

        Args:
            title: The book title.
            author: The author name.

        Returns:
            List of tuples, each containing (query_string, language_code).
        """
        normalized_title = self._normalize_title(title)
        last_name = self._author_last_name(author)

        base_queries = [
            f"author:{author} title:{normalized_title}",
            f"author:{last_name} title:{normalized_title}" if last_name else None,
            f"title:{normalized_title}",
        ]
        base_queries = [q for q in base_queries if q]

        return [(q, "en") for q in base_queries]

    def _find_best_matching(
        self,
        docs: List[Dict[str, Any]],
        target_title: str,
        target_author: str,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Finds the best matching document from a list of search results based on title and author similarity.

        Uses fuzzy string matching to compare the target title and author against each document's
        title and first author. Scores are weighted (70% title, 25% author) to determine the best match.
        Documents without a title are skipped.

        Args:
            docs: List of dictionaries representing search results from OpenLibrary API.
            target_title: The title string to match against.
            target_author: The author string to match against.

        Returns:
            A tuple containing:
            - The best matching document dict (or None if no valid matches).
            - The similarity score (float between 0.0 and 1.0).
        """
        best_doc = None
        best_score = 0.0

        for doc in docs:
            title = (doc.get("title") or "").strip()
            authors = doc.get("author_name") or []
            first_author = (authors[0] if authors else "").strip()

            if not title:
                continue

            title_score = (
                fuzz.token_set_ratio(target_title.lower(), title.lower()) / 100.0
            )
            author_score = (
                fuzz.token_set_ratio(target_author.lower(), first_author.lower())
                / 100.0
                if first_author
                else 0.0
            )
            score = 0.75 * title_score + 0.25 * author_score

            if score > best_score:
                best_score = score
                best_doc = doc

        return best_doc, best_score

    def _search_with_fuzzy_matching(
        self,
        query_specs: List[Tuple[str, str]],
        limit: int,
        normalized_title: str,
        author: str,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Run multiple searches and return the best matching book with its score.

        For each query spec (query string and language), performs a search with the given limit,
        then uses fuzzy matching to find the best book match. Returns the highest scoring match
        across all queries.

        Args:
            query_specs: List of (query_string, language_code) tuples to search.
            limit: Maximum number of results per search query.
            normalized_title: Normalized title string for fuzzy matching.
            author: Author string for fuzzy matching.

        Returns:
            Tuple of (best_matching_book_dict, highest_score). Returns (None, 0.0) if no matches.
        """
        best = None
        best_score = 0.0

        for query, lang in query_specs:
            books = self.fetch_books(query=query, limit=limit, language=lang)
            if books:
                doc, score = self._find_best_matching(books, normalized_title, author)
                if score > best_score:
                    best = doc
                    best_score = score

        return best, best_score

    def _build_url(self, key: str, suffix: str = ".json") -> str:
        """Build OpenLibrary API URL from key and suffix.

        Returns:
            str: URL in format 'https://openlibrary.org{prefix}{key}{suffix}'
        """
        prefix = "" if key.startswith("/") else "/"
        return f"https://openlibrary.org{prefix}{key}{suffix}"

    def extract_subject_slugs(
        self,
        data: Dict[str, Any],
        key: Literal["subjects", "related_subjects"] = "subjects",
    ) -> List[str]:
        """Extract and normalize subject strings to slugs from a data dict."""
        items = data.get(key, [])
        result = []
        for item in items:
            if not isinstance(item, str):
                continue
            slug = re.sub(r"[^a-z0-9]+", "_", item.strip().lower()).strip("_")
            if slug:
                result.append(slug)
        return result

    def fetch_books(
        self,
        query: str,
        limit: int = 5,
        page: int = 1,
        offset: int = 0,
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        """
        Fetch books from Open Library API by a search query.

        Args:
            query: The search query.
            limit: The maximum number of results to return.
            page: The page number to return.
            offset: The offset to start from.
            language: The language to search in.

        Returns:
            A list of dicts with fetched book data.
        """
        params = {
            "q": query,
            "limit": limit,
            "page": page,
            "offset": offset,
            "fields": ",".join(OPENLIB_FIELDS),
            "lang": language,
        }
        response = self._get_metadata(OPENLIB_BASE_URL, params) or {}
        return response.get("docs", []) or []

    def fetch_work_or_edition_json(self, key: str) -> Dict[str, Any]:
        return self._get_metadata(base_url=self._build_url(key)) or {}

    def fetch_ratings(
        self, key: str, fallback_to_edition: bool = False
    ) -> Dict[str, Any]:
        """Fetch ratings for a work or edition key.

        Args:
            key: The OpenLibrary work or edition key.
            fallback_to_edition: If True and work has no ratings, fetch ratings from first edition.

        Returns:
            Dict containing ratings data with 'summary' key.
        """
        ratings = (
            self._get_metadata(base_url=self._build_url(key, "/ratings.json")) or {}
        )

        # If no ratings count and we want edition fallback
        if fallback_to_edition:
            summary = ratings.get("summary", {}) if isinstance(ratings, dict) else {}
            if not summary:
                # Fetch editions and get first edition's ratings
                editions = self.fetch_editions(key, limit=1)
                edition_docs = editions.get("entries", []) or []
                if edition_docs:
                    edition_key = edition_docs[0].get("key")
                    if edition_key:
                        ratings = (
                            self._get_metadata(
                                base_url=self._build_url(edition_key, "/ratings.json")
                            )
                            or {}
                        )

        return ratings

    def fetch_editions(self, key: str, limit: int = 1) -> Dict[str, Any]:
        return (
            self._get_metadata(
                base_url=self._build_url(key, "/editions.json"), params={"limit": limit}
            )
            or {}
        )

    def fetch_books_by_subjects(
        self, subject: str, related_subs_flag: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Query OpenLibrary Subjects API for books matching a subject. Optional
        to query for related subjects to the search subject with details=true.

        Args:
            subject: Subject slug (e.g., "science_fiction")
            related_subs_flag: If True, query with ?details=true and return related subjects

        Returns:
            Tuple of (books_list, related_subjects)
            - books_list: List of book dicts with key, title, authors, isbn
            - related_subjects: Empty list if related_subs_flag=False, else list of related subject slugs
        """
        url = f"https://openlibrary.org/subjects/{subject}.json"
        if related_subs_flag:
            url += "?details=true"

        data = self._get_metadata(base_url=url) or {}

        # Extract books
        books = []
        for book in data.get("works", []):
            book_data = {
                "key": book.get("key"),
                "title": book.get("title"),
                "authors": [
                    author.get("name", "")
                    for author in book.get("authors", [])
                    if author.get("name")
                ],
                "isbn": book.get("isbn", []),
            }
            books.append(book_data)

        # Extract related subjects if details=True
        related_subjects = (
            self.extract_subject_slugs(data, "related_subjects")
            if related_subs_flag
            else []
        )

        return books, related_subjects

    def find_openlib_work_key(
        self,
        title: str,
        author: str,
        isbn13: Optional[str],
        isbn10: Optional[str],
    ) -> str:
        """
        Find OpenLibrary work key for a book by ISBN or fuzzy title/author matching.

        First attempts direct lookup by ISBN13 or ISBN10. If unsuccessful, performs fuzzy
        matching on title and author across multiple query variations and result limits
        to find the best matching book.

        Args:
            title: The book title.
            author: The author name.
            isbn13: ISBN13 string (optional, tried first).
            isbn10: ISBN10 string (optional, tried if isbn13 fails).

        Returns:
            The OpenLibrary work key string.

        Raises:
            ValueError: If no confident match is found after searches.
        """
        # Try ISBN lookup
        if key := self._lookup_by_isbn(isbn13, isbn10):
            return key

        # Build title+author queries
        normalized_title = self._normalize_title(title)
        query_specs = self._build_title_author_queries(title, author)

        # Attempt search with limit=10 per query to find a high-confidence match
        # Evaluates up to 30 books (3 query variations * 10 results each) for the best fuzzy match
        best, score = self._search_with_fuzzy_matching(
            query_specs=query_specs,
            limit=10,
            normalized_title=normalized_title,
            author=author,
        )
        if best and score >= 0.7:
            return best["key"]

        # Escalate to limit=50
        best, score = self._search_with_fuzzy_matching(
            query_specs=query_specs,
            limit=50,
            normalized_title=normalized_title,
            author=author,
        )
        if best and score >= 0.7:
            return best["key"]

        raise ValueError(f"OpenLibrary: no confident match for '{title}' by {author}")
