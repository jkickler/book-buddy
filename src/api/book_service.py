# Service for searching and enriching book data from Google Books and OpenLibrary APIs
from typing import Optional

from loguru import logger

from src.core.book import Book

from .client import GoogleBooksClient, OpenLibraryClient


class BookDataService:
    """Service for searching and enriching book data using Google Books and OpenLibrary APIs.

    Provides methods to:
    - Search for books by title/author and enrich them with metadata
    - Enrich existing Book objects with OpenLibrary metadata
    - Enrich books using a known OpenLibrary work key
    """

    def __init__(self):
        self.google_client = GoogleBooksClient()
        self.openlib_client = OpenLibraryClient()

    def _add_openlib_subjects(self, book: Book, ol_data: dict) -> None:
        """Add OpenLibrary subjects to the book. Extracts subject slugs from ol_data,
        and updates the book's subjects and openlib_key."""
        subjects = self.openlib_client.extract_subject_slugs(
            data=ol_data, key="subjects"
        )
        if subjects:
            # Openlib subjects are added to existing subjects
            book.subjects = book.subjects + subjects
        elif not book.subjects:
            book.subjects = []

    def _update_book_details(
        self, book: Book, ol_data: dict, editions_data: Optional[dict] = None
    ) -> None:
        """Update book details with OpenLibrary data. Extracts 'description', 'subtitle',
        'published_date', 'page_count' from ol_data and updates the book if fields are missing.
        """
        description = ol_data.get("description")
        if isinstance(description, dict):
            description = description.get("value")
        if description and not book.description:
            book.description = description

        subtitle = ol_data.get("subtitle")
        if subtitle and not book.subtitle:
            book.subtitle = subtitle

        published_date = ol_data.get("publish_date")
        if published_date and not book.published_date:
            book.published_date = str(published_date)
            if not book.published_year:
                book.published_year = self.openlib_client._extract_year(
                    book.published_date
                )

        if book.page_count is None and editions_data:
            edition_docs = editions_data.get("entries", []) or []
            if edition_docs:
                pages = edition_docs[0].get("number_of_pages")
                if pages:
                    book.page_count = pages

    def _add_openlib_ratings(self, book: Book, ol_work_key: str) -> None:
        """Add OpenLibrary ratings to the book.

        If Google ratings are present, skips. Otherwise fetches ratings with
        fallback to first edition if work has no ratings.
        """
        if book.google_average_rating is not None:
            return

        ratings = self.openlib_client.fetch_ratings(
            ol_work_key, fallback_to_edition=True
        )
        summary = ratings.get("summary", {}) if isinstance(ratings, dict) else {}
        book.openlib_average_rating = summary.get("average")
        book.openlib_ratings_count = summary.get("count")

    def enrich_with_openlib_key(self, book: Book, ol_work_key: str) -> Book:
        """Enrich a book using an existing OpenLibrary work key.

        Fetches work data, editions, and ratings from OpenLibrary and updates
        the book object with the retrieved metadata.

        Args:
            book: Book object to enrich.
            ol_work_key: OpenLibrary work key (e.g., "/works/OL123W").

        Returns:
            The enriched Book object.
        """
        book.openlib_key = ol_work_key
        ol_data = self.openlib_client.fetch_work_or_edition_json(ol_work_key)
        editions_data = self.openlib_client.fetch_editions(ol_work_key, limit=1)

        self._update_book_details(book, ol_data, editions_data)
        self._add_openlib_subjects(book, ol_data)
        self._add_openlib_ratings(book, ol_work_key)
        return book

    def search_books(
        self,
        book_title: str,
        book_author: str,
    ) -> Book:
        """Search for a book by title and author, then enrich with metadata.

        Searches Google Books across multiple languages, selects the best match,
        finds the corresponding OpenLibrary work key, and enriches the book
        with subjects, ratings, and additional metadata from OpenLibrary.

        Args:
            book_title: Title to search for.
            book_author: Author to search for.

        Returns:
            Enriched Book object with metadata from both Google Books and OpenLibrary.

        Raises:
            ValueError: If no results from Google Books or no confident OpenLibrary match.
        """
        logger.info(f"START search for '{book_title}' by {book_author}")
        books = []
        for language in ["en", "de"]:
            books.extend(
                self.google_client.fetch_books(
                    book_title,
                    book_author,
                    max_results=5,
                    lang_restrict=language,
                )
            )
        if not books:
            raise ValueError("Google Books: no results")

        last_error = None
        selected = None
        ol_work_key = None
        for candidate in books:
            try:
                ol_work_key = self.openlib_client.find_openlib_work_key(
                    title=candidate.title,
                    author=candidate.authors[0] if candidate.authors else book_author,
                    isbn13=candidate.isbn13,
                    isbn10=candidate.isbn10,
                )
                selected = candidate
                break
            except ValueError as exc:
                last_error = exc

        if not selected or not ol_work_key:
            raise last_error or ValueError("OpenLibrary: no confident match")
        result = self.enrich_with_openlib_key(selected, ol_work_key)
        logger.info(f"Found: {result.title}")
        return result

    def enrich_book_data(
        self,
        book: Book,
        book_title: Optional[str] = None,
        book_author: Optional[str] = None,
    ) -> Book:
        """Enrich an existing Book object with metadata from OpenLibrary.

        Finds the OpenLibrary work key using the book's ISBN or title/author,
        then fetches and applies subjects, description, ratings, and other
        metadata from OpenLibrary to the book object.

        Args:
            book: Book object to enrich. Must have title and authors, or provide them via optional params.
            book_title: Optional title to use if book is missing title.
            book_author: Optional author to use if book is missing authors.

        Returns:
            The enriched Book object with OpenLibrary metadata.

        Raises:
            ValueError: If book is missing title/author and not provided via optional params.
        """
        title_for_log = book.title or book_title
        logger.info(f"START enriching '{title_for_log}'")
        if not book.title or not book.authors:
            if book_title and book_author:
                book.title = book_title
                book.authors = [book_author]
            else:
                raise ValueError(
                    "Book is missing title/author; provide book_title/book_author or a complete Book"
                )

        ol_work_key = self.openlib_client.find_openlib_work_key(
            title=book.title,
            author=book.authors[0],
            isbn13=book.isbn13,
            isbn10=book.isbn10,
        )
        result = self.enrich_with_openlib_key(book, ol_work_key)
        logger.info(f"Enriched: {result.title}")
        return result
