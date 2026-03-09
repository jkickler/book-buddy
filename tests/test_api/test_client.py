"""Tests for API client functions."""

import pytest
from unittest.mock import MagicMock, patch, Mock
from src.api.client import (
    BaseApiClient,
    GoogleBooksClient,
    OpenLibraryClient,
    OPENLIB_BASE_URL,
    GOOGLE_BOOKS_BASE_URL,
)


class TestBaseApiClient:
    """Test cases for BaseApiClient."""

    def test_init_sets_user_agent(self):
        """Test that client initializes with User-Agent header."""
        client = BaseApiClient()
        assert "User-Agent" in client.session.headers
        assert "book-recommendation-app" in client.session.headers["User-Agent"]

    def test_get_metadata_success(self, mock_requests_session):
        """Test successful metadata fetch."""
        client = BaseApiClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {"key": "value"}

        result = client._get_metadata("http://test.com", {"param": "value"})

        assert result == {"key": "value"}
        mock_requests_session.get.assert_called_once()

    def test_get_metadata_failure(self, mock_requests_session):
        """Test metadata fetch with server error."""
        client = BaseApiClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 500

        result = client._get_metadata("http://test.com", {})

        assert result is None

    def test_get_metadata_exception(self, mock_requests_session):
        """Test metadata fetch with request exception."""
        import requests

        client = BaseApiClient()
        client.session = mock_requests_session
        mock_requests_session.get.side_effect = requests.RequestException(
            "Network error"
        )

        result = client._get_metadata("http://test.com", {})

        assert result is None

    def test_get_metadata_retry_success(self, mock_requests_session):
        """Test that retry logic eventually succeeds."""
        client = BaseApiClient()
        client.session = mock_requests_session
        # First two calls fail, third succeeds
        mock_requests_session.get.side_effect = [
            MagicMock(status_code=500),
            MagicMock(status_code=500),
            MagicMock(status_code=200, json=lambda: {"success": True}),
        ]

        result = client._get_metadata("http://test.com", {})

        assert result == {"success": True}
        assert mock_requests_session.get.call_count == 3

    def test_extract_year_from_iso_date(self):
        """Test extracting year from ISO date format."""
        client = BaseApiClient()
        assert client._extract_year("2023-01-15") == 2023
        assert client._extract_year("2020-12-31") == 2020

    def test_extract_year_from_year_only(self):
        """Test extracting year from year-only string."""
        client = BaseApiClient()
        assert client._extract_year("2023") == 2023

    def test_extract_year_none(self):
        """Test extracting year from None."""
        client = BaseApiClient()
        assert client._extract_year(None) is None

    def test_extract_year_fallback_regex(self):
        """Test fallback regex extraction."""
        client = BaseApiClient()
        assert client._extract_year("Published in 2023 by Publisher") == 2023
        assert client._extract_year("Copyright 1999") == 1999


class TestGoogleBooksClient:
    """Test cases for GoogleBooksClient."""

    def test_fetch_books_with_isbn(self, mock_requests_session):
        """Test fetching books with ISBN."""
        client = GoogleBooksClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {
            "items": [
                {
                    "id": "book123",
                    "volumeInfo": {
                        "title": "Test Book",
                        "authors": ["Test Author"],
                        "industryIdentifiers": [
                            {"type": "ISBN_13", "identifier": "9781234567890"},
                        ],
                    },
                }
            ]
        }

        books = client.fetch_books(title="Test", isbn="9781234567890")

        assert len(books) == 1
        assert books[0].title == "Test Book"
        assert books[0].isbn13 == "9781234567890"

    def test_fetch_books_with_title_author(self, mock_requests_session):
        """Test fetching books with title and author."""
        client = GoogleBooksClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {
            "items": [
                {
                    "id": "book123",
                    "volumeInfo": {
                        "title": "Test Book",
                        "authors": ["Test Author"],
                        "publishedDate": "2023-01-15",
                        "pageCount": 300,
                        "averageRating": 4.5,
                        "ratingsCount": 1000,
                        "categories": ["Fiction"],
                    },
                }
            ]
        }

        books = client.fetch_books(title="Test Book", author="Test Author")

        assert len(books) == 1
        assert books[0].published_year == 2023
        assert books[0].page_count == 300
        assert books[0].google_average_rating == 4.5

    def test_fetch_books_empty_response(self, mock_requests_session):
        """Test fetching books with empty response."""
        client = GoogleBooksClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {"items": []}

        books = client.fetch_books(title="NonExistent")

        assert books == []

    def test_fetch_books_isbn_fallback(self, mock_requests_session):
        """Test that ISBN search falls back to title/author."""
        client = GoogleBooksClient()
        client.session = mock_requests_session
        # First call (ISBN) returns empty
        # Second call (title/author) returns results
        mock_requests_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"items": []}),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "items": [{"id": "book1", "volumeInfo": {"title": "Found"}}]
                },
            ),
        ]

        books = client.fetch_books(title="Test", isbn="9781234567890")

        assert len(books) == 1
        assert books[0].title == "Found"


class TestOpenLibraryClient:
    """Test cases for OpenLibraryClient."""

    def test_normalize_title_with_subtitle(self):
        """Test title normalization removes subtitle."""
        client = OpenLibraryClient()
        assert client._normalize_title("Title: Subtitle") == "Title"
        assert client._normalize_title("Title - Subtitle") == "Title"
        assert client._normalize_title("Title/Subtitle") == "Title"

    def test_normalize_title_extra_whitespace(self):
        """Test title normalization handles whitespace."""
        client = OpenLibraryClient()
        assert (
            client._normalize_title("  Title   With   Spaces  ") == "Title With Spaces"
        )

    def test_author_last_name(self):
        """Test extracting author last name."""
        client = OpenLibraryClient()
        assert client._author_last_name("John Doe") == "Doe"
        assert client._author_last_name("Jane Mary Smith") == "Smith"
        assert client._author_last_name("Single") == "Single"

    def test_author_last_name_empty(self):
        """Test extracting last name from empty string."""
        client = OpenLibraryClient()
        assert client._author_last_name("") == ""
        assert client._author_last_name("   ") == ""

    def test_build_title_author_queries(self):
        """Test building query variations."""
        client = OpenLibraryClient()
        queries = client._build_title_author_queries("Test Title", "John Doe")

        assert len(queries) >= 2
        # Should include full author and title
        assert any("author:John Doe" in q for q, _ in queries)
        assert any("title:Test Title" in q for q, _ in queries)

    def test_build_url(self):
        """Test URL building."""
        client = OpenLibraryClient()
        assert (
            client._build_url("/works/OL123W")
            == "https://openlibrary.org/works/OL123W.json"
        )
        assert (
            client._build_url("works/OL123W")
            == "https://openlibrary.org/works/OL123W.json"
        )

    def test_extract_subject_slugs(self):
        """Test subject slug extraction."""
        client = OpenLibraryClient()
        data = {"subjects": ["Science Fiction", "Adventure Stories", "Fiction"]}
        slugs = client.extract_subject_slugs(data)

        assert "science_fiction" in slugs
        assert "adventure_stories" in slugs
        assert "fiction" in slugs

    def test_extract_subject_slugs_non_strings(self):
        """Test subject extraction with non-string values."""
        client = OpenLibraryClient()
        data = {"subjects": ["Fiction", 123, None, "Science Fiction"]}
        slugs = client.extract_subject_slugs(data)

        assert "fiction" in slugs
        assert "science_fiction" in slugs
        assert 123 not in slugs

    def test_fetch_books(self, mock_requests_session):
        """Test fetching books from OpenLibrary."""
        client = OpenLibraryClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {
            "docs": [
                {
                    "key": "/works/OL123W",
                    "title": "Test Book",
                    "author_name": ["Test Author"],
                },
            ]
        }

        books = client.fetch_books("test query", limit=5)

        assert len(books) == 1
        assert books[0]["title"] == "Test Book"

    def test_fetch_work_or_edition_json(self, mock_requests_session):
        """Test fetching work/edition JSON."""
        client = OpenLibraryClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {
            "title": "Test Work",
            "description": "Test Description",
        }

        result = client.fetch_work_or_edition_json("/works/OL123W")

        assert result["title"] == "Test Work"

    def test_fetch_ratings(self, mock_requests_session):
        """Test fetching ratings."""
        client = OpenLibraryClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {
            "summary": {"average": 4.5, "count": 100}
        }

        ratings = client.fetch_ratings("/works/OL123W")

        assert ratings["summary"]["average"] == 4.5

    def test_fetch_books_by_subjects(self, mock_requests_session):
        """Test fetching books by subject."""
        client = OpenLibraryClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {
            "works": [
                {
                    "key": "/works/OL123W",
                    "title": "Test Book",
                    "authors": [{"name": "Test Author"}],
                    "isbn": ["9781234567890"],
                }
            ]
        }

        books, related = client.fetch_books_by_subjects("fiction")

        assert len(books) == 1
        assert books[0]["title"] == "Test Book"
        assert related == []

    def test_fetch_books_by_subjects_with_related(self, mock_requests_session):
        """Test fetching books with related subjects."""
        client = OpenLibraryClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {
            "works": [{"key": "/works/OL123W", "title": "Test", "authors": []}],
            "related_subjects": ["Science Fiction", "Adventure"],
        }

        books, related = client.fetch_books_by_subjects(
            "fiction", related_subs_flag=True
        )

        assert len(related) == 2
        assert "science_fiction" in related

    def test_find_openlib_work_key_by_isbn(self, mock_requests_session):
        """Test finding work key by ISBN."""
        client = OpenLibraryClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {
            "docs": [{"key": "/works/OL123W", "title": "Test Book"}]
        }

        key = client.find_openlib_work_key(
            title="Test",
            author="Author",
            isbn13="9781234567890",
            isbn10=None,
        )

        assert key == "/works/OL123W"

    def test_find_openlib_work_key_not_found(self, mock_requests_session):
        """Test finding work key when no match found."""
        client = OpenLibraryClient()
        client.session = mock_requests_session
        mock_requests_session.get.return_value.status_code = 200
        mock_requests_session.get.return_value.json.return_value = {"docs": []}

        with pytest.raises(ValueError, match="no confident match"):
            client.find_openlib_work_key(
                title="NonExistent",
                author="Unknown",
                isbn13=None,
                isbn10=None,
            )

    def test_find_best_matching(self):
        """Test fuzzy matching for best document."""
        client = OpenLibraryClient()
        docs = [
            {"title": "Test Book Title", "author_name": ["Test Author"]},
            {"title": "Different Book", "author_name": ["Other Author"]},
        ]

        best, score = client._find_best_matching(docs, "Test Book", "Test Author")

        assert best is not None
        assert score > 0.5

    def test_find_best_matching_no_title(self):
        """Test matching skips documents without title."""
        client = OpenLibraryClient()
        docs = [
            {"title": "", "author_name": ["Author"]},
            {"title": "Valid Title", "author_name": ["Author"]},
        ]

        best, score = client._find_best_matching(docs, "Valid Title", "Author")

        assert best["title"] == "Valid Title"
