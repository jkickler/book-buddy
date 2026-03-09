"""Tests for enrich_and_score tool."""

import pytest
from unittest.mock import MagicMock, Mock, patch
from src.graph.tools.enrich_and_score import (
    _is_empty_query,
    fetch_single_book,
    book_query_to_dict,
    process_book_query,
)
from src.graph.schemas import BookQuery
from src.core.book import Book


class TestIsEmptyQuery:
    """Test cases for _is_empty_query function."""

    def test_empty_query_all_none(self):
        """Test query with all None fields is empty."""
        query = BookQuery(title=None, author=None, isbn=None)
        assert _is_empty_query(query) is True

    def test_empty_query_all_empty_strings(self):
        """Test query with all empty strings is empty."""
        query = BookQuery(title="", author="", isbn="")
        assert _is_empty_query(query) is True

    def test_not_empty_with_title(self):
        """Test query with title is not empty."""
        query = BookQuery(title="Test Book", author=None, isbn=None)
        assert _is_empty_query(query) is False

    def test_not_empty_with_author(self):
        """Test query with author is not empty."""
        query = BookQuery(title=None, author="Test Author", isbn=None)
        assert _is_empty_query(query) is False

    def test_not_empty_with_isbn(self):
        """Test query with ISBN is not empty."""
        query = BookQuery(title=None, author=None, isbn="9781234567890")
        assert _is_empty_query(query) is False


class TestFetchSingleBook:
    """Test cases for fetch_single_book function."""

    def test_fetch_by_isbn(self):
        """Test fetching book by ISBN."""
        mock_service = MagicMock()
        mock_book = Book(
            google_id="google1",
            openlib_key=None,
            title="Test Book",
            subtitle=None,
            authors=["Test Author"],
            subjects=[],
            description=None,
            isbn13="9781234567890",
            isbn10=None,
            published_date=None,
            published_year=None,
            url=None,
            page_count=None,
            google_average_rating=None,
            google_ratings_count=None,
            openlib_average_rating=None,
            openlib_ratings_count=None,
            openlib_edition_key=None,
        )
        mock_service.google_client.fetch_books.return_value = [mock_book]
        mock_service.enrich_book_data.return_value = mock_book

        query = BookQuery(title="Test", author="Author", isbn="9781234567890")
        result = fetch_single_book(query, mock_service)

        assert result.title == "Test Book"
        mock_service.google_client.fetch_books.assert_called_once_with(
            title="Test",
            author="Author",
            isbn="9781234567890",
        )

    def test_fetch_by_title_author_fallback(self):
        """Test falling back to title/author search."""
        mock_service = MagicMock()
        mock_service.google_client.fetch_books.return_value = []  # ISBN search returns empty
        mock_book = Book(
            google_id="google1",
            openlib_key=None,
            title="Test Book",
            subtitle=None,
            authors=["Test Author"],
            subjects=[],
            description=None,
            isbn13=None,
            isbn10=None,
            published_date=None,
            published_year=None,
            url=None,
            page_count=None,
            google_average_rating=None,
            google_ratings_count=None,
            openlib_average_rating=None,
            openlib_ratings_count=None,
            openlib_edition_key=None,
        )
        mock_service.search_books.return_value = mock_book

        query = BookQuery(title="Test Book", author="Test Author", isbn=None)
        result = fetch_single_book(query, mock_service)

        assert result.title == "Test Book"
        mock_service.search_books.assert_called_once_with("Test Book", "Test Author")

    def test_fetch_raises_on_no_info(self):
        """Test raising error when no title/author/isbn provided."""
        mock_service = MagicMock()
        mock_service.google_client.fetch_books.return_value = []

        query = BookQuery(title=None, author=None, isbn=None)

        with pytest.raises(ValueError, match="Missing title, author, or ISBN"):
            fetch_single_book(query, mock_service)


class TestBookQueryToDict:
    """Test cases for book_query_to_dict function."""

    def test_convert_complete_query(self):
        """Test converting complete query to dict."""
        query = BookQuery(title="Test", author="Author", isbn="123")
        result = book_query_to_dict(query)

        assert result == {"title": "Test", "author": "Author", "isbn": "123"}

    def test_convert_partial_query(self):
        """Test converting partial query to dict."""
        query = BookQuery(title="Test", author=None, isbn=None)
        result = book_query_to_dict(query)

        assert result == {"title": "Test", "author": None, "isbn": None}


class TestProcessBookQuery:
    """Test cases for process_book_query function."""

    def test_process_empty_query(self, mock_vector_store):
        """Test processing empty query returns clarify status."""
        mock_service = MagicMock()
        query = BookQuery(title=None, author=None, isbn=None)

        result = process_book_query(query, mock_service, mock_vector_store)

        assert result["status"] == "clarify"
        assert "Please provide" in result["message"]
        assert result["query"]["title"] is None

    def test_process_not_found(self, mock_vector_store):
        """Test processing query with no results returns not_found."""
        mock_service = MagicMock()
        mock_service.google_client.fetch_books.return_value = []
        mock_service.search_books.side_effect = ValueError("No results")

        query = BookQuery(title="NonExistent", author="Unknown", isbn=None)
        result = process_book_query(query, mock_service, mock_vector_store)

        assert result["status"] == "not_found"
        assert "No confident match" in result["message"]

    def test_process_success(self, mock_vector_store):
        """Test successful book processing."""
        mock_service = MagicMock()
        mock_book = Book(
            google_id="google1",
            openlib_key=None,
            title="Test Book",
            subtitle=None,
            authors=["Test Author"],
            subjects=["fiction"],
            description="Test description",
            isbn13="9781234567890",
            isbn10=None,
            published_date=None,
            published_year=2023,
            url=None,
            page_count=None,
            google_average_rating=None,
            google_ratings_count=None,
            openlib_average_rating=None,
            openlib_ratings_count=None,
            openlib_edition_key=None,
        )
        mock_service.google_client.fetch_books.return_value = [mock_book]
        mock_service.enrich_book_data.return_value = mock_book

        query = BookQuery(title="Test", author="Author", isbn=None)
        result = process_book_query(query, mock_service, mock_vector_store)

        assert result["status"] == "ok"
        assert "enriched_book" in result
        assert "similarity_scores" in result
        assert result["enriched_book"]["title"] == "Test Book"
