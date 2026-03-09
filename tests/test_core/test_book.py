"""Tests for the Book dataclass."""

import pytest
from src.core.book import Book


class TestBook:
    """Test cases for Book dataclass."""

    def test_book_creation_with_all_fields(self, sample_book):
        """Test creating a Book with all fields populated."""
        assert sample_book.title == "Test Book Title"
        assert sample_book.subtitle == "A Test Subtitle"
        assert sample_book.authors == ["John Doe", "Jane Smith"]
        assert sample_book.subjects == ["fiction", "science_fiction", "adventure"]
        assert sample_book.isbn13 == "9781234567890"
        assert sample_book.published_year == 2023
        assert sample_book.google_average_rating == 4.5

    def test_book_creation_with_minimal_fields(self, minimal_book):
        """Test creating a Book with only required fields."""
        assert minimal_book.title == "Minimal Book"
        assert minimal_book.authors == ["Unknown Author"]
        assert minimal_book.subjects == []
        assert minimal_book.isbn13 is None
        assert minimal_book.published_year is None

    def test_book_immutability(self, sample_book):
        """Test that Book dataclass fields can be modified (not frozen)."""
        # Book is not frozen, so we can modify it
        sample_book.title = "Modified Title"
        assert sample_book.title == "Modified Title"

    def test_book_equality(self):
        """Test Book equality comparison."""
        book1 = Book(
            google_id="id1",
            openlib_key=None,
            title="Title",
            subtitle=None,
            authors=["Author"],
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

        book2 = Book(
            google_id="id1",
            openlib_key=None,
            title="Title",
            subtitle=None,
            authors=["Author"],
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

        book3 = Book(
            google_id="id2",  # Different ID
            openlib_key=None,
            title="Title",
            subtitle=None,
            authors=["Author"],
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

        assert book1 == book2
        assert book1 != book3

    def test_book_repr(self, sample_book):
        """Test Book string representation."""
        repr_str = repr(sample_book)
        assert "Book" in repr_str
        assert "Test Book Title" in repr_str
