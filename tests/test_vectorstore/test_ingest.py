"""Tests for CSV ingestion functions."""

import pytest
from unittest.mock import MagicMock, Mock, patch, mock_open
import pandas as pd
from io import StringIO
from src.vectorstore.ingest import (
    _find_column,
    _row_value,
    _normalize_isbn,
    _to_float,
    _to_int,
    parse_csv,
    fill_missing_data,
)
from src.core.book import Book


class TestFindColumn:
    """Test cases for _find_column function."""

    def test_find_exact_match(self):
        """Test finding exact column name match."""
        columns = ["Title", "Author", "ISBN"]
        result = _find_column(columns, ["title"])
        assert result == "Title"

    def test_find_case_insensitive(self):
        """Test case-insensitive column matching."""
        columns = ["TITLE", "AUTHOR"]
        result = _find_column(columns, ["title"])
        assert result == "TITLE"

    def test_find_partial_match(self):
        """Test partial column name matching."""
        columns = ["Book Title", "Book Author"]
        result = _find_column(columns, ["title"])
        assert result == "Book Title"

    def test_find_first_match(self):
        """Test that first matching candidate is returned."""
        columns = ["Title", "Book Title"]
        result = _find_column(columns, ["title", "book title"])
        assert result == "Title"

    def test_find_no_match(self):
        """Test returning None when no match found."""
        columns = ["Name", "Date"]
        result = _find_column(columns, ["title", "author"])
        assert result is None


class TestRowValue:
    """Test cases for _row_value function."""

    def test_row_value_string(self):
        """Test extracting string value."""
        row = {"title": "Test Book", "author": "Test Author"}
        result = _row_value(row, "title")
        assert result == "Test Book"

    def test_row_value_strips_whitespace(self):
        """Test that whitespace is stripped."""
        row = {"title": "  Test Book  "}
        result = _row_value(row, "title")
        assert result == "Test Book"

    def test_row_value_none_column(self):
        """Test with None column name."""
        row = {"title": "Test"}
        result = _row_value(row, None)
        assert result == ""

    def test_row_value_nan(self):
        """Test handling NaN values."""
        row = pd.Series({"title": float("nan")})
        result = _row_value(row, "title")
        assert result == ""

    def test_row_value_none_value(self):
        """Test handling None values."""
        row = {"title": None}
        result = _row_value(row, "title")
        assert result == ""


class TestNormalizeIsbn:
    """Test cases for _normalize_isbn function."""

    def test_normalize_isbn_basic(self):
        """Test basic ISBN normalization."""
        result = _normalize_isbn("978-1-234-56789-0")
        assert result == "9781234567890"

    def test_normalize_isbn_with_spaces(self):
        """Test ISBN with spaces."""
        result = _normalize_isbn("978 1 234 56789 0")
        assert result == "9781234567890"

    def test_normalize_isbn_with_x(self):
        """Test ISBN with X character."""
        result = _normalize_isbn("123456789X")
        assert result == "123456789X"

    def test_normalize_isbn_with_formula_prefix(self):
        """Test ISBN with Excel formula prefix."""
        result = _normalize_isbn('="9781234567890"')
        assert result == "9781234567890"

    def test_normalize_isbn_empty(self):
        """Test empty ISBN."""
        result = _normalize_isbn("")
        assert result == ""

    def test_normalize_isbn_none(self):
        """Test None ISBN."""
        result = _normalize_isbn(None)
        assert result == ""


class TestToFloat:
    """Test cases for _to_float function."""

    def test_to_float_valid(self):
        """Test converting valid string to float."""
        result = _to_float("4.5")
        assert result == 4.5

    def test_to_float_integer(self):
        """Test converting integer string to float."""
        result = _to_float("5")
        assert result == 5.0

    def test_to_float_empty(self):
        """Test empty string returns None."""
        result = _to_float("")
        assert result is None

    def test_to_float_invalid(self):
        """Test invalid string returns None."""
        result = _to_float("not a number")
        assert result is None


class TestToInt:
    """Test cases for _to_int function."""

    def test_to_int_valid(self):
        """Test converting valid string to int."""
        result = _to_int("2023")
        assert result == 2023

    def test_to_int_from_float_string(self):
        """Test converting float string to int."""
        result = _to_int("2023.0")
        assert result == 2023

    def test_to_int_empty(self):
        """Test empty string returns None."""
        result = _to_int("")
        assert result is None

    def test_to_int_invalid(self):
        """Test invalid string returns None."""
        result = _to_int("not a number")
        assert result is None


class TestParseCsv:
    """Test cases for parse_csv function."""

    def test_parse_basic_csv(self):
        """Test parsing basic CSV content."""
        csv_content = """Title,Author,ISBN
Test Book,Test Author,9781234567890
"""
        file_obj = StringIO(csv_content)
        result = parse_csv(file_obj)

        assert len(result) == 1
        assert result[0]["title"] == "Test Book"
        assert result[0]["author"] == "Test Author"
        assert result[0]["isbn13"] == "9781234567890"

    def test_parse_csv_with_ratings(self):
        """Test parsing CSV with rating columns."""
        csv_content = """Title,Author,ISBN13,My Rating,User Rating
Test Book,Author,9781234567890,5,4.5
"""
        file_obj = StringIO(csv_content)
        result = parse_csv(file_obj)

        assert result[0]["goodreads_user_rating"] == 5.0
        assert result[0]["goodreads_community_rating"] == 4.5

    def test_parse_csv_missing_required_columns_raises(self):
        """Test that missing required columns raises error."""
        csv_content = """Name,Date
Test,2023
"""
        file_obj = StringIO(csv_content)

        with pytest.raises(ValueError, match="title and author"):
            parse_csv(file_obj)

    def test_parse_csv_multiple_rows(self):
        """Test parsing multiple rows."""
        csv_content = """Title,Author
Book 1,Author 1
Book 2,Author 2
Book 3,Author 3
"""
        file_obj = StringIO(csv_content)
        result = parse_csv(file_obj)

        assert len(result) == 3
        assert result[2]["title"] == "Book 3"

    def test_parse_csv_with_year(self):
        """Test parsing CSV with year published."""
        csv_content = """Title,Author,Year Published
Test Book,Author,2023
"""
        file_obj = StringIO(csv_content)
        result = parse_csv(file_obj)

        assert result[0]["published_year"] == 2023


class TestFillMissingData:
    """Test cases for fill_missing_data function."""

    def test_fill_missing_description(self):
        """Test filling missing description."""
        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
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

        candidate = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
            subtitle=None,
            authors=["Author"],
            subjects=[],
            description="Candidate Description",
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

        fill_missing_data(book, candidate)

        assert book.description == "Candidate Description"

    def test_does_not_overwrite_existing(self):
        """Test that existing fields are not overwritten."""
        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
            subtitle=None,
            authors=["Author"],
            subjects=[],
            description="Existing Description",
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

        candidate = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
            subtitle=None,
            authors=["Author"],
            subjects=[],
            description="New Description",
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

        fill_missing_data(book, candidate)

        assert book.description == "Existing Description"

    def test_fill_isbn(self):
        """Test filling missing ISBN."""
        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
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

        candidate = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
            subtitle=None,
            authors=["Author"],
            subjects=[],
            description=None,
            isbn13="9781234567890",
            isbn10="1234567890",
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

        fill_missing_data(book, candidate)

        assert book.isbn13 == "9781234567890"
        assert book.isbn10 == "1234567890"

    def test_fill_subjects(self):
        """Test filling missing subjects."""
        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
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

        candidate = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
            subtitle=None,
            authors=["Author"],
            subjects=["fiction", "adventure"],
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

        fill_missing_data(book, candidate)

        assert book.subjects == ["fiction", "adventure"]

    def test_fill_published_year(self):
        """Test filling missing published year."""
        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
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

        candidate = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
            subtitle=None,
            authors=["Author"],
            subjects=[],
            description=None,
            isbn13=None,
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

        fill_missing_data(book, candidate)

        assert book.published_year == 2023

    def test_fill_page_count(self):
        """Test filling missing page count."""
        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
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

        candidate = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
            subtitle=None,
            authors=["Author"],
            subjects=[],
            description=None,
            isbn13=None,
            isbn10=None,
            published_date=None,
            published_year=None,
            url=None,
            page_count=300,
            google_average_rating=None,
            google_ratings_count=None,
            openlib_average_rating=None,
            openlib_ratings_count=None,
            openlib_edition_key=None,
        )

        fill_missing_data(book, candidate)

        assert book.page_count == 300

    def test_fill_google_ratings(self):
        """Test filling missing Google ratings."""
        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
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

        candidate = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
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
            google_average_rating=4.5,
            google_ratings_count=1000,
            openlib_average_rating=None,
            openlib_ratings_count=None,
            openlib_edition_key=None,
        )

        fill_missing_data(book, candidate)

        assert book.google_average_rating == 4.5
        assert book.google_ratings_count == 1000

    def test_fill_google_id(self):
        """Test filling missing Google ID."""
        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
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

        candidate = Book(
            google_id="google123",
            openlib_key=None,
            title="Test",
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

        fill_missing_data(book, candidate)

        assert book.google_id == "google123"
