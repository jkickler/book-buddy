"""Tests for BookDataService."""

import pytest
from unittest.mock import MagicMock, Mock, patch
from src.api.book_service import BookDataService
from src.core.book import Book


class TestBookDataServiceInit:
    """Test cases for BookDataService initialization."""

    def test_init_creates_clients(self):
        """Test that service initializes both API clients."""
        service = BookDataService()
        assert service.google_client is not None
        assert service.openlib_client is not None


class TestAddOpenlibSubjects:
    """Test cases for _add_openlib_subjects method."""

    def test_add_subjects_to_empty_book(self):
        """Test adding subjects to book with no subjects."""
        service = BookDataService()
        service.openlib_client = MagicMock()
        service.openlib_client.extract_subject_slugs.return_value = [
            "fiction",
            "adventure",
        ]

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

        service._add_openlib_subjects(book, {"subjects": ["Fiction", "Adventure"]})

        assert book.subjects == ["fiction", "adventure"]

    def test_add_subjects_to_existing_subjects(self):
        """Test adding subjects appends to existing."""
        service = BookDataService()
        service.openlib_client = MagicMock()
        service.openlib_client.extract_subject_slugs.return_value = ["adventure"]

        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
            subtitle=None,
            authors=["Author"],
            subjects=["existing"],
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

        service._add_openlib_subjects(book, {"subjects": ["Adventure"]})

        assert "existing" in book.subjects
        assert "adventure" in book.subjects

    def test_no_subjects_sets_empty_list(self):
        """Test that no subjects sets empty list."""
        service = BookDataService()
        service.openlib_client = MagicMock()
        service.openlib_client.extract_subject_slugs.return_value = []

        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
            subtitle=None,
            authors=["Author"],
            subjects=None,
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

        service._add_openlib_subjects(book, {"subjects": []})

        assert book.subjects == []


class TestUpdateBookDetails:
    """Test cases for _update_book_details method."""

    def test_update_description(self):
        """Test updating book description."""
        service = BookDataService()

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

        service._update_book_details(book, {"description": "New description"}, {})

        assert book.description == "New description"

    def test_update_description_from_dict(self):
        """Test updating description when it's a dict."""
        service = BookDataService()

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

        service._update_book_details(
            book, {"description": {"type": "text", "value": "Dict description"}}, {}
        )

        assert book.description == "Dict description"

    def test_does_not_overwrite_existing_description(self):
        """Test that existing description is not overwritten."""
        service = BookDataService()

        book = Book(
            google_id=None,
            openlib_key=None,
            title="Test",
            subtitle=None,
            authors=["Author"],
            subjects=[],
            description="Existing description",
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

        service._update_book_details(book, {"description": "New description"}, {})

        assert book.description == "Existing description"

    def test_update_subtitle(self):
        """Test updating book subtitle."""
        service = BookDataService()

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

        service._update_book_details(book, {"subtitle": "A subtitle"}, {})

        assert book.subtitle == "A subtitle"

    def test_update_published_date(self):
        """Test updating published date and year."""
        service = BookDataService()
        service.openlib_client = MagicMock()
        service.openlib_client._extract_year.return_value = 2023

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

        service._update_book_details(book, {"publish_date": "2023-01-15"}, {})

        assert book.published_date == "2023-01-15"

    def test_update_page_count_from_editions(self):
        """Test updating page count from editions data."""
        service = BookDataService()

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

        editions_data = {"entries": [{"number_of_pages": 350}]}
        service._update_book_details(book, {}, editions_data)

        assert book.page_count == 350


class TestAddOpenlibRatings:
    """Test cases for _add_openlib_ratings method."""

    def test_skips_if_google_rating_exists(self):
        """Test that OpenLibrary ratings are skipped if Google rating exists."""
        service = BookDataService()
        service.openlib_client = MagicMock()

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
            google_average_rating=4.5,
            google_ratings_count=1000,
            openlib_average_rating=None,
            openlib_ratings_count=None,
            openlib_edition_key=None,
        )

        service._add_openlib_ratings(book, "/works/OL123W")

        # Should not fetch ratings
        service.openlib_client.fetch_ratings.assert_not_called()
        assert book.openlib_average_rating is None

    def test_adds_openlib_ratings(self):
        """Test adding OpenLibrary ratings when no Google rating."""
        service = BookDataService()
        service.openlib_client = MagicMock()
        service.openlib_client.fetch_ratings.return_value = {
            "summary": {"average": 4.2, "count": 500}
        }

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

        service._add_openlib_ratings(book, "/works/OL123W")

        assert book.openlib_average_rating == 4.2
        assert book.openlib_ratings_count == 500


class TestEnrichWithOpenlibKey:
    """Test cases for enrich_with_openlib_key method."""

    def test_enrich_book_with_openlib_key(self):
        """Test enriching a book with OpenLibrary key."""
        service = BookDataService()
        service.openlib_client = MagicMock()
        service.openlib_client.fetch_work_or_edition_json.return_value = {
            "description": "Test description",
            "subjects": ["Fiction"],
        }
        service.openlib_client.fetch_editions.return_value = {"entries": []}
        service.openlib_client.extract_subject_slugs.return_value = ["fiction"]
        service.openlib_client.fetch_ratings.return_value = {
            "summary": {"average": 4.0, "count": 100}
        }

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

        result = service.enrich_with_openlib_key(book, "/works/OL123W")

        assert result.openlib_key == "/works/OL123W"
        assert result.description == "Test description"


class TestSearchBooks:
    """Test cases for search_books method."""

    def test_search_books_no_results_raises(self):
        """Test that search raises error when no results."""
        service = BookDataService()
        service.google_client = MagicMock()
        service.google_client.fetch_books.return_value = []

        with pytest.raises(ValueError, match="no results"):
            service.search_books("NonExistent", "Unknown")

    def test_search_books_success(self):
        """Test successful book search."""
        service = BookDataService()
        service.google_client = MagicMock()
        service.google_client.fetch_books.return_value = [
            Book(
                google_id="google123",
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
        ]
        service.openlib_client = MagicMock()
        service.openlib_client.find_openlib_work_key.return_value = "/works/OL123W"
        service.openlib_client.fetch_work_or_edition_json.return_value = {}
        service.openlib_client.fetch_editions.return_value = {"entries": []}
        service.openlib_client.fetch_ratings.return_value = {}
        service.openlib_client.extract_subject_slugs.return_value = []

        result = service.search_books("Test Book", "Test Author")

        assert result.title == "Test Book"
        assert result.openlib_key == "/works/OL123W"


class TestEnrichBookData:
    """Test cases for enrich_book_data method."""

    def test_enrich_missing_title_author_raises(self):
        """Test that enriching book without title/author raises error."""
        service = BookDataService()

        book = Book(
            google_id=None,
            openlib_key=None,
            title="",
            subtitle=None,
            authors=[],
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

        with pytest.raises(ValueError, match="missing title/author"):
            service.enrich_book_data(book)

    def test_enrich_with_optional_params(self):
        """Test enriching with optional title/author params."""
        service = BookDataService()
        service.openlib_client = MagicMock()
        service.openlib_client.find_openlib_work_key.return_value = "/works/OL123W"
        service.openlib_client.fetch_work_or_edition_json.return_value = {}
        service.openlib_client.fetch_editions.return_value = {"entries": []}
        service.openlib_client.fetch_ratings.return_value = {}
        service.openlib_client.extract_subject_slugs.return_value = []

        book = Book(
            google_id=None,
            openlib_key=None,
            title="",
            subtitle=None,
            authors=[],
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

        result = service.enrich_book_data(
            book, book_title="Test Title", book_author="Test Author"
        )

        assert result.title == "Test Title"
        assert result.authors == ["Test Author"]
