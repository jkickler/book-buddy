"""Shared fixtures and test configuration for all tests."""

import pytest
from unittest.mock import MagicMock, Mock
from src.core.book import Book


@pytest.fixture
def sample_book() -> Book:
    """Create a sample Book instance for testing."""
    return Book(
        google_id="google123",
        openlib_key="/works/OL123W",
        title="Test Book Title",
        subtitle="A Test Subtitle",
        authors=["John Doe", "Jane Smith"],
        subjects=["fiction", "science_fiction", "adventure"],
        description="A test book description",
        isbn13="9781234567890",
        isbn10="1234567890",
        published_date="2023-01-15",
        published_year=2023,
        url="https://books.google.com/test",
        page_count=300,
        google_average_rating=4.5,
        google_ratings_count=1000,
        openlib_average_rating=4.2,
        openlib_ratings_count=500,
        openlib_edition_key="/books/OL456M",
    )


@pytest.fixture
def minimal_book() -> Book:
    """Create a minimal Book instance with only required fields."""
    return Book(
        google_id=None,
        openlib_key=None,
        title="Minimal Book",
        subtitle=None,
        authors=["Unknown Author"],
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


@pytest.fixture
def mock_vector_store():
    """Create a mock Chroma vector store."""
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "metadatas": [
            {
                "title": "Existing Book",
                "authors": ["Existing Author"],
                "subjects": ["fiction"],
                "isbn13": "9780000000001",
                "isbn10": "0000000001",
                "google_id": "google_existing",
                "openlib_key": "/works/OL999W",
                "goodreads_user_rating": 5.0,
            }
        ]
    }
    mock_store.similarity_search_with_score.return_value = [
        (MagicMock(metadata={"title": "Similar Book", "authors": ["Author"]}), 0.5)
    ]
    return mock_store


@pytest.fixture
def mock_chroma_class(monkeypatch):
    """Mock the Chroma class from langchain_chroma."""
    mock_chroma = MagicMock()

    def mock_from_documents(*args, **kwargs):
        instance = MagicMock()
        instance.persist = MagicMock()
        return instance

    mock_chroma.from_documents = mock_from_documents
    mock_chroma.return_value = MagicMock()

    monkeypatch.setattr("langchain_chroma.Chroma", mock_chroma)
    return mock_chroma


@pytest.fixture
def mock_requests_session(monkeypatch):
    """Mock requests.Session for API client tests."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_session.get.return_value = mock_response

    mock_session_class = MagicMock(return_value=mock_session)
    monkeypatch.setattr("requests.Session", mock_session_class)

    return mock_session


@pytest.fixture
def mock_openai_embeddings(monkeypatch):
    """Mock OpenAIEmbeddings to avoid API calls."""
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 3072]
    mock_embeddings.embed_query.return_value = [0.1] * 3072

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings", MagicMock(return_value=mock_embeddings)
    )
    return mock_embeddings


@pytest.fixture
def mock_streamlit(monkeypatch):
    """Mock Streamlit session state and components."""
    mock_st = MagicMock()
    mock_st.session_state = {
        "chat_msgs": [],
        "thread_id": "test-thread-id",
        "vector_store": None,
        "vector_store_ready": False,
        "upload_in_progress": False,
        "upload_error": None,
        "compiled_graph": None,
    }
    monkeypatch.setattr("streamlit.session_state", mock_st.session_state)
    monkeypatch.setattr("streamlit", mock_st)
    return mock_st
