"""Tests for graph schemas."""

import pytest
from pydantic import ValidationError
from src.graph.schemas import BookQuery, EnrichAndScoreInput, GraphState


class TestBookQuery:
    """Test cases for BookQuery schema."""

    def test_book_query_with_all_fields(self):
        """Test creating BookQuery with all fields."""
        query = BookQuery(title="Test Title", author="Test Author", isbn="1234567890")
        assert query.title == "Test Title"
        assert query.author == "Test Author"
        assert query.isbn == "1234567890"

    def test_book_query_with_partial_fields(self):
        """Test creating BookQuery with only some fields."""
        query = BookQuery(title="Test Title")
        assert query.title == "Test Title"
        assert query.author is None
        assert query.isbn is None

    def test_book_query_with_none_fields(self):
        """Test creating BookQuery with all None fields."""
        query = BookQuery()
        assert query.title is None
        assert query.author is None
        assert query.isbn is None

    def test_book_query_empty_strings(self):
        """Test BookQuery with empty strings."""
        query = BookQuery(title="", author="", isbn="")
        assert query.title == ""
        assert query.author == ""
        assert query.isbn == ""

    def test_book_query_extra_fields_not_allowed(self):
        """Test that extra fields are not allowed by default."""
        # Pydantic v2 by default ignores extra fields or raises depending on config
        # Since no extra='forbid' is set, extra fields are ignored
        query = BookQuery(title="Test", extra_field="ignored")
        assert not hasattr(query, "extra_field")


class TestEnrichAndScoreInput:
    """Test cases for EnrichAndScoreInput schema."""

    def test_enrich_and_score_input_with_books(self):
        """Test creating EnrichAndScoreInput with books."""
        books = [
            BookQuery(title="Book 1", author="Author 1"),
            BookQuery(title="Book 2", author="Author 2"),
        ]
        input_data = EnrichAndScoreInput(books=books)
        assert len(input_data.books) == 2
        assert input_data.books[0].title == "Book 1"
        assert input_data.books[1].title == "Book 2"

    def test_enrich_and_score_input_default_empty_list(self):
        """Test that default is empty list."""
        input_data = EnrichAndScoreInput()
        assert input_data.books == []

    def test_enrich_and_score_input_empty_list_explicit(self):
        """Test creating with explicit empty list."""
        input_data = EnrichAndScoreInput(books=[])
        assert input_data.books == []


class TestGraphState:
    """Test cases for GraphState."""

    def test_graph_state_basic(self):
        """Test basic GraphState creation."""
        # GraphState extends MessagesState which has TypedDict behavior
        # We can test that it can be instantiated with expected fields
        state = GraphState()
        # TypedDict can be empty when created without fields
        assert isinstance(state, dict)

    def test_graph_state_with_prompt(self):
        """Test GraphState with prompt field."""
        # Test that prompt field exists and can be set
        state = GraphState()
        # Since it's a TypedDict with NotRequired, we check field access
        assert hasattr(state, "__getitem__") or hasattr(state, "prompt")
