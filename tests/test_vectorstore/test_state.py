"""Tests for vector store state management."""

import pytest
from src.vectorstore.state import (
    get_cached_vector_store,
    set_cached_vector_store,
    clear_cached_vector_store,
)


class TestVectorStoreState:
    """Test cases for vector store state management."""

    def test_get_cached_vector_store_initially_none(self):
        """Test that cache is initially None."""
        clear_cached_vector_store()
        assert get_cached_vector_store() is None

    def test_set_cached_vector_store(self):
        """Test setting the cached vector store."""
        mock_store = {"name": "test_store"}
        set_cached_vector_store(mock_store)
        assert get_cached_vector_store() == mock_store

    def test_clear_cached_vector_store(self):
        """Test clearing the cached vector store."""
        mock_store = {"name": "test_store"}
        set_cached_vector_store(mock_store)
        clear_cached_vector_store()
        assert get_cached_vector_store() is None

    def test_set_cached_vector_store_to_none(self):
        """Test setting cache to None explicitly."""
        mock_store = {"name": "test_store"}
        set_cached_vector_store(mock_store)
        set_cached_vector_store(None)
        assert get_cached_vector_store() is None

    def test_cache_isolation_between_tests(self):
        """Test that cache state is properly isolated."""
        # This test relies on clear_cached_vector_store in setup
        set_cached_vector_store("store1")
        assert get_cached_vector_store() == "store1"

        clear_cached_vector_store()
        set_cached_vector_store("store2")
        assert get_cached_vector_store() == "store2"
