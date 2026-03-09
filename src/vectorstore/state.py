"""Vector store state management to avoid circular dependencies."""

from typing import Optional

from loguru import logger

# Global cache for the vector store instance
_vector_store: Optional[object] = None


def get_cached_vector_store() -> Optional[object]:
    """Get the cached vector store instance if it exists."""
    cached = _vector_store
    logger.info("Cache hit" if cached else "Cache miss")
    return cached


def set_cached_vector_store(vector_store: Optional[object]) -> None:
    """Set the cached vector store instance."""
    global _vector_store
    _vector_store = vector_store
    logger.info("Saved to cache" if vector_store else "Cleared cache")


def clear_cached_vector_store() -> None:
    """Clear the cached vector store instance."""
    global _vector_store
    _vector_store = None
    logger.info("Cleared cache")
