# Pydantic schemas for graph state and book query validation
from typing import NotRequired

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class BookQuery(BaseModel):
    """Represents a book query with optional title, author, and ISBN."""

    title: str | None = Field(default=None, description="Book title")
    author: str | None = Field(default=None, description="Book author")
    isbn: str | None = Field(default=None, description="ISBN-10 or ISBN-13")


class EnrichAndScoreInput(BaseModel):
    """Input schema for enriching and scoring a list of books."""

    books: list[BookQuery] = Field(
        default_factory=list,
        description="List of books to enrich and score",
    )


class GraphState(MessagesState):
    """State for graph execution, extending MessagesState with custom fields."""

    prompt: NotRequired[str]
    summary: NotRequired[str]
    is_blocked: NotRequired[bool]
    last_recommendations: NotRequired[list[dict]]
    last_checked_books: NotRequired[list[dict]]
