# Book data class with comprehensive metadata fields
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Book:
    google_id: Optional[str]
    openlib_key: Optional[str]
    title: str
    subtitle: Optional[str]
    authors: List[str]
    subjects: List[str]
    description: Optional[str]
    isbn13: Optional[str]
    isbn10: Optional[str]
    published_date: Optional[str]
    published_year: Optional[int]
    url: Optional[str]
    page_count: Optional[int]
    google_average_rating: Optional[float]
    google_ratings_count: Optional[int]
    openlib_average_rating: Optional[float]
    openlib_ratings_count: Optional[int]
    openlib_edition_key: Optional[str]
