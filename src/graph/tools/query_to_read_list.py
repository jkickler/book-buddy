# Tool for searching the user's to-read list by mood, topic, or description
import json

from langchain.tools import tool
from loguru import logger

from src.vectorstore.to_read_list import search_to_read_list, to_read_list_exists


@tool("query_to_read_list")
def query_to_read_list_tool(query: str, k: int = 5) -> str:
    """
    Search the user's to-read list using a natural language query for mood, topic, or description.

    Use this tool when the user asks to search their to-read list, such as:
    - "Show me books in my to-read list that are romantic"
    - "What sci-fi books do I have saved?"
    - "Find cozy mysteries in my list"
    If the user asks "What should I read next?", you can query the to-read list first before generating new recommendations.
    """
    if not to_read_list_exists():
        return json.dumps({"status": "ok", "matches": []})

    results = search_to_read_list(query, k=k)
    matches = []
    for doc, score in results:
        metadata = doc.metadata or {}
        matches.append(
            {
                "score": score,
                "book": {
                    "title": metadata.get("title"),
                    "authors": metadata.get("authors"),
                    "description": metadata.get("description"),
                    "subjects": metadata.get("subjects"),
                    "isbn13": metadata.get("isbn13"),
                    "isbn10": metadata.get("isbn10"),
                    "published_year": metadata.get("published_year"),
                    "url": metadata.get("url"),
                    "reason": metadata.get("reason"),
                },
            }
        )

    return json.dumps({"status": "ok", "matches": matches})
