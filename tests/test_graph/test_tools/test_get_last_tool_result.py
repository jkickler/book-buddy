"""Tests for get_last_tool_result tool."""

import pytest
from unittest.mock import MagicMock
from src.graph.tools.get_last_tool_result import get_last_tool_result_tool
from src.graph.schemas import GraphState


class TestGetLastToolResultTool:
    """Test cases for get_last_tool_result_tool."""

    def test_returns_last_tool_result(self):
        """Test returning last tool result when available."""
        state = GraphState()
        state["last_tool_result"] = {"status": "ok", "results": [{"book": "Test Book"}]}

        # The tool decorator wraps the function, we need to access the underlying function
        # or test the function that was decorated
        from src.graph.tools import get_last_tool_result as gltr_module

        result = gltr_module.get_last_tool_result_tool.func(state)

        assert result["status"] == "ok"
        assert result["results"][0]["book"] == "Test Book"

    def test_returns_empty_message_when_no_result(self):
        """Test returning empty message when no last result."""
        state = GraphState()
        # No last_tool_result set

        from src.graph.tools import get_last_tool_result as gltr_module

        result = gltr_module.get_last_tool_result_tool.func(state)

        assert result["status"] == "empty"
        assert "No previous tool results" in result["message"]

    def test_returns_empty_message_when_none(self):
        """Test returning empty message when last result is None."""
        state = GraphState()
        state["last_tool_result"] = None

        from src.graph.tools import get_last_tool_result as gltr_module

        result = gltr_module.get_last_tool_result_tool.func(state)

        assert result["status"] == "empty"
        assert "No previous tool results" in result["message"]
