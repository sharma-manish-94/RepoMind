"""
Tests for the MCP server module.

Tests cover:
- Tool listing
- Tool execution
- Error handling

Author: RepoMind Team
"""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


class TestServerToolListing:
    """Tests for MCP tool listing."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_tools(self):
        """Test that list_tools returns available tools."""
        from repomind.server import list_tools

        tools = await list_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

        tool_names = [t.name for t in tools]
        assert "index_repo" in tool_names
        assert "semantic_grep" in tool_names
        assert "get_context" in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_has_descriptions(self):
        """Test that tools have descriptions."""
        from repomind.server import list_tools

        tools = await list_tools()

        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 0

    @pytest.mark.asyncio
    async def test_list_tools_has_input_schema(self):
        """Test that tools have input schemas."""
        from repomind.server import list_tools

        tools = await list_tools()

        for tool in tools:
            assert tool.inputSchema is not None
            assert "type" in tool.inputSchema


class TestServerToolExecution:
    """Tests for MCP tool execution."""

    @pytest.mark.asyncio
    @patch('repomind.server.index_repo')
    async def test_call_index_repo(self, mock_index_repo):
        """Test calling index_repo tool."""
        from repomind.server import call_tool

        mock_index_repo.return_value = {"status": "success", "chunks_stored": 10}

        result = await call_tool("index_repo", {"repo_path": "/test/path"})

        assert len(result) == 1
        assert result[0].type == "text"

        data = json.loads(result[0].text)
        assert data["status"] == "success"

    @pytest.mark.asyncio
    @patch('repomind.server.semantic_grep')
    async def test_call_semantic_grep(self, mock_grep):
        """Test calling semantic_grep tool."""
        from repomind.server import call_tool

        mock_grep.return_value = {"query": "test", "results": []}

        result = await call_tool("semantic_grep", {"query": "test query"})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "results" in data

    @pytest.mark.asyncio
    @patch('repomind.server.get_context')
    async def test_call_get_context(self, mock_context):
        """Test calling get_context tool."""
        from repomind.server import call_tool

        mock_context.return_value = {"found": True, "symbol": "test_func"}

        result = await call_tool("get_context", {"symbol_name": "test_func"})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["found"] is True

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        """Test calling unknown tool returns error."""
        from repomind.server import call_tool

        result = await call_tool("unknown_tool", {})

        assert len(result) == 1
        data = json.loads(result[0].text)
        # Should return error or handle gracefully
        assert "error" in data or "tool" in data

    @pytest.mark.asyncio
    @patch('repomind.server.index_repo')
    async def test_call_tool_with_error(self, mock_index_repo):
        """Test tool error handling."""
        from repomind.server import call_tool

        mock_index_repo.side_effect = Exception("Test error")

        result = await call_tool("index_repo", {"repo_path": "/test/path"})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data


class TestServerIntegration:
    """Integration tests for the server."""

    def test_app_exists(self):
        """Test that the MCP app is created."""
        from repomind.server import app

        assert app is not None

    def test_run_server_function_exists(self):
        """Test that run_server function exists."""
        from repomind.server import run_server

        assert callable(run_server)


class TestExecuteTool:
    """Tests for _execute_tool internal function."""

    @pytest.mark.asyncio
    @patch('repomind.server.index_repo')
    async def test_execute_index_repo(self, mock_index_repo):
        """Test executing index_repo."""
        from repomind.server import _execute_tool
        from repomind.constants import MCPToolName

        mock_index_repo.return_value = {"status": "success"}

        result = await _execute_tool(
            MCPToolName.INDEX_REPO.value,
            {"repo_path": "/path/to/repo"}
        )

        assert result["status"] == "success"

    @pytest.mark.asyncio
    @patch('repomind.server.semantic_grep')
    async def test_execute_semantic_grep(self, mock_grep):
        """Test executing semantic_grep."""
        from repomind.server import _execute_tool
        from repomind.constants import MCPToolName

        mock_grep.return_value = {"results": []}

        result = await _execute_tool(
            MCPToolName.SEMANTIC_GREP.value,
            {"query": "test"}
        )

        assert "results" in result


    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self):
        """Test executing unknown tool raises ValueError."""
        from repomind.server import _execute_tool

        with pytest.raises(ValueError):
            await _execute_tool("completely_unknown_tool", {})
