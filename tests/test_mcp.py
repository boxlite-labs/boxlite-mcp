"""
BoxLite MCP Server Tests

Tests all computer control actions exposed by the MCP server.
Requires boxlite to be installed and working.
"""

import pytest
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPTestClient:
    """Test client that manages MCP session lifecycle."""

    def __init__(self):
        self.session = None
        self._read = None
        self._write = None
        self._stdio_cm = None
        self._session_cm = None

    async def connect(self):
        """Connect to MCP server."""
        server_path = Path(__file__).parent.parent / "server.py"
        server_params = StdioServerParameters(
            command="python",
            args=["-u", str(server_path)],
        )

        self._stdio_cm = stdio_client(server_params)
        self._read, self._write = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(self._read, self._write)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()

    async def disconnect(self):
        """Disconnect from MCP server."""
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._stdio_cm:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass

    async def call_computer(self, action: str, **kwargs) -> Any:
        """Call computer tool with an action."""
        arguments = {"action": action, **kwargs}
        return await self.session.call_tool("computer", arguments)


@pytest.mark.asyncio
async def test_list_tools():
    """Test that server lists tools correctly."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.session.list_tools()
        assert result.tools
        tool_names = [t.name for t in result.tools]
        assert "computer" in tool_names
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_screenshot():
    """Test screenshot action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer("screenshot")
        assert result.content
        content = result.content[0]
        assert hasattr(content, 'data') or hasattr(content, 'text')
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_cursor_position():
    """Test cursor_position action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer("cursor_position")
        assert result.content
        assert "Cursor position" in result.content[0].text
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_mouse_move():
    """Test mouse_move action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer("mouse_move", coordinate=[512, 384])
        assert result.content
        assert "Moved cursor" in result.content[0].text
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_left_click():
    """Test left_click action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer("left_click", coordinate=[512, 384])
        assert result.content
        assert "click" in result.content[0].text.lower()
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_right_click():
    """Test right_click action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer("right_click")
        assert result.content
        assert "click" in result.content[0].text.lower()
        # Close any context menu
        await client.call_computer("key", key="Escape")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_middle_click():
    """Test middle_click action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer("middle_click", coordinate=[512, 384])
        assert result.content
        assert "click" in result.content[0].text.lower()
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_double_click():
    """Test double_click action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer("double_click", coordinate=[512, 384])
        assert result.content
        assert "double" in result.content[0].text.lower()
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_triple_click():
    """Test triple_click action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer("triple_click", coordinate=[512, 384])
        assert result.content
        assert "triple" in result.content[0].text.lower()
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_type():
    """Test type action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer("type", text="Hello")
        assert result.content
        assert "Typed" in result.content[0].text
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_key():
    """Test key action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer("key", key="Escape")
        assert result.content
        assert "Pressed key" in result.content[0].text
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_scroll():
    """Test scroll action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer(
            "scroll",
            coordinate=[512, 384],
            scroll_direction="down",
            scroll_amount=3
        )
        assert result.content
        assert "Scrolled" in result.content[0].text
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_left_click_drag():
    """Test left_click_drag action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_computer(
            "left_click_drag",
            start_coordinate=[300, 300],
            end_coordinate=[400, 400]
        )
        assert result.content
        assert "Dragged" in result.content[0].text
    finally:
        await client.disconnect()
