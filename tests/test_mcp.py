"""
BoxLite MCP Server Tests

Tests all tools exposed by the MCP server:
- computer: Desktop control
- browser: CDP browser
- code_interpreter: Python execution
- sandbox: Generic container

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

    async def call_tool(self, tool_name: str, action: str, **kwargs) -> Any:
        """Call a tool with an action."""
        arguments = {"action": action, **kwargs}
        return await self.session.call_tool(tool_name, arguments)

    async def call_computer(self, action: str, **kwargs) -> Any:
        """Call computer tool with an action."""
        return await self.call_tool("computer", action, **kwargs)

    async def call_browser(self, action: str, **kwargs) -> Any:
        """Call browser tool with an action."""
        return await self.call_tool("browser", action, **kwargs)

    async def call_code_interpreter(self, action: str, **kwargs) -> Any:
        """Call code_interpreter tool with an action."""
        return await self.call_tool("code_interpreter", action, **kwargs)

    async def call_sandbox(self, action: str, **kwargs) -> Any:
        """Call sandbox tool with an action."""
        return await self.call_tool("sandbox", action, **kwargs)


@pytest.mark.asyncio
async def test_list_tools():
    """Test that server lists all tools correctly."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.session.list_tools()
        assert result.tools
        tool_names = [t.name for t in result.tools]
        assert "computer" in tool_names
        assert "browser" in tool_names
        assert "code_interpreter" in tool_names
        assert "sandbox" in tool_names
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


# ============================================================================
# Browser Tool Tests
# ============================================================================

@pytest.mark.asyncio
async def test_browser_start_stop():
    """Test browser start and stop actions."""
    client = MCPTestClient()
    try:
        await client.connect()

        # Start browser
        result = await client.call_browser("start")
        assert result.content
        text = result.content[0].text
        assert "Browser started" in text
        assert "Endpoint:" in text

        # Extract browser_id from response
        lines = text.split("\n")
        browser_id = None
        for line in lines:
            if "Browser started with ID:" in line:
                browser_id = line.split(": ")[1].strip()
                break
        assert browser_id

        # Stop browser
        result = await client.call_browser("stop", browser_id=browser_id)
        assert result.content
        assert "stopped" in result.content[0].text.lower()
    finally:
        await client.disconnect()


# ============================================================================
# Code Interpreter Tool Tests
# ============================================================================

@pytest.mark.asyncio
async def test_code_interpreter_start_stop():
    """Test code_interpreter start and stop actions."""
    client = MCPTestClient()
    try:
        await client.connect()

        # Start interpreter
        result = await client.call_code_interpreter("start")
        assert result.content
        text = result.content[0].text
        assert "Code interpreter started" in text

        # Extract interpreter_id from response
        interpreter_id = text.split(": ")[1].strip()
        assert interpreter_id

        # Stop interpreter
        result = await client.call_code_interpreter("stop", interpreter_id=interpreter_id)
        assert result.content
        assert "stopped" in result.content[0].text.lower()
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_code_interpreter_run():
    """Test code_interpreter run action."""
    client = MCPTestClient()
    try:
        await client.connect()

        # Start interpreter
        result = await client.call_code_interpreter("start")
        text = result.content[0].text
        interpreter_id = text.split(": ")[1].strip()

        # Run Python code
        result = await client.call_code_interpreter(
            "run",
            interpreter_id=interpreter_id,
            code="print(2 + 2)"
        )
        assert result.content
        assert "4" in result.content[0].text

        # Run more complex code
        result = await client.call_code_interpreter(
            "run",
            interpreter_id=interpreter_id,
            code="import sys; print(sys.version_info.major)"
        )
        assert result.content
        assert "3" in result.content[0].text

        # Stop interpreter
        await client.call_code_interpreter("stop", interpreter_id=interpreter_id)
    finally:
        await client.disconnect()


# ============================================================================
# Sandbox Tool Tests
# ============================================================================

@pytest.mark.asyncio
async def test_sandbox_start_stop():
    """Test sandbox start and stop actions."""
    client = MCPTestClient()
    try:
        await client.connect()

        # Start sandbox with alpine image
        result = await client.call_sandbox("start", image="alpine")
        assert result.content
        text = result.content[0].text
        assert "Sandbox started" in text

        # Extract sandbox_id from response
        sandbox_id = text.split(": ")[1].strip()
        assert sandbox_id

        # Stop sandbox
        result = await client.call_sandbox("stop", sandbox_id=sandbox_id)
        assert result.content
        assert "stopped" in result.content[0].text.lower()
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_sandbox_exec():
    """Test sandbox exec action."""
    client = MCPTestClient()
    try:
        await client.connect()

        # Start sandbox
        result = await client.call_sandbox("start", image="alpine")
        text = result.content[0].text
        sandbox_id = text.split(": ")[1].strip()

        # Execute command
        result = await client.call_sandbox(
            "exec",
            sandbox_id=sandbox_id,
            command="echo 'Hello from sandbox'"
        )
        assert result.content
        assert "Hello from sandbox" in result.content[0].text

        # Execute another command
        result = await client.call_sandbox(
            "exec",
            sandbox_id=sandbox_id,
            command="cat /etc/os-release | head -1"
        )
        assert result.content
        assert "Alpine" in result.content[0].text or "alpine" in result.content[0].text.lower()

        # Stop sandbox
        await client.call_sandbox("stop", sandbox_id=sandbox_id)
    finally:
        await client.disconnect()
