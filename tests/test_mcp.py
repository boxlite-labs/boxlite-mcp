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

    async def call_box(self, action: str, **kwargs) -> Any:
        """Call box management tool with an action."""
        return await self.call_tool("box", action, **kwargs)


@pytest.mark.asyncio
async def test_list_tools():
    """Test that server lists all tools correctly."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.session.list_tools()
        assert result.tools
        tool_names = [t.name for t in result.tools]
        assert "box" in tool_names
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


@pytest.mark.asyncio
async def test_sandbox_with_volumes():
    """Test sandbox with volume mounts."""
    import tempfile
    import os

    client = MCPTestClient()
    try:
        await client.connect()

        # Create a temporary directory with test file
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("Hello from host")

            # Start sandbox with volume mount
            result = await client.call_sandbox(
                "start",
                image="alpine",
                volumes=[[tmpdir, "/mnt/host"]]
            )
            assert result.content
            text = result.content[0].text
            assert "Sandbox started" in text
            sandbox_id = text.split(": ")[1].strip()

            # Read mounted file from sandbox
            result = await client.call_sandbox(
                "exec",
                sandbox_id=sandbox_id,
                command="cat /mnt/host/test.txt"
            )
            assert result.content
            assert "Hello from host" in result.content[0].text

            # Stop sandbox
            await client.call_sandbox("stop", sandbox_id=sandbox_id)
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_sandbox_with_readonly_volumes():
    """Test sandbox with read-only volume mounts."""
    import tempfile
    import os

    client = MCPTestClient()
    try:
        await client.connect()

        # Create a temporary directory with test file
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "readonly.txt")
            with open(test_file, "w") as f:
                f.write("Read-only file")

            # Start sandbox with read-only volume mount
            result = await client.call_sandbox(
                "start",
                image="alpine",
                volumes=[[tmpdir, "/mnt/host", True]]
            )
            assert result.content
            text = result.content[0].text
            assert "Sandbox started" in text
            sandbox_id = text.split(": ")[1].strip()

            # Read mounted file from sandbox
            result = await client.call_sandbox(
                "exec",
                sandbox_id=sandbox_id,
                command="cat /mnt/host/readonly.txt"
            )
            assert result.content
            assert "Read-only file" in result.content[0].text

            # Try to write to read-only volume (should fail)
            result = await client.call_sandbox(
                "exec",
                sandbox_id=sandbox_id,
                command="touch /mnt/host/newfile.txt"
            )
            assert result.content
            # Should have exit code indicating failure
            assert "Read-only file system" in result.content[0].text or "Permission denied" in result.content[0].text or "exit_code" in result.content[0].text

            # Stop sandbox
            await client.call_sandbox("stop", sandbox_id=sandbox_id)
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_sandbox_with_empty_volumes():
    """Test sandbox with empty volumes parameter (should work without error)."""
    client = MCPTestClient()
    try:
        await client.connect()

        # Start sandbox without volumes (None/empty)
        result = await client.call_sandbox("start", image="alpine")
        assert result.content
        text = result.content[0].text
        assert "Sandbox started" in text
        sandbox_id = text.split(": ")[1].strip()

        # Execute a simple command to verify it works
        result = await client.call_sandbox(
            "exec",
            sandbox_id=sandbox_id,
            command="echo 'test'"
        )
        assert result.content
        assert "test" in result.content[0].text

        # Stop sandbox
        await client.call_sandbox("stop", sandbox_id=sandbox_id)
    finally:
        await client.disconnect()


# ============================================================================
# Computer Tool Tests with Volumes
# ============================================================================

@pytest.mark.asyncio
async def test_computer_start_stop_with_empty_volumes():
    """Test computer start and stop with empty volumes parameter (should work without error)."""
    client = MCPTestClient()
    try:
        await client.connect()

        # Start computer without volumes
        result = await client.call_computer("start")
        assert result.content
        text = result.content[0].text
        assert "Computer started" in text

        # Extract computer_id from response
        lines = text.split("\n")
        computer_id = None
        for line in lines:
            if "Computer started with ID:" in line:
                computer_id = line.split(": ")[1].strip()
                break
        assert computer_id

        # Take a screenshot to verify it works
        result = await client.call_computer("screenshot", computer_id=computer_id)
        assert result.content
        content = result.content[0]
        assert hasattr(content, 'data')

        # Stop computer
        result = await client.call_computer("stop", computer_id=computer_id)
        assert result.content
        assert "stopped" in result.content[0].text.lower()
    finally:
        await client.disconnect()


# ============================================================================
# Box Management Tool Tests
# ============================================================================

@pytest.mark.asyncio
async def test_box_list():
    """Test box list action."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_box("list")
        assert result.content
        # May be empty or have boxes — just check no error
        text = result.content[0].text
        assert "Boxes" in text or "No boxes found" in text
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_box_metrics():
    """Test box runtime metrics."""
    client = MCPTestClient()
    try:
        await client.connect()
        result = await client.call_box("metrics")
        assert result.content
        text = result.content[0].text
        # Should have at least one metric field
        assert ":" in text
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_box_list_with_sandbox():
    """Test box list shows a running sandbox."""
    client = MCPTestClient()
    try:
        await client.connect()

        # Start a named sandbox
        result = await client.call_sandbox("start", image="alpine", name="test-list-box")
        text = result.content[0].text
        sandbox_id = text.split(": ")[1].strip()

        # List boxes — should see the sandbox
        result = await client.call_box("list")
        assert result.content
        text = result.content[0].text
        assert "Boxes" in text

        # Get specific box info
        result = await client.call_box("get", box_id=sandbox_id)
        assert result.content
        text = result.content[0].text
        assert sandbox_id in text or "id:" in text.lower()

        # Stop sandbox
        await client.call_sandbox("stop", sandbox_id=sandbox_id)
    finally:
        await client.disconnect()


# ============================================================================
# Enhanced Sandbox Tests
# ============================================================================

@pytest.mark.asyncio
async def test_sandbox_named():
    """Test sandbox with a name."""
    client = MCPTestClient()
    try:
        await client.connect()

        result = await client.call_sandbox("start", image="alpine", name="my-sandbox")
        assert result.content
        text = result.content[0].text
        assert "Sandbox started" in text
        sandbox_id = text.split(": ")[1].strip()

        # Run a command
        result = await client.call_sandbox(
            "exec", sandbox_id=sandbox_id, command="echo hello"
        )
        assert "hello" in result.content[0].text

        await client.call_sandbox("stop", sandbox_id=sandbox_id)
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_sandbox_with_env():
    """Test sandbox start with environment variables."""
    client = MCPTestClient()
    try:
        await client.connect()

        result = await client.call_sandbox(
            "start", image="alpine", env={"MY_VAR": "hello_world"}
        )
        text = result.content[0].text
        assert "Sandbox started" in text
        sandbox_id = text.split(": ")[1].strip()

        # Check env var is set
        result = await client.call_sandbox(
            "exec", sandbox_id=sandbox_id, command="echo $MY_VAR"
        )
        assert "hello_world" in result.content[0].text

        await client.call_sandbox("stop", sandbox_id=sandbox_id)
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_sandbox_copy_in_out():
    """Test sandbox copy_in and copy_out."""
    import tempfile
    import os

    client = MCPTestClient()
    try:
        await client.connect()

        result = await client.call_sandbox("start", image="alpine")
        text = result.content[0].text
        sandbox_id = text.split(": ")[1].strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            src_file = os.path.join(tmpdir, "input.txt")
            with open(src_file, "w") as f:
                f.write("copy test data")

            # Copy file into sandbox
            result = await client.call_sandbox(
                "copy_in",
                sandbox_id=sandbox_id,
                host_path=src_file,
                container_dest="/tmp/input.txt",
            )
            assert "copied" in result.content[0].text.lower()

            # Verify file exists in sandbox
            result = await client.call_sandbox(
                "exec", sandbox_id=sandbox_id, command="cat /tmp/input.txt"
            )
            assert "copy test data" in result.content[0].text

            # Copy file back out
            dest_file = os.path.join(tmpdir, "output.txt")
            result = await client.call_sandbox(
                "copy_out",
                sandbox_id=sandbox_id,
                container_src="/tmp/input.txt",
                host_dest=dest_file,
            )
            assert "copied" in result.content[0].text.lower()

        await client.call_sandbox("stop", sandbox_id=sandbox_id)
    finally:
        await client.disconnect()


# ============================================================================
# Enhanced Code Interpreter Tests
# ============================================================================

@pytest.mark.asyncio
async def test_code_interpreter_install():
    """Test code_interpreter install action."""
    client = MCPTestClient()
    try:
        await client.connect()

        result = await client.call_code_interpreter("start")
        text = result.content[0].text
        interpreter_id = text.split(": ")[1].strip()

        # Install a package
        result = await client.call_code_interpreter(
            "install", interpreter_id=interpreter_id, packages=["requests"]
        )
        assert result.content

        # Verify import works
        result = await client.call_code_interpreter(
            "run",
            interpreter_id=interpreter_id,
            code="import requests; print(requests.__version__)"
        )
        assert result.content

        await client.call_code_interpreter("stop", interpreter_id=interpreter_id)
    finally:
        await client.disconnect()


# ============================================================================
# Enhanced Browser Tests
# ============================================================================

@pytest.mark.asyncio
async def test_browser_start_with_playwright_endpoint():
    """Test browser start returns both CDP and Playwright endpoints."""
    client = MCPTestClient()
    try:
        await client.connect()

        result = await client.call_browser("start")
        assert result.content
        text = result.content[0].text
        assert "Browser started" in text
        assert "Endpoint:" in text

        lines = text.split("\n")
        browser_id = None
        for line in lines:
            if "Browser started with ID:" in line:
                browser_id = line.split(": ")[1].strip()
                break
        assert browser_id

        # Stop
        await client.call_browser("stop", browser_id=browser_id)
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_browser_run_command():
    """Test running shell command in browser container."""
    client = MCPTestClient()
    try:
        await client.connect()

        result = await client.call_browser("start")
        text = result.content[0].text
        browser_id = None
        for line in text.split("\n"):
            if "Browser started with ID:" in line:
                browser_id = line.split(": ")[1].strip()
                break
        assert browser_id

        # Run a command
        result = await client.call_browser(
            "run_command", browser_id=browser_id, command="echo 'hello from browser'"
        )
        assert result.content
        assert "hello from browser" in result.content[0].text

        await client.call_browser("stop", browser_id=browser_id)
    finally:
        await client.disconnect()


# ============================================================================
# Enhanced Computer Tests
# ============================================================================

@pytest.mark.asyncio
async def test_computer_run_command():
    """Test running shell command in computer container."""
    client = MCPTestClient()
    try:
        await client.connect()

        result = await client.call_computer("start")
        text = result.content[0].text
        computer_id = None
        for line in text.split("\n"):
            if "Computer started with ID:" in line:
                computer_id = line.split(": ")[1].strip()
                break
        assert computer_id

        # Run a command
        result = await client.call_computer(
            "run_command", computer_id=computer_id, command="echo 'hello from computer'"
        )
        assert result.content
        assert "hello from computer" in result.content[0].text

        await client.call_computer("stop", computer_id=computer_id)
    finally:
        await client.disconnect()


# ============================================================================
# Box Reuse (reuse_existing) Tests
# ============================================================================

@pytest.mark.asyncio
async def test_sandbox_reuse_by_name():
    """Test sandbox reuse: create with auto_remove=False, stop, restart with reuse_existing=True."""
    client = MCPTestClient()
    try:
        await client.connect()

        # Create a named sandbox with auto_remove=False so it persists
        result = await client.call_sandbox(
            "start", image="alpine", name="reuse-test", auto_remove=False
        )
        text = result.content[0].text
        assert "Sandbox started" in text
        sandbox_id_1 = text.split(": ")[1].split("\n")[0].strip()

        # Write a marker file so we can verify it's the same box
        result = await client.call_sandbox(
            "exec", sandbox_id=sandbox_id_1,
            command="echo 'marker-data' > /tmp/reuse-marker.txt"
        )

        # Stop the sandbox (but it persists because auto_remove=False)
        await client.call_sandbox("stop", sandbox_id=sandbox_id_1)

        # Restart with the same name and reuse_existing=True
        result = await client.call_sandbox(
            "start", image="alpine", name="reuse-test",
            reuse_existing=True, auto_remove=False
        )
        text = result.content[0].text
        assert "Sandbox started" in text
        sandbox_id_2 = text.split(": ")[1].split("\n")[0].strip()

        # Should be the same box (same ID)
        assert sandbox_id_1 == sandbox_id_2

        # The response should indicate it was reused (created=False)
        assert "Created: False" in text

        # Verify marker file still exists
        result = await client.call_sandbox(
            "exec", sandbox_id=sandbox_id_2,
            command="cat /tmp/reuse-marker.txt"
        )
        assert "marker-data" in result.content[0].text

        # Cleanup: force remove
        await client.call_box("remove", box_id=sandbox_id_2, force=True)
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_sandbox_start_created_field():
    """Test that sandbox start returns created=True for new boxes."""
    client = MCPTestClient()
    try:
        await client.connect()

        result = await client.call_sandbox("start", image="alpine")
        text = result.content[0].text
        assert "Sandbox started" in text
        assert "Created: True" in text
        sandbox_id = text.split(": ")[1].split("\n")[0].strip()

        await client.call_sandbox("stop", sandbox_id=sandbox_id)
    finally:
        await client.disconnect()
