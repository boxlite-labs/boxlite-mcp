#!/usr/bin/env python3
"""
BoxLite MCP Server - Isolated Sandbox Environments

Provides multiple sandbox tools:
- computer: Full desktop environment (Anthropic computer use API compatible)
- browser: Browser with CDP endpoint for automation
- code_interpreter: Python code execution sandbox
- sandbox: Generic container for shell commands
"""
import logging
import random
import socket
import sys
from typing import Optional

import anyio
import boxlite
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import ImageContent, TextContent, Tool

# Configure logging to stderr only (to avoid interfering with MCP stdio protocol)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("boxlite-mcp")


def find_available_port(start: int = 10000, end: int = 65535) -> int:
    """Find an available port by attempting to bind to it.

    Args:
        start: Start of port range to search (default: 10000)
        end: End of port range to search (default: 65535)

    Returns:
        An available port number

    Raises:
        RuntimeError: If no available port is found in the range
    """
    # Try random ports within the range
    ports = list(range(start, end + 1))
    random.shuffle(ports)

    for port in ports[:100]:  # Try up to 100 random ports
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue

    raise RuntimeError(f"Could not find an available port in range {start}-{end}")


class BrowserToolHandler:
    """Handler for browser tool actions."""

    def __init__(self):
        self._browsers: dict[str, boxlite.BrowserBox] = {}
        self._lock = anyio.Lock()

    def _get_browser(self, browser_id: str) -> boxlite.BrowserBox:
        if browser_id not in self._browsers:
            raise RuntimeError(f"Browser '{browser_id}' not found. Use 'start' action first.")
        return self._browsers[browser_id]

    async def start(self, **kwargs) -> dict:
        """Start a new browser instance."""
        async with self._lock:
            try:
                logger.info("Creating BrowserBox...")
                browser = boxlite.BrowserBox()
                await browser.__aenter__()
                browser_id = browser.id
                endpoint = browser.endpoint()
                logger.info(f"BrowserBox {browser_id} created. Endpoint: {endpoint}")
                self._browsers[browser_id] = browser
                return {"browser_id": browser_id, "endpoint": endpoint}
            except BaseException as e:
                error_msg = f"Failed to start BrowserBox: {e}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg)

    async def stop(self, browser_id: str, **kwargs) -> dict:
        """Stop a browser instance."""
        async with self._lock:
            if browser_id not in self._browsers:
                raise RuntimeError(f"Browser '{browser_id}' not found")
            browser = self._browsers[browser_id]
            logger.info(f"Shutting down BrowserBox {browser_id}...")
            try:
                await browser.__aexit__(None, None, None)
                logger.info(f"BrowserBox {browser_id} shut down successfully")
            except BaseException as e:
                logger.error(f"Error during BrowserBox {browser_id} cleanup: {e}", exc_info=True)
            finally:
                del self._browsers[browser_id]
            return {"success": True}

    async def shutdown_all(self):
        """Cleanup all browser instances."""
        async with self._lock:
            for browser_id, browser in list(self._browsers.items()):
                logger.info(f"Shutting down BrowserBox {browser_id}...")
                try:
                    await browser.__aexit__(None, None, None)
                except BaseException as e:
                    logger.error(f"Error during BrowserBox {browser_id} cleanup: {e}", exc_info=True)
            self._browsers.clear()


class CodeInterpreterToolHandler:
    """Handler for code_interpreter tool actions."""

    def __init__(self):
        self._interpreters: dict[str, boxlite.CodeBox] = {}
        self._lock = anyio.Lock()

    def _get_interpreter(self, interpreter_id: str) -> boxlite.CodeBox:
        if interpreter_id not in self._interpreters:
            raise RuntimeError(f"Interpreter '{interpreter_id}' not found. Use 'start' action first.")
        return self._interpreters[interpreter_id]

    async def start(self, **kwargs) -> dict:
        """Start a new code interpreter instance."""
        async with self._lock:
            try:
                logger.info("Creating CodeBox...")
                interpreter = boxlite.CodeBox()
                await interpreter.__aenter__()
                interpreter_id = interpreter.id
                logger.info(f"CodeBox {interpreter_id} created")
                self._interpreters[interpreter_id] = interpreter
                return {"interpreter_id": interpreter_id}
            except BaseException as e:
                error_msg = f"Failed to start CodeBox: {e}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg)

    async def stop(self, interpreter_id: str, **kwargs) -> dict:
        """Stop a code interpreter instance."""
        async with self._lock:
            if interpreter_id not in self._interpreters:
                raise RuntimeError(f"Interpreter '{interpreter_id}' not found")
            interpreter = self._interpreters[interpreter_id]
            logger.info(f"Shutting down CodeBox {interpreter_id}...")
            try:
                await interpreter.__aexit__(None, None, None)
                logger.info(f"CodeBox {interpreter_id} shut down successfully")
            except BaseException as e:
                logger.error(f"Error during CodeBox {interpreter_id} cleanup: {e}", exc_info=True)
            finally:
                del self._interpreters[interpreter_id]
            return {"success": True}

    async def run(self, interpreter_id: str, code: str, **kwargs) -> dict:
        """Execute Python code."""
        interpreter = self._get_interpreter(interpreter_id)
        output = await interpreter.run(code)
        return {"output": output}

    async def shutdown_all(self):
        """Cleanup all interpreter instances."""
        async with self._lock:
            for interpreter_id, interpreter in list(self._interpreters.items()):
                logger.info(f"Shutting down CodeBox {interpreter_id}...")
                try:
                    await interpreter.__aexit__(None, None, None)
                except BaseException as e:
                    logger.error(f"Error during CodeBox {interpreter_id} cleanup: {e}", exc_info=True)
            self._interpreters.clear()


class SandboxToolHandler:
    """Handler for sandbox tool actions."""

    def __init__(self):
        self._sandboxes: dict[str, boxlite.SimpleBox] = {}
        self._lock = anyio.Lock()

    def _get_sandbox(self, sandbox_id: str) -> boxlite.SimpleBox:
        if sandbox_id not in self._sandboxes:
            raise RuntimeError(f"Sandbox '{sandbox_id}' not found. Use 'start' action first.")
        return self._sandboxes[sandbox_id]

    async def start(self, image: str, volumes: Optional[list] = None, **kwargs) -> dict:
        """Start a new sandbox instance."""
        async with self._lock:
            try:
                logger.info(f"Creating SimpleBox with image '{image}'...")
                # Convert volume list format to tuples for boxlite
                boxlite_volumes = None
                if volumes:
                    boxlite_volumes = [tuple(v) for v in volumes]
                    logger.info(f"Creating SimpleBox with volumes: {boxlite_volumes}")

                sandbox = boxlite.SimpleBox(image=image, volumes=boxlite_volumes)
                await sandbox.__aenter__()
                sandbox_id = sandbox.id
                logger.info(f"SimpleBox {sandbox_id} created")
                self._sandboxes[sandbox_id] = sandbox
                return {"sandbox_id": sandbox_id}
            except BaseException as e:
                error_msg = f"Failed to start SimpleBox: {e}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg)

    async def stop(self, sandbox_id: str, **kwargs) -> dict:
        """Stop a sandbox instance."""
        async with self._lock:
            if sandbox_id not in self._sandboxes:
                raise RuntimeError(f"Sandbox '{sandbox_id}' not found")
            sandbox = self._sandboxes[sandbox_id]
            logger.info(f"Shutting down SimpleBox {sandbox_id}...")
            try:
                await sandbox.__aexit__(None, None, None)
                logger.info(f"SimpleBox {sandbox_id} shut down successfully")
            except BaseException as e:
                logger.error(f"Error during SimpleBox {sandbox_id} cleanup: {e}", exc_info=True)
            finally:
                del self._sandboxes[sandbox_id]
            return {"success": True}

    async def exec(self, sandbox_id: str, command: str, **kwargs) -> dict:
        """Execute a shell command."""
        sandbox = self._get_sandbox(sandbox_id)
        result = await sandbox.exec("sh", "-c", command)
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    async def shutdown_all(self):
        """Cleanup all sandbox instances."""
        async with self._lock:
            for sandbox_id, sandbox in list(self._sandboxes.items()):
                logger.info(f"Shutting down SimpleBox {sandbox_id}...")
                try:
                    await sandbox.__aexit__(None, None, None)
                except BaseException as e:
                    logger.error(f"Error during SimpleBox {sandbox_id} cleanup: {e}", exc_info=True)
            self._sandboxes.clear()


class ComputerToolHandler:
    """
    Handler for computer use actions.

    Manages multiple ComputerBox instances and delegates MCP tool calls to their APIs.
    """

    def __init__(self, memory_mib: int = 4096, cpus: int = 4):
        self._memory_mib = memory_mib
        self._cpus = cpus
        self._computers: dict[str, boxlite.ComputerBox] = {}
        self._lock = anyio.Lock()

    def _get_computer(self, computer_id: str) -> boxlite.ComputerBox:
        """Get a ComputerBox by ID."""
        if computer_id not in self._computers:
            raise RuntimeError(f"Computer '{computer_id}' not found. Use 'start' action first.")
        return self._computers[computer_id]

    async def start(self, volumes: Optional[list] = None, **kwargs) -> dict:
        """Start a new computer instance and return its ID."""
        async with self._lock:
            try:
                # Find available ports for HTTP and HTTPS
                gui_http_port = find_available_port()
                gui_https_port = find_available_port()
                logger.info(f"Creating ComputerBox with ports HTTP={gui_http_port}, HTTPS={gui_https_port}...")

                # Convert volume list format to tuples for boxlite
                boxlite_volumes = None
                if volumes:
                    boxlite_volumes = [tuple(v) for v in volumes]
                    logger.info(f"Creating ComputerBox with volumes: {boxlite_volumes}")

                computer = boxlite.ComputerBox(
                    cpu=self._cpus,
                    memory=self._memory_mib,
                    gui_http_port=gui_http_port,
                    gui_https_port=gui_https_port,
                    volumes=boxlite_volumes,
                )
                await computer.__aenter__()
                computer_id = computer.id
                logger.info(f"ComputerBox {computer_id} created with ports HTTP={gui_http_port}, HTTPS={gui_https_port}")

                # Wait for desktop to be ready
                logger.info(f"Waiting for desktop {computer_id} to become ready...")
                await computer.wait_until_ready()
                logger.info(f"Desktop {computer_id} is ready")

                self._computers[computer_id] = computer
                return {
                    "computer_id": computer_id,
                    "gui_http_port": gui_http_port,
                    "gui_https_port": gui_https_port,
                }

            except BaseException as e:
                error_msg = f"Failed to start ComputerBox: {e}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg)

    async def stop(self, computer_id: str, **kwargs) -> dict:
        """Stop and cleanup a specific computer instance."""
        async with self._lock:
            if computer_id not in self._computers:
                raise RuntimeError(f"Computer '{computer_id}' not found")

            computer = self._computers[computer_id]
            logger.info(f"Shutting down ComputerBox {computer_id}...")
            try:
                await computer.__aexit__(None, None, None)
                logger.info(f"ComputerBox {computer_id} shut down successfully")
            except BaseException as e:
                logger.error(f"Error during ComputerBox {computer_id} cleanup: {e}", exc_info=True)
            finally:
                del self._computers[computer_id]

            return {"success": True}

    async def shutdown_all(self):
        """Cleanup all ComputerBox instances."""
        async with self._lock:
            for computer_id, computer in list(self._computers.items()):
                logger.info(f"Shutting down ComputerBox {computer_id}...")
                try:
                    await computer.__aexit__(None, None, None)
                    logger.info(f"ComputerBox {computer_id} shut down successfully")
                except BaseException as e:
                    logger.error(
                        f"Error during ComputerBox {computer_id} cleanup: {e}",
                        exc_info=True,
                    )
            self._computers.clear()

    # Action handlers - delegation to ComputerBox API

    async def screenshot(self, computer_id: str, **kwargs) -> dict:
        """Capture screenshot."""
        computer = self._get_computer(computer_id)
        result = await computer.screenshot()
        return {
            "image_data": result["data"],
            "width": result["width"],
            "height": result["height"],
        }

    async def mouse_move(self, computer_id: str, coordinate: list[int], **kwargs) -> dict:
        """Move mouse to coordinates."""
        computer = self._get_computer(computer_id)
        x, y = coordinate
        await computer.mouse_move(x, y)
        return {"success": True}

    async def left_click(self, computer_id: str, coordinate: Optional[list[int]] = None,
                         **kwargs) -> dict:
        """Click left mouse button."""
        computer = self._get_computer(computer_id)
        if coordinate:
            x, y = coordinate
            await computer.mouse_move(x, y)
        await computer.left_click()
        return {"success": True}

    async def right_click(self, computer_id: str, coordinate: Optional[list[int]] = None,
                          **kwargs) -> dict:
        """Click right mouse button."""
        computer = self._get_computer(computer_id)
        if coordinate:
            x, y = coordinate
            await computer.mouse_move(x, y)
        await computer.right_click()
        return {"success": True}

    async def middle_click(self, computer_id: str, coordinate: Optional[list[int]] = None,
                           **kwargs) -> dict:
        """Click middle mouse button."""
        computer = self._get_computer(computer_id)
        if coordinate:
            x, y = coordinate
            await computer.mouse_move(x, y)
        await computer.middle_click()
        return {"success": True}

    async def double_click(self, computer_id: str, coordinate: Optional[list[int]] = None,
                           **kwargs) -> dict:
        """Double click left mouse button."""
        computer = self._get_computer(computer_id)
        if coordinate:
            x, y = coordinate
            await computer.mouse_move(x, y)
        await computer.double_click()
        return {"success": True}

    async def triple_click(self, computer_id: str, coordinate: Optional[list[int]] = None,
                           **kwargs) -> dict:
        """Triple click left mouse button."""
        computer = self._get_computer(computer_id)
        if coordinate:
            x, y = coordinate
            await computer.mouse_move(x, y)
        await computer.triple_click()
        return {"success": True}

    async def left_click_drag(self, computer_id: str, start_coordinate: list[int],
                              end_coordinate: list[int], **kwargs) -> dict:
        """Drag from start to end coordinates."""
        computer = self._get_computer(computer_id)
        start_x, start_y = start_coordinate
        end_x, end_y = end_coordinate
        await computer.left_click_drag(start_x, start_y, end_x, end_y)
        return {"success": True}

    async def type(self, computer_id: str, text: str, **kwargs) -> dict:
        """Type text."""
        computer = self._get_computer(computer_id)
        await computer.type(text)
        return {"success": True}

    async def key(self, computer_id: str, key: str, **kwargs) -> dict:
        """Press key or key combination."""
        computer = self._get_computer(computer_id)
        await computer.key(key)
        return {"success": True}

    async def scroll(self, computer_id: str, coordinate: list[int], scroll_direction: str,
                     scroll_amount: int = 3, **kwargs) -> dict:
        """Scroll at coordinates."""
        computer = self._get_computer(computer_id)
        x, y = coordinate
        await computer.scroll(x, y, scroll_direction, scroll_amount)
        return {"success": True}

    async def cursor_position(self, computer_id: str, **kwargs) -> dict:
        """Get current cursor position."""
        computer = self._get_computer(computer_id)
        x, y = await computer.cursor_position()
        return {"x": x, "y": y}


async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting BoxLite MCP Server")

    # Create handlers and server
    computer_handler = ComputerToolHandler()
    browser_handler = BrowserToolHandler()
    code_handler = CodeInterpreterToolHandler()
    sandbox_handler = SandboxToolHandler()
    server = Server("boxlite")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            # Browser tool
            Tool(
                name="browser",
                description="""Start a browser with Chrome DevTools Protocol (CDP) endpoint.

Use this to get a browser endpoint that can be connected to via Puppeteer, Playwright, or Selenium.

Actions:
- start: Start browser instance (returns browser_id and endpoint URL)
- stop: Stop browser instance (requires browser_id)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["start", "stop"],
                            "description": "The action to perform",
                        },
                        "browser_id": {
                            "type": "string",
                            "description": "Browser instance ID (required for 'stop')",
                        },
                    },
                    "required": ["action"],
                },
            ),

            # Code interpreter tool
            Tool(
                name="code_interpreter",
                description="""Execute Python code in an isolated sandbox.

Actions:
- start: Start Python interpreter (returns interpreter_id)
- stop: Stop interpreter (requires interpreter_id)
- run: Execute Python code (requires interpreter_id and code)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["start", "stop", "run"],
                            "description": "The action to perform",
                        },
                        "interpreter_id": {
                            "type": "string",
                            "description": "Interpreter instance ID (required for 'stop' and 'run')",
                        },
                        "code": {
                            "type": "string",
                            "description": "Python code to execute (for 'run' action)",
                        },
                    },
                    "required": ["action"],
                },
            ),

            # Sandbox tool
            Tool(
                name="sandbox",
                description="""Run shell commands in an isolated container.

Actions:
- start: Start container with specified image (requires image, returns sandbox_id)
- stop: Stop container (requires sandbox_id)
- exec: Execute shell command (requires sandbox_id and command)

Volume mounts:
- volumes: List of volume mounts. Each mount can be:
  - A list [host_path, guest_path] for read-write access
  - A list [host_path, guest_path, true] for read-only access (read_only=true)
  - Example: [["/tmp", "/mnt/tmp"], ["/home", "/mnt/home", true]]""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["start", "stop", "exec"],
                            "description": "The action to perform",
                        },
                        "sandbox_id": {
                            "type": "string",
                            "description": "Sandbox instance ID (required for 'stop' and 'exec')",
                        },
                        "image": {
                            "type": "string",
                            "description": "Container image to use (for 'start' action, e.g., 'alpine', 'ubuntu')",
                        },
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute (for 'exec' action)",
                        },
                        "volumes": {
                            "type": "array",
                            "description": "Volume mounts (for 'start' action)",
                            "items": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 3,
                                "description": "Volume mount as [host_path, guest_path] or [host_path, guest_path, read_only]",
                            },
                        },
                    },
                    "required": ["action"],
                },
            ),
            # Computer tool (existing)
            Tool(
                name="computer",
                description="""Control a desktop computer through an isolated sandbox environment.

This tool allows you to interact with applications, manipulate files, and browse the web just like a human using a desktop computer. The computer starts with a clean Ubuntu environment with XFCE desktop.

Lifecycle actions:
- start: Start a new computer instance (returns computer_id, gui_http_port, gui_https_port)
- stop: Stop a computer instance (requires computer_id)

Computer actions (all require computer_id):
- screenshot: Capture the current screen
- mouse_move: Move cursor to coordinates
- left_click, right_click, middle_click: Click mouse buttons
- double_click, triple_click: Multiple clicks
- left_click_drag: Click and drag between coordinates
- type: Type text
- key: Press keys (e.g., 'Return', 'ctrl+c')
- scroll: Scroll in a direction
- cursor_position: Get current cursor position

Volume mounts:
- volumes: List of volume mounts (for 'start' action). Each mount can be:
  - A list [host_path, guest_path] for read-write access
  - A list [host_path, guest_path, true] for read-only access (read_only=true)
  - Example: [["/tmp", "/mnt/tmp"], ["/home", "/mnt/home", true]]

Coordinates use [x, y] format with origin at top-left (0, 0).
Screen resolution is 1024x768 pixels.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "start",
                                "stop",
                                "screenshot",
                                "mouse_move",
                                "left_click",
                                "right_click",
                                "middle_click",
                                "double_click",
                                "triple_click",
                                "left_click_drag",
                                "type",
                                "key",
                                "scroll",
                                "cursor_position",
                            ],
                            "description": "The action to perform",
                        },
                        "computer_id": {
                            "type": "string",
                            "description": (
                                "The computer instance ID (returned by 'start', "
                                "required for all other actions except 'start')"
                            ),
                        },
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Coordinates [x, y] for actions that require a position",
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to type (for 'type' action)",
                        },
                        "key": {
                            "type": "string",
                            "description": "Key to press (for 'key' action), e.g., 'Return', 'Escape', 'ctrl+c'",
                        },
                        "scroll_direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Direction to scroll (for 'scroll' action)",
                        },
                        "scroll_amount": {
                            "type": "integer",
                            "description": "Number of scroll units (for 'scroll' action, default: 3)",
                            "default": 3,
                        },
                        "start_coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Starting coordinates for 'left_click_drag' action",
                        },
                        "end_coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Ending coordinates for 'left_click_drag' action",
                        },
                        "volumes": {
                            "type": "array",
                            "description": "Volume mounts (for 'start' action)",
                            "items": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 3,
                                "description": "Volume mount as [host_path, guest_path] or [host_path, guest_path, read_only]",
                            },
                        },
                    },
                    "required": ["action"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
        """Handle tool calls."""
        action = arguments.get("action")
        if not action:
            return [TextContent(type="text", text="Missing 'action' parameter")]

        logger.info(f"Tool '{name}' action: {action} with args: {arguments}")

        try:
            # Route to browser handler
            if name == "browser":
                return await handle_browser_tool(browser_handler, action, arguments)
            # Route to code_interpreter handler
            elif name == "code_interpreter":
                return await handle_code_interpreter_tool(code_handler, action, arguments)
            # Route to sandbox handler
            elif name == "sandbox":
                return await handle_sandbox_tool(sandbox_handler, action, arguments)
            # Route to computer handler
            elif name == "computer":
                return await handle_computer_tool(computer_handler, action, arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except BaseException as exception:
            logger.error(f"Tool execution error: {exception}", exc_info=True)
            return [
                TextContent(
                    type="text",
                    text=f"Error executing {name}/{action}: {str(exception)}",
                )
            ]

    async def handle_browser_tool(handler, action: str, arguments: dict) -> list[TextContent]:
        """Handle browser tool actions."""
        if action == "start":
            result = await handler.start(**arguments)
            return [
                TextContent(
                    type="text",
                    text=f"Browser started with ID: {result['browser_id']}\nEndpoint: {result['endpoint']}",
                )
            ]
        elif action == "stop":
            await handler.stop(**arguments)
            return [TextContent(type="text", text="Browser stopped successfully")]
        else:
            return [TextContent(type="text", text=f"Unknown browser action: {action}")]

    async def handle_code_interpreter_tool(handler, action: str, arguments: dict) -> list[TextContent]:
        """Handle code_interpreter tool actions."""
        if action == "start":
            result = await handler.start(**arguments)
            return [
                TextContent(
                    type="text",
                    text=f"Code interpreter started with ID: {result['interpreter_id']}",
                )
            ]
        elif action == "stop":
            await handler.stop(**arguments)
            return [TextContent(type="text", text="Code interpreter stopped successfully")]
        elif action == "run":
            result = await handler.run(**arguments)
            return [TextContent(type="text", text=result["output"])]
        else:
            return [TextContent(type="text", text=f"Unknown code_interpreter action: {action}")]

    async def handle_sandbox_tool(handler, action: str, arguments: dict) -> list[TextContent]:
        """Handle sandbox tool actions."""
        if action == "start":
            result = await handler.start(**arguments)
            return [
                TextContent(
                    type="text",
                    text=f"Sandbox started with ID: {result['sandbox_id']}",
                )
            ]
        elif action == "stop":
            await handler.stop(**arguments)
            return [TextContent(type="text", text="Sandbox stopped successfully")]
        elif action == "exec":
            result = await handler.exec(**arguments)
            output_parts = []
            if result["stdout"]:
                output_parts.append(result["stdout"])
            if result["stderr"]:
                output_parts.append(f"stderr: {result['stderr']}")
            output_parts.append(f"exit_code: {result['exit_code']}")
            return [TextContent(type="text", text="\n".join(output_parts))]
        else:
            return [TextContent(type="text", text=f"Unknown sandbox action: {action}")]

    async def handle_computer_tool(handler, action: str, arguments: dict) -> list[TextContent | ImageContent]:
        """Handle computer tool actions."""
        action_handler = getattr(handler, action, None)
        if not action_handler:
            return [TextContent(type="text", text=f"Unknown action: {action}")]

        result = await action_handler(**arguments)

        # Format response based on action
        if action == "start":
            computer_id = result["computer_id"]
            gui_http_port = result["gui_http_port"]
            gui_https_port = result["gui_https_port"]
            return [
                TextContent(
                    type="text",
                    text=f"Computer started with ID: {computer_id}\nGUI HTTP port: {gui_http_port}\nGUI HTTPS port: {gui_https_port}",
                )
            ]
        elif action == "stop":
            return [
                TextContent(
                    type="text",
                    text="Computer stopped successfully",
                )
            ]
        elif action == "screenshot":
            return [
                ImageContent(
                    type="image",
                    data=result["image_data"],
                    mimeType="image/png",
                )
            ]
        elif action == "cursor_position":
            x, y = result["x"], result["y"]
            return [
                TextContent(
                    type="text",
                    text=f"Cursor position: [{x}, {y}]",
                )
            ]
        elif action == "mouse_move":
            coord = arguments.get("coordinate", [])
            return [
                TextContent(
                    type="text",
                    text=f"Moved cursor to {coord}",
                )
            ]
        elif action in ["left_click", "right_click", "middle_click"]:
            coord = arguments.get("coordinate")
            if coord:
                return [
                    TextContent(
                        type="text",
                        text=f"Moved to {coord} and clicked {action.replace('_', ' ')}",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"Clicked {action.replace('_', ' ')}",
                    )
                ]
        elif action in ["double_click", "triple_click"]:
            coord = arguments.get("coordinate")
            if coord:
                return [
                    TextContent(
                        type="text",
                        text=f"Moved to {coord} and {action.replace('_', ' ')}ed",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"{action.replace('_', ' ').capitalize()}ed",
                    )
                ]
        elif action == "left_click_drag":
            start = arguments.get("start_coordinate", [])
            end = arguments.get("end_coordinate", [])
            return [
                TextContent(
                    type="text",
                    text=f"Dragged from {start} to {end}",
                )
            ]
        elif action == "type":
            text = arguments.get("text", "")
            preview = text[:50] + "..." if len(text) > 50 else text
            return [
                TextContent(
                    type="text",
                    text=f"Typed: {preview}",
                )
            ]
        elif action == "key":
            key = arguments.get("key", "")
            return [
                TextContent(
                    type="text",
                    text=f"Pressed key: {key}",
                )
            ]
        elif action == "scroll":
            direction = arguments.get("scroll_direction", "")
            amount = arguments.get("scroll_amount", 3)
            coord = arguments.get("coordinate", [])
            return [
                TextContent(
                    type="text",
                    text=f"Scrolled {direction} {amount} units at {coord}",
                )
            ]
        else:
            return [
                TextContent(
                    type="text",
                    text=f"Action completed: {action}",
                )
            ]

    # Run the server
    try:
        # Run MCP server on stdio
        async with stdio_server() as streams:
            logger.info("MCP server running on stdio")
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except BaseException as e:
        if isinstance(e, (SystemExit, GeneratorExit)):
            raise
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        await computer_handler.shutdown_all()
        await browser_handler.shutdown_all()
        await code_handler.shutdown_all()
        await sandbox_handler.shutdown_all()


def run():
    """Sync entry point for CLI."""
    anyio.run(main)


if __name__ == "__main__":
    run()
