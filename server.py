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


def _boxinfo_to_dict(info) -> dict:
    """Convert a BoxInfo object to a JSON-serializable dict."""
    return {
        "id": info.id,
        "name": getattr(info, "name", None),
        "state": str(info.state),
        "image": getattr(info, "image", None),
        "cpus": getattr(info, "cpus", None),
        "memory_mib": getattr(info, "memory_mib", None),
        "created_at": str(getattr(info, "created_at", "")),
    }


def _format_run_result(result) -> str:
    """Format an ExecResult into a human-readable string."""
    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"stderr: {result.stderr}")
    parts.append(f"exit_code: {result.exit_code}")
    if getattr(result, "error_message", None):
        parts.append(f"error: {result.error_message}")
    return "\n".join(parts)


class BoxManagementHandler:
    """Handler for cross-cutting box management operations.

    Provides list, get, remove, and metrics across all box types.
    """

    def __init__(self, browser_handler, code_handler, sandbox_handler, computer_handler):
        self._handlers = {
            "browser": browser_handler,
            "code_interpreter": code_handler,
            "sandbox": sandbox_handler,
            "computer": computer_handler,
        }

    def _get_runtime(self):
        """Get the boxlite runtime (Rust singleton)."""
        from boxlite.boxlite import Boxlite
        return Boxlite.default()

    def _find_box_instance(self, box_id: str):
        """Find a SimpleBox instance across all handlers by ID or name."""
        for handler in self._handlers.values():
            for store_name in ("_browsers", "_interpreters", "_sandboxes", "_computers"):
                store = getattr(handler, store_name, None)
                if store and box_id in store:
                    return store[box_id]
        return None

    async def list_boxes(self, state: Optional[str] = None, **kwargs) -> list[dict]:
        """List all boxes managed by the runtime."""
        runtime = self._get_runtime()
        infos = runtime.list_info()
        result = [_boxinfo_to_dict(i) for i in infos]
        if state:
            result = [b for b in result if b["state"].lower() == state.lower()]
        return result

    async def get(self, box_id: str, **kwargs) -> dict:
        """Get info for a specific box by ID or name."""
        runtime = self._get_runtime()
        info = runtime.get_info(box_id)
        return _boxinfo_to_dict(info)

    async def remove(self, box_id: str, force: bool = False, **kwargs) -> dict:
        """Remove a box by ID or name."""
        runtime = self._get_runtime()
        runtime.remove(box_id, force=force)
        # Also remove from handler dicts if present
        for handler in self._handlers.values():
            for store_name in ("_browsers", "_interpreters", "_sandboxes", "_computers"):
                store = getattr(handler, store_name, None)
                if store and box_id in store:
                    del store[box_id]
        return {"success": True}

    async def metrics(self, box_id: Optional[str] = None, **kwargs) -> dict:
        """Get runtime metrics, or per-box metrics if box_id provided."""
        if box_id:
            instance = self._find_box_instance(box_id)
            if instance and hasattr(instance, "_box") and instance._box:
                m = instance._box.metrics()
            else:
                # Fall back to runtime-level get + metrics
                runtime = self._get_runtime()
                box = runtime.get(box_id)
                if box is None:
                    raise RuntimeError(f"Box '{box_id}' not found")
                m = box.metrics()
            return {
                "cpu_percent": getattr(m, "cpu_percent", None),
                "memory_bytes": getattr(m, "memory_bytes", None),
                "commands_run_total": getattr(m, "commands_executed_total", None),
                "run_errors_total": getattr(m, "exec_errors_total", None),
                "total_create_duration_ms": getattr(m, "total_create_duration_ms", None),
                "network_bytes_sent": getattr(m, "network_bytes_sent", None),
                "network_bytes_received": getattr(m, "network_bytes_received", None),
                "network_tcp_connections": getattr(m, "network_tcp_connections", None),
            }
        else:
            runtime = self._get_runtime()
            m = runtime.metrics()
            return {
                "num_running_boxes": getattr(m, "num_running_boxes", None),
                "boxes_created_total": getattr(m, "boxes_created_total", None),
                "boxes_failed_total": getattr(m, "boxes_failed_total", None),
                "total_commands_run": getattr(m, "total_commands_executed", None),
                "total_run_errors": getattr(m, "total_exec_errors", None),
            }


class BrowserToolHandler:
    """Handler for browser tool actions."""

    def __init__(self):
        self._browsers: dict[str, boxlite.BrowserBox] = {}
        self._lock = anyio.Lock()

    def _get_browser(self, browser_id: str) -> boxlite.BrowserBox:
        if browser_id not in self._browsers:
            raise RuntimeError(f"Browser '{browser_id}' not found. Use 'start' action first.")
        return self._browsers[browser_id]

    async def start(self, name: Optional[str] = None, reuse_existing: bool = False,
                    browser: Optional[str] = None,
                    cpus: Optional[int] = None, memory_mib: Optional[int] = None,
                    port: Optional[int] = None, cdp_port: Optional[int] = None,
                    **kwargs) -> dict:
        """Start a new browser instance."""
        async with self._lock:
            try:
                logger.info("Creating BrowserBox...")
                # BrowserBox takes a BrowserBoxOptions for browser config
                opts_kwargs = {}
                if browser:
                    opts_kwargs["browser"] = browser
                if cpus is not None:
                    opts_kwargs["cpu"] = cpus
                if memory_mib is not None:
                    opts_kwargs["memory"] = memory_mib
                if port is not None:
                    opts_kwargs["port"] = port
                if cdp_port is not None:
                    opts_kwargs["cdp_port"] = cdp_port

                options = boxlite.BrowserBoxOptions(**opts_kwargs) if opts_kwargs else None
                bb = boxlite.BrowserBox(
                    options=options, name=name, reuse_existing=reuse_existing
                )
                await bb.__aenter__()
                browser_id = bb.id
                # endpoint() is async in boxlite >= 0.5.10
                cdp_endpoint = await bb.endpoint()
                playwright_ep = None
                try:
                    playwright_ep = await bb.playwright_endpoint()
                except Exception:
                    pass
                logger.info(f"BrowserBox {browser_id} created. CDP: {cdp_endpoint}")
                self._browsers[browser_id] = bb
                result = {
                    "browser_id": browser_id,
                    "endpoint": cdp_endpoint,
                    "created": bb.created,
                }
                if playwright_ep:
                    result["playwright_endpoint"] = playwright_ep
                return result
            except BaseException as e:
                error_msg = f"Failed to start BrowserBox: {e}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg)

    async def stop(self, browser_id: str, **kwargs) -> dict:
        """Stop a browser instance."""
        async with self._lock:
            if browser_id not in self._browsers:
                raise RuntimeError(f"Browser '{browser_id}' not found")
            bb = self._browsers[browser_id]
            logger.info(f"Shutting down BrowserBox {browser_id}...")
            try:
                await bb.__aexit__(None, None, None)
                logger.info(f"BrowserBox {browser_id} shut down successfully")
            except BaseException as e:
                logger.error(f"Error during BrowserBox {browser_id} cleanup: {e}", exc_info=True)
            finally:
                del self._browsers[browser_id]
            return {"success": True}

    async def run_command(self, browser_id: str, command: str, **kwargs) -> dict:
        """Run a shell command inside the browser container."""
        bb = self._get_browser(browser_id)
        result = await bb.exec("sh", "-c", command)
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

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

    async def start(self, name: Optional[str] = None, reuse_existing: bool = False,
                    cpus: Optional[int] = None,
                    memory_mib: Optional[int] = None, image: Optional[str] = None,
                    **kwargs) -> dict:
        """Start a new code interpreter instance."""
        async with self._lock:
            try:
                logger.info("Creating CodeBox...")
                code_kwargs = {}
                if image:
                    code_kwargs["image"] = image
                if cpus is not None:
                    code_kwargs["cpus"] = cpus
                if memory_mib is not None:
                    code_kwargs["memory_mib"] = memory_mib
                interpreter = boxlite.CodeBox(
                    name=name, reuse_existing=reuse_existing, **code_kwargs
                )
                await interpreter.__aenter__()
                interpreter_id = interpreter.id
                logger.info(f"CodeBox {interpreter_id} created")
                self._interpreters[interpreter_id] = interpreter
                return {"interpreter_id": interpreter_id, "created": interpreter.created}
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

    async def run(self, interpreter_id: str, code: str, timeout: Optional[int] = None,
                  **kwargs) -> dict:
        """Run Python code."""
        interpreter = self._get_interpreter(interpreter_id)
        run_kwargs = {}
        if timeout is not None:
            run_kwargs["timeout"] = timeout
        output = await interpreter.run(code, **run_kwargs)
        return {"output": output}

    async def install(self, interpreter_id: str, packages: list[str], **kwargs) -> dict:
        """Install Python packages in the interpreter."""
        interpreter = self._get_interpreter(interpreter_id)
        output = await interpreter.install_packages(*packages)
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

    async def start(self, image: str, name: Optional[str] = None,
                    reuse_existing: bool = False,
                    volumes: Optional[list] = None, cpus: Optional[int] = None,
                    memory_mib: Optional[int] = None, disk_size_gb: Optional[int] = None,
                    env: Optional[dict] = None, working_dir: Optional[str] = None,
                    ports: Optional[list] = None, network: Optional[bool] = None,
                    auto_remove: bool = True, **kwargs) -> dict:
        """Start a new sandbox instance."""
        async with self._lock:
            try:
                logger.info(f"Creating SimpleBox with image '{image}'...")
                sandbox_kwargs: dict = {}
                if cpus is not None:
                    sandbox_kwargs["cpus"] = cpus
                if memory_mib is not None:
                    sandbox_kwargs["memory_mib"] = memory_mib
                if disk_size_gb is not None:
                    sandbox_kwargs["disk_size_gb"] = disk_size_gb
                if env:
                    sandbox_kwargs["env"] = list(env.items())
                if working_dir:
                    sandbox_kwargs["working_dir"] = working_dir
                if network is not None:
                    sandbox_kwargs["network"] = network
                if volumes:
                    sandbox_kwargs["volumes"] = [tuple(v) for v in volumes]
                if ports:
                    sandbox_kwargs["ports"] = [tuple(p) for p in ports]

                sandbox = boxlite.SimpleBox(
                    image=image, name=name, auto_remove=auto_remove,
                    reuse_existing=reuse_existing, **sandbox_kwargs
                )
                await sandbox.__aenter__()
                sandbox_id = sandbox.id
                logger.info(f"SimpleBox {sandbox_id} created (new={sandbox.created})")
                self._sandboxes[sandbox_id] = sandbox
                return {"sandbox_id": sandbox_id, "created": sandbox.created}
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

    async def run_command(self, sandbox_id: str, command: str,
                          env: Optional[dict] = None, **kwargs) -> dict:
        """Run a shell command in the sandbox."""
        sandbox = self._get_sandbox(sandbox_id)
        run_kwargs = {}
        if env:
            run_kwargs["env"] = env
        result = await sandbox.exec("sh", "-c", command, **run_kwargs)
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    async def copy_in(self, sandbox_id: str, host_path: str,
                      container_dest: str, **kwargs) -> dict:
        """Copy files from host into the sandbox."""
        sandbox = self._get_sandbox(sandbox_id)
        await sandbox.copy_in(host_path, container_dest)
        return {"success": True}

    async def copy_out(self, sandbox_id: str, container_src: str,
                       host_dest: str, **kwargs) -> dict:
        """Copy files from sandbox to host."""
        sandbox = self._get_sandbox(sandbox_id)
        await sandbox.copy_out(container_src, host_dest)
        return {"success": True}

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

    def __init__(self):
        self._computers: dict[str, boxlite.ComputerBox] = {}
        self._lock = anyio.Lock()

    def _get_computer(self, computer_id: str) -> boxlite.ComputerBox:
        """Get a ComputerBox by ID."""
        if computer_id not in self._computers:
            raise RuntimeError(f"Computer '{computer_id}' not found. Use 'start' action first.")
        return self._computers[computer_id]

    async def start(self, name: Optional[str] = None, reuse_existing: bool = False,
                    cpus: int = 4, memory_mib: int = 4096,
                    volumes: Optional[list] = None, **kwargs) -> dict:
        """Start a new computer instance and return its ID."""
        async with self._lock:
            try:
                gui_http_port = find_available_port()
                gui_https_port = find_available_port()
                logger.info(f"Creating ComputerBox with ports HTTP={gui_http_port}, HTTPS={gui_https_port}...")

                computer_kwargs = {
                    "cpu": cpus,
                    "memory": memory_mib,
                    "gui_http_port": gui_http_port,
                    "gui_https_port": gui_https_port,
                }
                if volumes:
                    computer_kwargs["volumes"] = [tuple(v) for v in volumes]

                computer = boxlite.ComputerBox(
                    name=name, reuse_existing=reuse_existing, **computer_kwargs
                )
                await computer.__aenter__()
                computer_id = computer.id
                logger.info(f"ComputerBox {computer_id} created")

                logger.info(f"Waiting for desktop {computer_id} to become ready...")
                await computer.wait_until_ready()
                logger.info(f"Desktop {computer_id} is ready")

                self._computers[computer_id] = computer
                return {
                    "computer_id": computer_id,
                    "gui_http_port": gui_http_port,
                    "gui_https_port": gui_https_port,
                    "created": computer.created,
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

    async def run_command(self, computer_id: str, command: str, **kwargs) -> dict:
        """Run a shell command inside the computer container."""
        computer = self._get_computer(computer_id)
        result = await computer.exec("sh", "-c", command)
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

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
    browser_handler = BrowserToolHandler()
    code_handler = CodeInterpreterToolHandler()
    sandbox_handler = SandboxToolHandler()
    computer_handler = ComputerToolHandler()
    box_handler = BoxManagementHandler(browser_handler, code_handler, sandbox_handler, computer_handler)
    server = Server("boxlite")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            # Box management tool (new)
            Tool(
                name="box",
                description="""Manage boxes (containers) across all tool types.

Actions:
- list: List all boxes (optional: state filter)
- get: Get info for a box by ID or name (requires box_id)
- remove: Remove a box by ID or name (requires box_id, optional: force)
- metrics: Get runtime metrics, or per-box metrics if box_id provided""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["list", "get", "remove", "metrics"],
                            "description": "The action to perform",
                        },
                        "box_id": {
                            "type": "string",
                            "description": "Box ID or name (required for 'get', 'remove'; optional for 'metrics')",
                        },
                        "state": {
                            "type": "string",
                            "description": "Filter by state for 'list' action (e.g., 'running', 'stopped')",
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force removal (for 'remove' action, default: false)",
                            "default": False,
                        },
                    },
                    "required": ["action"],
                },
            ),

            # Browser tool
            Tool(
                name="browser",
                description="""Start a browser with Chrome DevTools Protocol (CDP) endpoint.

Use this to get a browser endpoint that can be connected to via Puppeteer, Playwright, or Selenium.

Actions:
- start: Start browser instance (returns browser_id, CDP endpoint, and Playwright endpoint)
- stop: Stop browser instance (requires browser_id)
- run_command: Run shell command inside browser container (requires browser_id, command)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["start", "stop", "run_command"],
                            "description": "The action to perform",
                        },
                        "browser_id": {
                            "type": "string",
                            "description": "Browser instance ID (required for 'stop', 'run_command')",
                        },
                        "name": {
                            "type": "string",
                            "description": "Human-readable name for the browser (for 'start')",
                        },
                        "reuse_existing": {
                            "type": "boolean",
                            "description": "If true and a box with the given name exists, reuse it (requires name)",
                            "default": False,
                        },
                        "browser": {
                            "type": "string",
                            "enum": ["chromium", "firefox", "webkit"],
                            "description": "Browser engine (for 'start', default: chromium)",
                        },
                        "cpus": {
                            "type": "integer",
                            "description": "Number of CPU cores (for 'start')",
                        },
                        "memory_mib": {
                            "type": "integer",
                            "description": "Memory limit in MiB (for 'start')",
                        },
                        "port": {
                            "type": "integer",
                            "description": "Playwright server port (for 'start')",
                        },
                        "cdp_port": {
                            "type": "integer",
                            "description": "CDP port (for 'start')",
                        },
                        "command": {
                            "type": "string",
                            "description": "Shell command to run (for 'run_command')",
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
- run: Run Python code (requires interpreter_id and code)
- install: Install Python packages (requires interpreter_id and packages list)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["start", "stop", "run", "install"],
                            "description": "The action to perform",
                        },
                        "interpreter_id": {
                            "type": "string",
                            "description": "Interpreter instance ID (required for 'stop', 'run', 'install')",
                        },
                        "code": {
                            "type": "string",
                            "description": "Python code to run (for 'run' action)",
                        },
                        "packages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Python packages to install (for 'install' action)",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (for 'run' action)",
                        },
                        "name": {
                            "type": "string",
                            "description": "Human-readable name (for 'start')",
                        },
                        "reuse_existing": {
                            "type": "boolean",
                            "description": "If true and a box with the given name exists, reuse it (requires name)",
                            "default": False,
                        },
                        "cpus": {
                            "type": "integer",
                            "description": "Number of CPU cores (for 'start')",
                        },
                        "memory_mib": {
                            "type": "integer",
                            "description": "Memory limit in MiB (for 'start')",
                        },
                        "image": {
                            "type": "string",
                            "description": "Container image (for 'start', default: Python slim)",
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
- start: Start container (requires image, returns sandbox_id)
- stop: Stop container (requires sandbox_id)
- exec: Run shell command (requires sandbox_id and command)
- copy_in: Copy files from host into container (requires sandbox_id, host_path, container_dest)
- copy_out: Copy files from container to host (requires sandbox_id, container_src, host_dest)""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["start", "stop", "exec", "copy_in", "copy_out"],
                            "description": "The action to perform",
                        },
                        "sandbox_id": {
                            "type": "string",
                            "description": "Sandbox instance ID (required for all actions except 'start')",
                        },
                        "image": {
                            "type": "string",
                            "description": "Container image (for 'start', e.g., 'alpine', 'ubuntu')",
                        },
                        "command": {
                            "type": "string",
                            "description": "Shell command (for 'exec' action)",
                        },
                        "name": {
                            "type": "string",
                            "description": "Human-readable name (for 'start')",
                        },
                        "reuse_existing": {
                            "type": "boolean",
                            "description": "If true and a box with the given name exists, reuse it (requires name)",
                            "default": False,
                        },
                        "cpus": {
                            "type": "integer",
                            "description": "Number of CPU cores (for 'start')",
                        },
                        "memory_mib": {
                            "type": "integer",
                            "description": "Memory limit in MiB (for 'start')",
                        },
                        "disk_size_gb": {
                            "type": "integer",
                            "description": "Disk size in GB (for 'start')",
                        },
                        "env": {
                            "type": "object",
                            "description": "Environment variables as {key: value} (for 'start' or 'exec')",
                        },
                        "working_dir": {
                            "type": "string",
                            "description": "Working directory inside container (for 'start')",
                        },
                        "network": {
                            "type": "boolean",
                            "description": "Enable networking (for 'start')",
                        },
                        "auto_remove": {
                            "type": "boolean",
                            "description": "Auto-remove on stop (for 'start', default: true)",
                            "default": True,
                        },
                        "volumes": {
                            "type": "array",
                            "description": "Volume mounts as [[host, guest], [host, guest, readonly]]",
                            "items": {"type": "array", "minItems": 2, "maxItems": 3},
                        },
                        "ports": {
                            "type": "array",
                            "description": "Port mappings as [[host_port, guest_port], ...]",
                            "items": {"type": "array", "minItems": 2, "maxItems": 2},
                        },
                        "host_path": {
                            "type": "string",
                            "description": "Source path on host (for 'copy_in')",
                        },
                        "container_dest": {
                            "type": "string",
                            "description": "Destination path in container (for 'copy_in')",
                        },
                        "container_src": {
                            "type": "string",
                            "description": "Source path in container (for 'copy_out')",
                        },
                        "host_dest": {
                            "type": "string",
                            "description": "Destination path on host (for 'copy_out')",
                        },
                    },
                    "required": ["action"],
                },
            ),

            # Computer tool
            Tool(
                name="computer",
                description="""Control a desktop computer through an isolated sandbox environment.

This tool allows you to interact with applications, manipulate files, and browse the web just like a human using a desktop computer. The computer starts with a clean Ubuntu environment with XFCE desktop.

Lifecycle actions:
- start: Start a new computer instance (returns computer_id, gui_http_port, gui_https_port)
- stop: Stop a computer instance (requires computer_id)
- run_command: Run shell command inside the computer (requires computer_id, command)

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

Coordinates use [x, y] format with origin at top-left (0, 0).
Screen resolution is 1024x768 pixels.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "start", "stop", "run_command",
                                "screenshot", "mouse_move",
                                "left_click", "right_click", "middle_click",
                                "double_click", "triple_click",
                                "left_click_drag", "type", "key",
                                "scroll", "cursor_position",
                            ],
                            "description": "The action to perform",
                        },
                        "computer_id": {
                            "type": "string",
                            "description": "Computer instance ID (returned by 'start', required for all other actions)",
                        },
                        "name": {
                            "type": "string",
                            "description": "Human-readable name (for 'start')",
                        },
                        "reuse_existing": {
                            "type": "boolean",
                            "description": "If true and a box with the given name exists, reuse it (requires name)",
                            "default": False,
                        },
                        "cpus": {
                            "type": "integer",
                            "description": "Number of CPU cores (for 'start', default: 4)",
                            "default": 4,
                        },
                        "memory_mib": {
                            "type": "integer",
                            "description": "Memory limit in MiB (for 'start', default: 4096)",
                            "default": 4096,
                        },
                        "command": {
                            "type": "string",
                            "description": "Shell command (for 'run_command')",
                        },
                        "coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Coordinates [x, y]",
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to type (for 'type' action)",
                        },
                        "key": {
                            "type": "string",
                            "description": "Key to press (for 'key' action)",
                        },
                        "scroll_direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right"],
                            "description": "Direction to scroll (for 'scroll' action)",
                        },
                        "scroll_amount": {
                            "type": "integer",
                            "description": "Scroll units (for 'scroll', default: 3)",
                            "default": 3,
                        },
                        "start_coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Start coords for 'left_click_drag'",
                        },
                        "end_coordinate": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "End coords for 'left_click_drag'",
                        },
                        "volumes": {
                            "type": "array",
                            "description": "Volume mounts (for 'start')",
                            "items": {"type": "array", "minItems": 2, "maxItems": 3},
                        },
                    },
                    "required": ["action"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
        """Handle tool calls."""
        action = arguments.get("action")
        if not action:
            return [TextContent(type="text", text="Missing 'action' parameter")]

        logger.info(f"Tool '{name}' action: {action} with args: {arguments}")

        try:
            if name == "box":
                return await handle_box_tool(box_handler, action, arguments)
            elif name == "browser":
                return await handle_browser_tool(browser_handler, action, arguments)
            elif name == "code_interpreter":
                return await handle_code_interpreter_tool(code_handler, action, arguments)
            elif name == "sandbox":
                return await handle_sandbox_tool(sandbox_handler, action, arguments)
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

    async def handle_box_tool(handler, action: str, arguments: dict) -> list[TextContent]:
        """Handle box management actions."""
        if action == "list":
            boxes = await handler.list_boxes(**arguments)
            if not boxes:
                return [TextContent(type="text", text="No boxes found")]
            lines = []
            for b in boxes:
                name_part = f" ({b['name']})" if b.get("name") else ""
                lines.append(f"  {b['id']}{name_part}  state={b['state']}  image={b.get('image', '?')}")
            return [TextContent(type="text", text=f"Boxes ({len(boxes)}):\n" + "\n".join(lines))]
        elif action == "get":
            info = await handler.get(**arguments)
            lines = [f"{k}: {v}" for k, v in info.items() if v is not None]
            return [TextContent(type="text", text="\n".join(lines))]
        elif action == "remove":
            await handler.remove(**arguments)
            return [TextContent(type="text", text=f"Box '{arguments.get('box_id')}' removed")]
        elif action == "metrics":
            result = await handler.metrics(**arguments)
            lines = [f"{k}: {v}" for k, v in result.items() if v is not None]
            return [TextContent(type="text", text="\n".join(lines))]
        else:
            return [TextContent(type="text", text=f"Unknown box action: {action}")]

    async def handle_browser_tool(handler, action: str, arguments: dict) -> list[TextContent]:
        """Handle browser tool actions."""
        if action == "start":
            result = await handler.start(**arguments)
            parts = [f"Browser started with ID: {result['browser_id']}"]
            parts.append(f"Endpoint: {result['endpoint']}")
            if result.get("playwright_endpoint"):
                parts.append(f"Playwright endpoint: {result['playwright_endpoint']}")
            if result.get("created") is not None:
                parts.append(f"Created: {result['created']}")
            return [TextContent(type="text", text="\n".join(parts))]
        elif action == "stop":
            await handler.stop(**arguments)
            return [TextContent(type="text", text="Browser stopped successfully")]
        elif action == "run_command":
            result = await handler.run_command(**arguments)
            return [TextContent(type="text", text=_format_run_result_dict(result))]
        else:
            return [TextContent(type="text", text=f"Unknown browser action: {action}")]

    async def handle_code_interpreter_tool(handler, action: str, arguments: dict) -> list[TextContent]:
        """Handle code_interpreter tool actions."""
        if action == "start":
            result = await handler.start(**arguments)
            created = result.get("created")
            msg = f"Code interpreter started with ID: {result['interpreter_id']}"
            if created is not None:
                msg += f"\nCreated: {created}"
            return [TextContent(type="text", text=msg)]
        elif action == "stop":
            await handler.stop(**arguments)
            return [TextContent(type="text", text="Code interpreter stopped successfully")]
        elif action == "run":
            result = await handler.run(**arguments)
            return [TextContent(type="text", text=result["output"])]
        elif action == "install":
            result = await handler.install(**arguments)
            return [TextContent(type="text", text=result["output"])]
        else:
            return [TextContent(type="text", text=f"Unknown code_interpreter action: {action}")]

    def _format_run_result_dict(result: dict) -> str:
        """Format a run result dict into human-readable text."""
        parts = []
        if result.get("stdout"):
            parts.append(result["stdout"])
        if result.get("stderr"):
            parts.append(f"stderr: {result['stderr']}")
        parts.append(f"exit_code: {result['exit_code']}")
        return "\n".join(parts)

    async def handle_sandbox_tool(handler, action: str, arguments: dict) -> list[TextContent]:
        """Handle sandbox tool actions."""
        if action == "start":
            result = await handler.start(**arguments)
            created = result.get("created")
            msg = f"Sandbox started with ID: {result['sandbox_id']}"
            if created is not None:
                msg += f"\nCreated: {created}"
            return [TextContent(type="text", text=msg)]
        elif action == "stop":
            await handler.stop(**arguments)
            return [TextContent(type="text", text="Sandbox stopped successfully")]
        elif action == "exec":
            result = await handler.run_command(**arguments)
            return [TextContent(type="text", text=_format_run_result_dict(result))]
        elif action == "copy_in":
            await handler.copy_in(**arguments)
            return [TextContent(type="text", text="Files copied into sandbox")]
        elif action == "copy_out":
            await handler.copy_out(**arguments)
            return [TextContent(type="text", text="Files copied from sandbox")]
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
            created = result.get("created")
            msg = f"Computer started with ID: {computer_id}\nGUI HTTP port: {gui_http_port}\nGUI HTTPS port: {gui_https_port}"
            if created is not None:
                msg += f"\nCreated: {created}"
            return [
                TextContent(
                    type="text",
                    text=msg,
                )
            ]
        elif action == "stop":
            return [TextContent(type="text", text="Computer stopped successfully")]
        elif action == "run_command":
            return [TextContent(type="text", text=_format_run_result_dict(result))]
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
        await browser_handler.shutdown_all()
        await code_handler.shutdown_all()
        await sandbox_handler.shutdown_all()
        await computer_handler.shutdown_all()


def run():
    """Sync entry point for CLI."""
    anyio.run(main)


if __name__ == "__main__":
    run()
