#!/usr/bin/env python3
"""
BoxLite MCP Server - Computer Use via Isolated Sandbox

Provides a single 'computer' tool matching Anthropic's computer use API.
Runs a full desktop environment inside an isolated sandbox.
"""
import logging
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

    async def start(self, **kwargs) -> dict:
        """Start a new computer instance and return its ID."""
        async with self._lock:
            try:
                logger.info("Creating ComputerBox...")
                computer = boxlite.ComputerBox(cpu=self._cpus, memory=self._memory_mib)
                await computer.__aenter__()
                computer_id = computer.id
                logger.info(f"ComputerBox {computer_id} created. Desktop at: {computer.endpoint()}")

                # Wait for desktop to be ready
                logger.info(f"Waiting for desktop {computer_id} to become ready...")
                await computer.wait_until_ready()
                logger.info(f"Desktop {computer_id} is ready")

                self._computers[computer_id] = computer
                return {"computer_id": computer_id}

            except Exception as e:
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
            except Exception as e:
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
                except Exception as e:
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
    logger.info("Starting BoxLite Computer MCP Server")

    # Create handler and server
    handler = ComputerToolHandler()
    server = Server("boxlite-computer")

    # Register unified computer tool (Anthropic-compatible)
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available computer use tools."""
        return [
            Tool(
                name="computer",
                description="""Control a desktop computer through an isolated sandbox environment.

This tool allows you to interact with applications, manipulate files, and browse the web just like a human using a desktop computer. The computer starts with a clean Ubuntu environment with XFCE desktop.

Lifecycle actions:
- start: Start a new computer instance (returns computer_id)
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
                    },
                    "required": ["action"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
        """Handle unified computer tool calls."""
        if name != "computer":
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        action = arguments.get("action")
        if not action:
            return [TextContent(type="text", text="Missing 'action' parameter")]

        logger.info(f"Computer action: {action} with args: {arguments}")

        try:
            # Route action to handler method
            action_handler = getattr(handler, action, None)
            if not action_handler:
                return [TextContent(type="text", text=f"Unknown action: {action}")]

            result = await action_handler(**arguments)

            # Format response based on action
            if action == "start":
                computer_id = result["computer_id"]
                return [
                    TextContent(
                        type="text",
                        text=f"Computer started with ID: {computer_id}",
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

        except Exception as exception:
            logger.error(f"Action execution error: {exception}", exc_info=True)
            return [
                TextContent(
                    type="text",
                    text=f"Error executing {action}: {str(exception)}",
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
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        await handler.shutdown_all()


def run():
    """Sync entry point for CLI."""
    anyio.run(main)


if __name__ == "__main__":
    run()
