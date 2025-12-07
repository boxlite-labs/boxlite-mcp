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
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

from mcp.server import Server

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

    Manages a ComputerBox instance and delegates MCP tool calls to its API.
    """

    def __init__(self, memory_mib: int = 4096, cpus: int = 4):
        self._memory_mib = memory_mib
        self._cpus = cpus
        self._computer: Optional[boxlite.ComputerBox] = None
        self._init_error: Optional[str] = None
        self._lock = anyio.Lock()
        self._initialized = False

    async def _ensure_ready(self):
        """Get ComputerBox, initializing on first call."""
        if self._initialized:
            if self._init_error:
                raise RuntimeError(f"ComputerBox initialization failed: {self._init_error}")
            return None

        async with self._lock:
            # Double-check after acquiring lock
            if self._initialized:
                if self._init_error:
                    raise RuntimeError(f"ComputerBox initialization failed: {self._init_error}")
                return None

            try:
                logger.info("Creating ComputerBox...")
                self._computer = boxlite.ComputerBox(cpu=self._cpus, memory=self._memory_mib)
                await self._computer.__aenter__()
                logger.info(f"ComputerBox created. Desktop at: {self._computer.endpoint()}")

                # Wait for desktop to be ready
                logger.info("Waiting for desktop to become ready...")
                await self._computer.wait_until_ready()
                logger.info("Desktop is ready")
                self._init_error = None

            except Exception as e:
                error_msg = f"Failed to initialize ComputerBox: {e}"
                logger.error(error_msg, exc_info=True)
                self._init_error = error_msg
                self._computer = None
                raise

            self._initialized = True
            return None

    async def shutdown(self):
        """Cleanup ComputerBox."""
        if self._computer:
            logger.info("Shutting down ComputerBox...")
            try:
                await self._computer.__aexit__(None, None, None)
                logger.info("ComputerBox shut down successfully")
            except Exception as e:
                logger.error(f"Error during ComputerBox cleanup: {e}", exc_info=True)
            finally:
                self._computer = None

    # Action handlers - delegation to ComputerBox API

    async def screenshot(self, **kwargs) -> dict:
        """Capture screenshot."""
        await self._ensure_ready()
        result = await self._computer.screenshot()
        return {
            "image_data": result["data"],
            "width": result["width"],
            "height": result["height"],
        }

    async def mouse_move(self, coordinate: list[int], **kwargs) -> dict:
        """Move mouse to coordinates."""
        await self._ensure_ready()
        x, y = coordinate
        await self._computer.mouse_move(x, y)
        return {"success": True}

    async def left_click(self, coordinate: Optional[list[int]] = None, **kwargs) -> dict:
        """Click left mouse button."""
        await self._ensure_ready()
        if coordinate:
            x, y = coordinate
            await self._computer.mouse_move(x, y)
        await self._computer.left_click()
        return {"success": True}

    async def right_click(self, coordinate: Optional[list[int]] = None, **kwargs) -> dict:
        """Click right mouse button."""
        await self._ensure_ready()
        if coordinate:
            x, y = coordinate
            await self._computer.mouse_move(x, y)
        await self._computer.right_click()
        return {"success": True}

    async def middle_click(self, coordinate: Optional[list[int]] = None, **kwargs) -> dict:
        """Click middle mouse button."""
        await self._ensure_ready()
        if coordinate:
            x, y = coordinate
            await self._computer.mouse_move(x, y)
        await self._computer.middle_click()
        return {"success": True}

    async def double_click(self, coordinate: Optional[list[int]] = None, **kwargs) -> dict:
        """Double click left mouse button."""
        await self._ensure_ready()
        if coordinate:
            x, y = coordinate
            await self._computer.mouse_move(x, y)
        await self._computer.double_click()
        return {"success": True}

    async def triple_click(self, coordinate: Optional[list[int]] = None, **kwargs) -> dict:
        """Triple click left mouse button."""
        await self._ensure_ready()
        if coordinate:
            x, y = coordinate
            await self._computer.mouse_move(x, y)
        await self._computer.triple_click()
        return {"success": True}

    async def left_click_drag(self, start_coordinate: list[int], end_coordinate: list[int],
                              **kwargs) -> dict:
        """Drag from start to end coordinates."""
        await self._ensure_ready()
        start_x, start_y = start_coordinate
        end_x, end_y = end_coordinate
        await self._computer.left_click_drag(start_x, start_y, end_x, end_y)
        return {"success": True}

    async def type(self, text: str, **kwargs) -> dict:
        """Type text."""
        await self._ensure_ready()
        await self._computer.type(text)
        return {"success": True}

    async def key(self, key: str, **kwargs) -> dict:
        """Press key or key combination."""
        await self._ensure_ready()
        await self._computer.key(key)
        return {"success": True}

    async def scroll(self, coordinate: list[int], scroll_direction: str, scroll_amount: int = 3,
                     **kwargs) -> dict:
        """Scroll at coordinates."""
        await self._ensure_ready()
        x, y = coordinate
        await self._computer.scroll(x, y, scroll_direction, scroll_amount)
        return {"success": True}

    async def cursor_position(self, **kwargs) -> dict:
        """Get current cursor position."""
        await self._ensure_ready()
        x, y = await self._computer.cursor_position()
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

Available actions:
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
            if action == "screenshot":
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
        await handler.shutdown()


def run():
    """Sync entry point for CLI."""
    anyio.run(main)


if __name__ == "__main__":
    run()
