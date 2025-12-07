# boxlite-mcp

MCP server for computer use through an isolated sandbox environment.

Provides a `computer` tool compatible with [Anthropic's computer use API](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use), enabling AI agents to control a full desktop environment safely within a secure sandbox.

## Features

- Full Ubuntu desktop with XFCE environment
- Anthropic computer use API compatible
- Mouse control (click, drag, scroll)
- Keyboard input (typing, key combinations)
- Screenshot capture
- Secure sandbox isolation

## Installation

```bash
# Using uv (recommended)
uv pip install boxlite-mcp

# Or using pip
pip install boxlite-mcp
```

## Configuration

### Claude Desktop

Add to your `claude_desktop_config.json` (see `examples/claude_desktop_config.example.json`):

```json
{
  "mcpServers": {
    "computer": {
      "command": "uv",
      "args": ["run", "boxlite-mcp"]
    }
  }
}
```

## Usage

Once configured, the MCP server provides a `computer` tool with the following actions:

| Action | Description |
|--------|-------------|
| `screenshot` | Capture the current screen |
| `mouse_move` | Move cursor to coordinates |
| `left_click` | Left click (optionally at coordinates) |
| `right_click` | Right click (optionally at coordinates) |
| `middle_click` | Middle click (optionally at coordinates) |
| `double_click` | Double click |
| `triple_click` | Triple click |
| `left_click_drag` | Click and drag between coordinates |
| `type` | Type text |
| `key` | Press key or key combination (e.g., 'Return', 'ctrl+c') |
| `scroll` | Scroll in a direction |
| `cursor_position` | Get current cursor position |

### Coordinate System

- Origin `[0, 0]` is at the top-left corner
- Screen resolution: 1024x768 pixels
- Coordinates are specified as `[x, y]` arrays

## Development

```bash
# Clone the repository
git clone https://github.com/boxlite-labs/boxlite-mcp.git
cd boxlite-mcp

# Install with dev dependencies
uv sync --extra dev

# Run linting
uv run ruff check .

# Run tests
uv run pytest
```

## License

Apache-2.0
