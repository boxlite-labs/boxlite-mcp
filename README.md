# boxlite-mcp

MCP server for computer use through an isolated sandbox environment.

Provides a `computer` tool compatible with [Anthropic's computer use API](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use), enabling AI agents to control a full desktop environment safely.

## Features

- Full Ubuntu desktop with XFCE environment (1024x768)
- Anthropic computer use API compatible
- Mouse control (click, drag, scroll)
- Keyboard input (typing, key combinations)
- Screenshot capture
- Secure sandbox isolation

## Quick Start

### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "computer": {
      "command": "uvx",
      "args": ["--prerelease=allow", "boxlite-mcp"]
    }
  }
}
```

> **Note:** `--prerelease=allow` is required until boxlite reaches stable release.

### Manual Installation

```bash
pip install boxlite-mcp --pre
```

## Available Actions

| Action | Description | Parameters |
|--------|-------------|------------|
| `screenshot` | Capture current screen | - |
| `mouse_move` | Move cursor | `coordinate: [x, y]` |
| `left_click` | Left click | `coordinate?: [x, y]` |
| `right_click` | Right click | `coordinate?: [x, y]` |
| `middle_click` | Middle click | `coordinate?: [x, y]` |
| `double_click` | Double click | `coordinate?: [x, y]` |
| `triple_click` | Triple click | `coordinate?: [x, y]` |
| `left_click_drag` | Click and drag | `start_coordinate: [x, y]`, `end_coordinate: [x, y]` |
| `type` | Type text | `text: string` |
| `key` | Press key combination | `key: string` (e.g., `Return`, `ctrl+c`) |
| `scroll` | Scroll | `coordinate: [x, y]`, `scroll_direction: up\|down\|left\|right`, `scroll_amount?: int` |
| `cursor_position` | Get cursor position | - |

Coordinates use `[x, y]` format with origin at top-left `[0, 0]`.

## Development

```bash
git clone https://github.com/boxlite-labs/boxlite-mcp.git
cd boxlite-mcp

# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .
```

## License

Apache-2.0
