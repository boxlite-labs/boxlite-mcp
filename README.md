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

### Claude Code

```bash
claude mcp add computer -- uvx --prerelease=allow boxlite-mcp
```

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

### Lifecycle Actions

| Action | Description | Parameters |
|--------|-------------|------------|
| `start` | Start a new computer instance | - |
| `stop` | Stop a computer instance | `computer_id: string` |

### Computer Actions

All actions below require `computer_id` (returned by `start`).

| Action | Description | Parameters |
|--------|-------------|------------|
| `screenshot` | Capture current screen | `computer_id` |
| `mouse_move` | Move cursor | `computer_id`, `coordinate: [x, y]` |
| `left_click` | Left click | `computer_id`, `coordinate?: [x, y]` |
| `right_click` | Right click | `computer_id`, `coordinate?: [x, y]` |
| `middle_click` | Middle click | `computer_id`, `coordinate?: [x, y]` |
| `double_click` | Double click | `computer_id`, `coordinate?: [x, y]` |
| `triple_click` | Triple click | `computer_id`, `coordinate?: [x, y]` |
| `left_click_drag` | Click and drag | `computer_id`, `start_coordinate: [x, y]`, `end_coordinate: [x, y]` |
| `type` | Type text | `computer_id`, `text: string` |
| `key` | Press key combination | `computer_id`, `key: string` (e.g., `Return`, `ctrl+c`) |
| `scroll` | Scroll | `computer_id`, `coordinate: [x, y]`, `scroll_direction: up\|down\|left\|right`, `scroll_amount?: int` |
| `cursor_position` | Get cursor position | `computer_id` |

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

### Testing Local Code

To test your local changes with Claude Code:

```bash
claude mcp add computer-dev -- uv run --directory /path/to/boxlite-mcp python -m server
```

## License

Apache-2.0
