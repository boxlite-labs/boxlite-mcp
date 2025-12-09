# boxlite-mcp

> **Powered by [BoxLite](https://boxlite-labs.github.io/website/)** - An embeddable virtual machine runtime following the SQLite philosophy. BoxLite provides hardware-level isolation for AI agents with no daemon required, combining container simplicity with VM security. Coming soon as open source!

MCP server for computer use through an isolated sandbox environment.

Provides a `computer` tool compatible with [Anthropic's computer use API](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use), enabling AI agents to control a full desktop environment safely.

## Demo

[▶️ Watch the demo on YouTube](https://youtu.be/JjwLg6ww234)



https://github.com/user-attachments/assets/0685d428-64e4-4a68-adfe-c24dc0dc5ae8



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
claude mcp add computer -- uvx boxlite-mcp
```

### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "computer": {
      "command": "uvx",
      "args": ["boxlite-mcp"]
    }
  }
}
```

### Manual Installation

```bash
pip install boxlite-mcp
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
