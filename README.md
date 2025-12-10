# boxlite-mcp

MCP server providing isolated sandbox environments for AI agents.

## The Problem

Today's AI apps don't just generate text – they write and run code, call tools, read your files, and hit the network. Running all of this directly on your machine creates real risks:

- **Security** - Malicious or buggy code can damage your system
- **Privacy** - Sensitive files and credentials are exposed
- **Reliability** - Runaway processes can consume resources or crash your machine

BoxLite solves this by giving AI agents their own isolated VM – full freedom inside, complete safety outside.

## Powered by BoxLite

[BoxLite](https://boxlite-labs.github.io/website/) is an embeddable virtual machine runtime that follows the SQLite philosophy - simple, lightweight, and zero-configuration.

### Why BoxLite?

- **Hardware-level isolation** - True VM security, not just containers. Your AI agent runs in a completely isolated environment.
- **No daemon required** - Unlike Docker, BoxLite doesn't need a background service. Just import and use.
- **Embeddable** - Designed to be embedded directly into your applications, like SQLite for compute.
- **Fast startup** - VMs boot in seconds, not minutes.
- **Cross-platform** - Works on macOS and Linux.

### Use Cases

- **AI Agent Sandboxing** - Let AI agents execute code, browse the web, and use applications safely
- **Secure Code Execution** - Run untrusted code without risk to your host system
- **Browser Automation** - Headless browser with CDP for web scraping and testing
- **Development Environments** - Disposable, reproducible dev environments

## Demo

[▶️ Watch the demo on YouTube](https://youtu.be/JjwLg6ww234)

https://github.com/user-attachments/assets/0685d428-64e4-4a68-adfe-c24dc0dc5ae8

## Available Tools

| Tool | Description |
|------|-------------|
| `computer` | Full Ubuntu desktop with XFCE. Anthropic computer use API compatible. |
| `browser` | Chromium browser with CDP endpoint for Puppeteer/Playwright/Selenium |
| `code_interpreter` | Python code execution sandbox |
| `sandbox` | Generic container for running shell commands |

## Quick Start

### Claude Code

```bash
claude mcp add boxlite -- uvx boxlite-mcp
```

### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "boxlite": {
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

## Development

```bash
git clone https://github.com/boxlite-labs/boxlite-mcp.git
cd boxlite-mcp
uv sync --extra dev
uv run pytest
```

## License

Apache-2.0
