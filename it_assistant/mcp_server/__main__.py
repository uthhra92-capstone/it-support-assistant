"""Entry point: `python -m it_assistant.mcp_server` runs the tool server over stdio.

The agent starts this automatically as a subprocess. stdout carries MCP protocol
messages, so all logging goes to stderr and to var/logs/mcp_server.log.
"""

from it_assistant.config import get_settings
from it_assistant.logging_config import setup_logging
from it_assistant.mcp_server.server import create_server


def main() -> None:
    settings = get_settings()
    setup_logging(settings, log_file="mcp_server.log")
    create_server(settings).run(transport="stdio")


if __name__ == "__main__":
    main()
