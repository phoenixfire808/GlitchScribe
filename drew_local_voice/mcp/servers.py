"""Reference MCP server: local filesystem access with read-only default.

Provides: read_file, list_directory. Write operations blocked by default filter.
Can be used as a standalone MCP server subprocess or as a reference implementation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _read_file(args: dict) -> dict:
    """Read a local file."""
    path = args.get("path", "").strip()
    if not path:
        return {"content": [{"type": "text", "text": "Error: no path provided"}]}
    try:
        full = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(full):
            return {"content": [{"type": "text", "text": f"File not found: {full}"}]}
        with open(full, encoding="utf-8") as f:
            text = f.read()[:50_000]
        return {"content": [{"type": "text", "text": text}]}
    except Exception as exc:
        return {"content": [{"type": "text", "text": f"Error reading file: {exc}"}]}


def _list_directory(args: dict) -> dict:
    """List a directory."""
    path = args.get("path", ".").strip()
    try:
        full = os.path.abspath(os.path.expanduser(path))
        entries = sorted(os.listdir(full))
        text = "\n".join(entries)
        return {"content": [{"type": "text", "text": text}]}
    except Exception as exc:
        return {"content": [{"type": "text", "text": f"Error listing directory: {exc}"}]}


TOOLS = {
    "read_file": {
        "description": "Read the contents of a local file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
            },
            "required": ["path"],
        },
        "handler": _read_file,
    },
    "list_directory": {
        "description": "List the contents of a local directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the directory."},
            },
        },
        "handler": _list_directory,
    },
}


def handle_list_tools() -> dict:
    return {
        "tools": [
            {
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["inputSchema"],
            }
            for name, spec in TOOLS.items()
        ]
    }


def handle_call_tool(name: str, arguments: dict) -> dict:
    spec = TOOLS.get(name)
    if spec is None:
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}
    return spec["handler"](arguments)


def main() -> None:
    """Run as a subprocess MCP server using stdio transport."""
    request_id = 0
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            msg = json.loads(line)
            method = msg.get("method", "")
            msg_id = msg.get("id")

            if method == "tools/list":
                result = handle_list_tools()
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": result}
            elif method == "tools/call":
                params = msg.get("params", {})
                name = params.get("name", "")
                arguments = params.get("arguments", {})
                result = handle_call_tool(name, arguments)
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": result}
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stderr.write(f"MCP server error: {exc}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
