#!/usr/bin/env python3
"""
Adds the thenvoi (Band.ai) MCP server to Claude Desktop's config.
Run once: python3 patch_claude_config.py
"""
import json
import shutil
from pathlib import Path
from datetime import datetime

CONFIG = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"

THENVOI_ENTRY = {
    "command": "thenvoi-mcp",
    "env": {
        "THENVOI_USER_KEY": "band_u_1781560120_stvNLctPDDTfv4pZ7VfaQlrxWn7zqQ4T",
        "THENVOI_BASE_URL": "https://app.band.ai",
        "THENVOI_MCP_SCOPE": "human",
    },
}

def main():
    # Backup first
    backup = CONFIG.with_suffix(f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(CONFIG, backup)
    print(f"✓ Backed up to {backup.name}")

    data = json.loads(CONFIG.read_text())

    if "mcpServers" not in data:
        data["mcpServers"] = {}

    if "thenvoi" in data["mcpServers"]:
        print("ℹ thenvoi entry already exists — updating it")

    data["mcpServers"]["thenvoi"] = THENVOI_ENTRY

    CONFIG.write_text(json.dumps(data, indent=2))
    print("✓ Added thenvoi MCP server to claude_desktop_config.json")
    print()
    print("Next steps:")
    print("  1. Install the MCP package:  pip install band-mcp")
    print("     (or: uv tool install band-mcp)")
    print("  2. Quit and relaunch Claude Desktop")
    print("  3. The 'thenvoi' tools will appear in the MCP panel")
    print()
    print("Note: Your key (band_u_...) is a USER key → human-scope tools.")
    print("For agent-scope tools you'll need a separate THENVOI_AGENT_KEY.")

if __name__ == "__main__":
    main()
