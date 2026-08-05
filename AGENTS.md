# ainow project notes

ainow is a single-file (ainow.py, ~590 lines) multi-provider agentic coding
harness. OpenAI-compatible API, prompt_toolkit REPL, tool-use loop (read/write/
edit files, list directories, run shell commands). Supports OpenRouter, DeepSeek,
Kimi/Moonshot, Longcat.

## Setup
- Repo: /home/paul/Projects/ainow
- Venv: venv/ in repo root, created with `python3 -m venv venv`
- Launch: `bin/ainow` (auto-detects venv python)
- Config lives in ~/.config/ainow/ (providers.json, models.json, history)
- providers.json is NOT in the repo (gitignored); template is providers.example.json

## Current state
- Core harness is feature-complete per user. User is testing and will raise issues
  as they find them.
- Goal: user's own custom harness — lightweight, no bloat, does exactly what they want.
