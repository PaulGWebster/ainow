# ainow project notes

ainow is a single-file (ainow.py) multi-provider agentic coding harness.
OpenAI-compatible API, prompt_toolkit REPL, tool-use loop (read/write/
edit files, list directories, run shell commands). Supports OpenRouter, DeepSeek,
Kimi/Moonshot, Longcat.

## Setup
- Repo: /home/paul/Projects/ainow
- Venv: venv/ in repo root, created with `python3 -m venv venv`
- Launch: `bin/ainow` (auto-detects venv python)
- Config lives in ~/.config/ainow/ (providers.json, models.json, history, logs/)
- providers.json is NOT in the repo (gitignored); template is providers.example.json
- Logs: ~/.config/ainow/logs/ (httpd.log, ainow.log)

## Current state
- Core harness feature-complete. HTTP file-transfer server (httpd) added.
- User is testing and will raise issues as they find them.
- Goal: user's own custom harness — lightweight, no bloat, does exactly what they want.

## Features
- Multi-provider chat (OpenRouter, DeepSeek, Kimi/Moonshot, Longcat)
- Tools: read_file, list_dir, write_file, edit_file, bash
- Built-in HTTP file transfer server (ainow httpd start|stop)
- Context management: token counting, usage bar, compress, preprompt/postprompt inserts
- Context warnings at 70% and 90% thresholds
- Tab completion for model specs
- Model catalogue caching (ainow --refresh-models)
- Configurable prompt format via ~/.config/ainow/prompt.format

## Context management
- `/ctx` — stats: message count by role, token usage, visual bar graph
- `/ctx compress` — summarise older messages via the LLM, keep recent 2 turns
- `/ctx window N` — override the detected context window size
- Preprompt: if `~/.config/ainow/preprompt.format` exists, shown before each prompt
- Postprompt: `~/.config/ainow/postprompt.format` shown after each LLM response
  - Default postprompt: `{elapsed}s · {tokens}/{window} ({pct}%)`
- Template vars: `{time}`, `{provider}`, `{model}`, `{messages}`, `{tokens}`, `{window}`, `{pct}`, `{elapsed}`
- Token counting tries tiktoken (o200k_base), falls back to char/4 estimate
- Context window detected by substring match on model id (claude=200k, gemini=1M, most=128k)
