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
- Context window detected by substring match on model id (claude=200k, gemini=1M, kimi=1M, most=128k)

## Journal

Harness-level changes only. Anything referencing private infrastructure, hosts,
paths, positions, or ongoing projects lives in a separate private journal outside
this repo (see system.local / journal.env), NOT here — this file is public.

### 2026-09-12
- Debugged a wedged interactive session: py-spy showed the process parked at the
  prompt_toolkit input prompt (asyncio select), not in a request or blocked on the
  network. Unresponsiveness was a dead prompt not receiving input, not an API hang.
  (py-spy into a throwaway venv is a handy way to introspect a stuck session.)
- Fixed a real context-window bug: _CTX_WINDOWS mapped "kimi" to 128_000 but Kimi K3
  is 1M, so the harness under-reported the window 8x and showed false "context blown"
  warnings. Fixed kimi -> 1_000_000. Verified a providers.json ctx_window override
  still takes precedence.
- Added a `journal` tool (registered in TOOLS + TOOL_SCHEMA, non-destructive so no
  approval prompt): t_journal(text, section) appends a durable note to a persistent
  working-memory journal over ssh. section=LOG (default) adds a dated LOG entry;
  section=THREADS adds an OPEN THREADS bullet. Target is configured via env
  (AINOW_JOURNAL_SSH / AINOW_JOURNAL_FILE), auto-loaded from
  ~/.config/ainow/journal.env (gitignored), so no private host/path is baked into
  this public repo. Gotcha learned: ssh joins argv into a remote shell string, so
  passing note text as an ssh argument re-exposes quoting bugs; base64-encoding the
  text and decoding it remotely (piped over ssh stdin) is the robust pattern.
- Added per-session transcripts: _transcript_start/_tx append
  logs/transcript-<ts>-<pid>.md as the conversation happens (user turns, assistant
  text, tool calls + results), started in both repl and one-shot. Survives a crash,
  unlike the in-memory message list.
- Added an optional private system-prompt overlay: if ~/.config/ainow/system.local
  (gitignored) exists, its contents are appended to SYSTEM via _system(). This is
  the supported place for private context (hosts, paths, ongoing projects) that
  must not be committed.
