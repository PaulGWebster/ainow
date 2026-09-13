# ainow

Multi-provider agentic coding harness. Real filesystem + shell access, like an
OpenHands/Claude Code session but pointed at whichever model you name.

## Use

```bash
ainow openrouter/moonshotai/kimi-k2        # interactive
ainow deepseek/deepseek-v4-pro "fix the lint errors in checkmail.py"
ainow kimi/kimi-k3 --yolo "…"              # skip approval prompts
```

Tab completion on the model spec:

```
ainow open<TAB>              → openrouter/
ainow openrouter/k<TAB>      → openrouter/moonshotai/kimi-k2, …/kimi-k3, kwaipilot/kat-coder-…
ainow openrouter/claude<TAB> → openrouter/anthropic/claude-opus-4.5, …
```

Matching is basename-aware, so `k` finds `moonshotai/kimi-k2` rather than every
id containing the letter k. Ranking: model-name prefix → full-id prefix →
substring in model name → substring in vendor.

## Listing models

```bash
ainow --models                      # every provider + model count
ainow --models openrouter           # all 338, one per line
ainow --models openrouter kimi      # filtered
ainow --models openrouter | less    # pipes cleanly
```

The count goes to stderr, so `ainow --models openrouter > models.txt` gives you
a clean file. In-session, `/models [pattern]` does the same for the current
provider and reports "N of M" rather than silently truncating.

### The catalogue is not the whole story

`--models` reports what `/v1/models` returns. On OpenRouter that is 338 ids, but
more model strings are addressable than are listed:

* **Routing variants** — `:nitro` (throughput), `:floor` (price), `:online`
  (web search). Only 15 variant ids appear in the catalogue, yet
  `moonshotai/kimi-k2:nitro` resolves fine. `:free` is not universal —
  `kimi-k2:free` 404s with a pointer to the paid slug.
* **`~…-latest` aliases** — 11 of these, e.g. `~anthropic/claude-haiku-latest`,
  `~google/gemini-pro-latest`, `~deepseek/deepseek-v4-flash-latest`. These *are*
  catalogued, so they do tab-complete.

ainow never validates a model against the cache — the string is passed straight
through to the provider. So anything the API accepts works, whether or not
completion offers it. Completion is a convenience, not a whitelist.

## Keys

| key | effect |
|---|---|
| `Ctrl-C` | interrupt the running generation / tool loop, return to prompt. Also clears a part-typed line. **Never exits.** |
| `Ctrl-D` | exit |
| `Ctrl-Q` | exit |

## Commands

```
/help            /clear              reset conversation
/model <spec>    switch model mid-session
/models [pat]    list cached models for current provider
/ctx [compress|window N]  context stats, compress, set window
/think [low|high|max|off]   set kimi-k3 think effort (off = provider default)
/reasoning [on|off]         show the model's reasoning as it streams
/websearch [on|off]         register moonshot builtin $web_search
/auto [on|off]   run tools without asking
/httpd [start|stop]  start/stop built-in file transfer server
/exit
```

## Tools available to the model

`read_file`, `list_dir`, `write_file`, `edit_file`, `bash`

`write_file`, `edit_file` and `bash` prompt for approval by default
(`y` / `N` / `a` = always). `--yolo` or `/auto on` disables that.

## Providers

Read from `~/.config/ainow/providers.json` (mode 0600), generated from
`~/.aicreds` and `~/.api`.

| provider | endpoint |
|---|---|
| `openrouter` | https://openrouter.ai/api/v1 |
| `deepseek` | https://api.deepseek.com/v1 |
| `kimi` / `moonshot` | https://api.moonshot.ai/v1 |
| `longcat` | https://api.longcat.chat/openai/v1 |

All are OpenAI-compatible and go through the `openai` SDK with a per-provider
`base_url`. Adding another means one entry in `providers.json` — no code change.

**Claude** is reachable as `openrouter/anthropic/claude-*`. The local `claude`
CLI's OAuth session is not wired in: that token is scoped to Claude Code, not a
general-purpose API credential, so it isn't usable to drive a third-party
harness. OpenRouter is the clean path.

**Abacus** was dropped — subscription cancelled, and its models are on
OpenRouter anyway.

## HTTP file transfer server

Built-in HTTP server for transferring files to/from the machine. Starts on a free
port, prints the URL so the model (or you) can reach it.

```bash
# CLI
ainow httpd start [-port N] [-root PATH] [-user U] [-pass P]
ainow httpd stop
ainow httpd              # status

# REPL
/httpd start [-port N] [-root PATH] [-user U] [-pass P]
/httpd stop
/httpd                   # status
```

Endpoints (HTTP Basic auth required):
- `GET  /`          — HTML upload form
- `PUT  /filename`  — upload a file
- `GET  /filename`  — download a file
- `GET  /.files`    — JSON file listing

Defaults: user=`ainow`, password is random (printed on start), root=`~/.ainow/httpd/`.
The REPL server runs as a daemon thread and auto-stops on exit.

## Context management

ainow tracks token usage and warns when you're nearing the model's context window.

    /ctx              -- message counts, token usage, visual bar, % of window
    /ctx compress     -- summarise older messages (keep last 2 turns + system)
    /ctx window 200000 -- override detected context window size

**Warnings** appear automatically at 70% and 90% of the context window.

**Pre/post prompt inserts** -- custom messages shown before/after each turn:

| file | when | default |
|---|---|---|
| `~/.config/ainow/preprompt.format` | before prompt | none (off) |
| `~/.config/ainow/postprompt.format` | after response | `{elapsed}s * {tokens}/{window} ({pct}%)` |

Template variables: `{time}`, `{provider}`, `{model}`, `{messages}`, `{tokens}`,
`{window}`, `{pct}`, `{elapsed}`, `{cached}`, `{reasoning}`.

Token counting tries tiktoken (o200k_base) and falls back to a char/4 estimate.
Context window size comes from the model metadata cached at `--refresh-models`
(`context_length` from /v1/models), falling back to substring matching on the
model id (kimi-k3 = 1048576, claude = 200k, gemini = 1M, most = 128k).

## kimi-k3

ainow exercises the kimi-k3 extensions advertised by the Moonshot API:

* **Thinking** — k3 is thinking-only. Reasoning streams as `reasoning_content`,
  is shown dimmed under a `── thinking ──` marker (toggle: `/reasoning`), and is
  persisted in history so the model keeps its chain of thought across turns.
* **Think effort** — `/think low|high|max` sends `think_effort` in the request.
  `off` reverts to the provider default (max). The valid-effort list is read
  from the model metadata.
* **Builtin web search** — `/websearch on` registers the server-side
  `$web_search` tool; the harness echoes the search arguments back as the tool
  result, per the Moonshot builtin-function protocol.
* **Images and video** — put `image:///path/to/pic.png` (or `video://…`) in a
  prompt and it is sent as a base64 content part. `read_file` on a media file
  returns a reference instead of dumping base64 into the transcript.
* **Prompt caching** — `{cached}` reports `cached_tokens` from the streamed
  usage payload; `{reasoning}` reports the reasoning-token count.

## Model cache

Completion reads `~/.config/ainow/models.json`, never the network. Refresh with:

```bash
ainow --refresh-models
```

Worth running when a provider ships something new.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# or: pip install -e .
```

Copy `providers.example.json` to `~/.config/ainow/providers.json` and fill in
your API keys.

For tab completion, source `completions/ainow` from your `.bashrc`:

```bash
source /path/to/ainow/completions/ainow
```

## Layout

```
ainow.py                     main harness
bin/ainow                    launcher script
completions/ainow            bash completion
providers.example.json       config template
requirements.txt             Python dependencies
pyproject.toml               project metadata
README.md
```
