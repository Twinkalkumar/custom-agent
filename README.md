# Local Chat Agent (Ollama + Web Search)

A fully local agent: the LLM runs on your machine via Ollama, and it can
search the web (DuckDuckGo, no API key needed) whenever it decides it needs to.

## 1. Install Ollama (the local LLM runtime)
Download from https://ollama.com and install it, then pull a model that
supports tool calling:

```bash
ollama pull llama3.1
```

(Other tool-capable options: `qwen2.5`, `mistral-nemo`. Bigger models reason
about *when* to call tools more reliably but need more RAM/VRAM.)

Make sure Ollama is running in the background (it usually starts a local
server automatically on `http://localhost:11434`).

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the agent

```bash
python agent.py
```

Chat normally. Ask something like "what's the weather in Tokyo right now"
or "who won the last F1 race" — you'll see it print `[tool call] web_search(...)`
before answering, which means it decided on its own to search.

## How it works (the short version)

- `messages` is just a growing list of `{role, content}` dicts — the entire
  conversation history, including tool results.
- `TOOLS` is a JSON schema describing what functions exist and what
  arguments they take. This is sent to the model on every call so it knows
  what's available.
- When the model wants to use a tool, it doesn't run any code itself — it
  just replies with a structured `tool_calls` request. **Your code** is what
  actually executes `web_search()`, then appends the result back into
  `messages` as a `role: "tool"` message.
- The model is called again with that result in context, and either answers
  or calls another tool. This request → tool → request cycle is the entire
  "agent loop."

## Running with Docker (recommended for portability)

This runs Ollama and the agent as two containers, wired together with
`docker-compose`. Models are stored in a persistent volume so you don't
re-download them on every rebuild.

**First-time setup — create your env file** (needed even for local-only use,
since this is where the provider switch lives):
```bash
cp .env.example .env
```
The default `LLM_PROVIDER=ollama` in that file keeps everything fully offline.

```bash
# 1. Start Ollama in the background
docker compose up -d ollama

# 2. Pull a tool-capable model into the running Ollama container
docker exec -it ollama ollama list
docker exec -it ollama ollama pull llama3.2:1b

# 3. Build and start the web UI
docker compose up -d agent
```

Then open **http://localhost:5000** in your browser and chat.

The small dot in the top-left of the page shows live connection status to
Ollama (green = connected, red = unreachable). Hit "reset" to clear the
conversation history.

To stop everything:

```bash
docker compose down
```

Your pulled models persist in the `ollama_data` volume even after `down`,
so you won't need to re-pull them next time.

**Notes:**
- To use a different model, change `OLLAMA_MODEL` in `docker-compose.yml`
  and `docker exec -it ollama ollama pull <model>`.
- GPU acceleration: uncomment the `deploy.resources` block in
  `docker-compose.yml` (requires the NVIDIA Container Toolkit on the host).
  Without a GPU, Ollama runs on CPU — fine for small models, slow for large ones.

## Switching between local and cloud models

By default this runs fully offline via Ollama. To use a cloud model instead
(OpenAI, or an OpenAI-compatible provider like Groq or Together.ai), nothing
in the agent loop or task system changes — just set a couple of env vars.

**1. Create your env file from the example:**
```bash
cp .env.example .env
```

**2. Edit `.env`:**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

**3. Rebuild and restart:**
```bash
docker compose build agent
docker compose up -d agent
```

The web UI's model tag and status dot will now reflect the cloud backend
instead of Ollama. Switch back anytime by setting `LLM_PROVIDER=ollama`
in `.env` and restarting — no code changes either way.

**To use a different OpenAI-compatible provider** (e.g. Groq, which is free
and fast), set `OPENAI_BASE_URL` in `.env` too — see the comments in
`.env.example` for example endpoints.

**Worth knowing before switching:**
- Cloud calls cost money per request (check the provider's pricing).
- Your conversation content is sent to that provider's servers — it's no
  longer offline/private once you switch.
- `.env` holds a real secret key — make sure it's in `.gitignore` and never
  committed or shared.

## Instruction-driven tasks (instructions.yaml)

Behavior is defined in `instructions.yaml`, not hardcoded in Python. Each
"task" in that file is a self-contained personality: its own system prompt,
and its own subset of tools it's allowed to use.

```yaml
tasks:
  general:
    description: "General chat assistant that searches the web when needed"
    system_prompt: >
      You are a helpful local assistant. Use the web_search tool whenever...
    tools: [web_search]

  coder_helper:
    description: "Coding Q&A, no web access"
    system_prompt: >
      You are a concise coding assistant...
    tools: []
```

- **To change how the agent behaves**: edit the `system_prompt` for a task
  and restart — no code change, no rebuild (it's mounted as a volume in
  `docker-compose.yml`).
- **To add a brand-new task**: add a new entry under `tasks:` with its own
  prompt and tool list.
- **To add a new tool**: write the Python function + JSON schema in
  `TOOL_REGISTRY` inside `agent.py`, then reference its name in any task's
  `tools:` list in `instructions.yaml`. Nothing else needs to change — the
  agent loop, the web UI, and the terminal version all pick it up automatically.
- **Switching tasks**:
  - Web UI: use the dropdown next to the model name.
  - Terminal: type `/task <name>` mid-conversation, or pick one at startup.

Switching tasks resets the conversation, since a new task usually means a
different system prompt and tool set.

## Web UI vs terminal

`app.py` is a small Flask server that reuses the exact same tool-calling
loop from `agent.py` (`run_agent_turn`, `TOOLS`, `web_search`) — nothing
about the agent logic changes, it's just fronted by a browser instead of
a terminal `input()` loop. `/api/chat` takes a message, runs it through
the loop, and returns the final answer as JSON.

To run the terminal version instead inside Docker:

```bash
docker compose run --rm agent python -u agent.py
```

## Where to go from here

- **Add more tools**: e.g. a calculator, file reader, or a Google Custom
  Search API call instead of DuckDuckGo (needs an API key + Custom Search
  Engine ID from https://programmablesearchengine.google.com/).
- **Add memory**: persist `messages` to a file/SQLite between runs so the
  agent remembers past conversations.
- **Add a UI**: wrap `run_agent_turn()` in a simple Flask/Gradio app instead
  of the terminal loop.
- **Swap models**: any Ollama model that supports `tools` will drop in by
  changing the `MODEL` variable.
