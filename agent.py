"""
Local AI Agent — chat + web search, with a swappable LLM backend.

Architecture:
  User -> messages[] -> LLM -> (text answer) OR (tool_call request)
  If tool_call: run the real Python function -> feed result back into messages[]
  -> call LLM again -> repeat until it gives a plain text answer.

PROVIDERS: controlled by the LLM_PROVIDER env var.
  - "ollama" (default): fully local, via a running Ollama server.
  - "openai": any OpenAI-compatible cloud API (OpenAI itself, or compatible
    providers like Groq/Together.ai via OPENAI_BASE_URL).
The task system, tool registry, and agent loop shape are identical either
way — only how a single "chat" call is made and parsed differs.

TASKS: behavior is driven by instructions.yaml, not hardcoded here. Each
"task" in that file has its own system prompt and its own subset of tools.
"""

import os
import json
import time

import ollama
import yaml
from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()  # "ollama" or "openai"

# --- Ollama (local) config ---
# Model must support "tool calling". Good options (pull first):
#   ollama pull llama3.1
#   ollama pull qwen2.5
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

# --- OpenAI-compatible (cloud) config ---
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # set to use Groq/Together/etc instead of openai.com

_openai_client = None
if PROVIDER == "openai":
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "Set it as an environment variable (see .env.example)."
        )
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    _openai_client = OpenAI(**kwargs)

# Convenience for display/status purposes
MODEL = OPENAI_MODEL if PROVIDER == "openai" else OLLAMA_MODEL
BACKEND_LABEL = f"openai:{OPENAI_MODEL}" if PROVIDER == "openai" else f"ollama:{OLLAMA_MODEL}"

INSTRUCTIONS_FILE = os.environ.get("INSTRUCTIONS_FILE", "instructions.yaml")


def wait_for_ollama(timeout: int = 60):
    """Block until the local Ollama server is reachable. No-op for cloud providers."""
    if PROVIDER != "ollama":
        return
    start = time.time()
    while time.time() - start < timeout:
        try:
            ollama_client.list()
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Could not reach Ollama at {OLLAMA_HOST} after {timeout}s")


def check_connection() -> dict:
    """Lightweight health check used by the web UI's status dot."""
    if PROVIDER == "openai":
        try:
            _openai_client.models.list()
            return {"connected": True, "provider": "openai", "model": OPENAI_MODEL}
        except Exception as e:
            return {"connected": False, "provider": "openai", "error": str(e)}
    else:
        try:
            ollama_client.list()
            return {"connected": True, "provider": "ollama", "model": OLLAMA_MODEL, "host": OLLAMA_HOST}
        except Exception as e:
            return {"connected": False, "provider": "ollama", "error": str(e)}


# ---------------------------------------------------------------------------
# 2. TOOL REGISTRY — every tool the agent could ever use, in one place.
#    instructions.yaml picks a subset of these names per task.
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo, no API key needed) and return results as text."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = [f"- {r['title']}: {r['body']} (source: {r['href']})" for r in results]
        return "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


TOOL_REGISTRY = {
    "web_search": {
        "function": web_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for current, real-time, or factual information "
                    "the model may not know or that changes over time."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query to look up."},
                        "max_results": {"type": "integer", "description": "How many results to fetch (default 5)."},
                    },
                    "required": ["query"],
                },
            },
        },
    },
    # Add new tools here, e.g.:
    # "read_file": {"function": read_file, "schema": {...}},
}


# ---------------------------------------------------------------------------
# 3. TASKS — loaded from instructions.yaml
# ---------------------------------------------------------------------------
def load_tasks(path: str = None) -> dict:
    path = path or INSTRUCTIONS_FILE
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_task(name: str, tasks_data: dict = None) -> dict:
    tasks_data = tasks_data or load_tasks()
    task = tasks_data["tasks"][name]
    tool_names = task.get("tools", [])
    tools_schema = [TOOL_REGISTRY[t]["schema"] for t in tool_names if t in TOOL_REGISTRY]
    functions = {t: TOOL_REGISTRY[t]["function"] for t in tool_names if t in TOOL_REGISTRY}
    return {
        "name": name,
        "description": task.get("description", ""),
        "system_prompt": task["system_prompt"].strip(),
        "tools": tools_schema,
        "functions": functions,
    }


def list_tasks(tasks_data: dict = None) -> list:
    tasks_data = tasks_data or load_tasks()
    return [{"name": name, "description": t.get("description", "")} for name, t in tasks_data["tasks"].items()]


def default_task_name(tasks_data: dict = None) -> str:
    tasks_data = tasks_data or load_tasks()
    return tasks_data.get("default_task", next(iter(tasks_data["tasks"])))


# ---------------------------------------------------------------------------
# 4. THE AGENT LOOP — one version per provider, since message/tool-call
#    formats differ slightly between Ollama and OpenAI-style APIs. Both
#    expose the same run_agent_turn(messages, tools, functions) signature.
# ---------------------------------------------------------------------------
def _run_turn_ollama(messages, tools, functions) -> str:
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=messages, tools=tools or None)
    msg = response["message"]
    messages.append(msg)

    while msg.get("tool_calls"):
        for call in msg["tool_calls"]:
            fn_name = call["function"]["name"]
            fn_args = call["function"]["arguments"]
            print(f"  [tool call] {fn_name}({fn_args})")
            fn = functions.get(fn_name)
            result = fn(**fn_args) if fn else f"Unknown tool: {fn_name}"
            messages.append({"role": "tool", "content": result})

        response = ollama_client.chat(model=OLLAMA_MODEL, messages=messages, tools=tools or None)
        msg = response["message"]
        messages.append(msg)

    return msg["content"]


def _run_turn_openai(messages, tools, functions) -> str:
    response = _openai_client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, tools=tools or None
    )
    msg = response.choices[0].message
    messages.append(msg.model_dump(exclude_none=True))

    while msg.tool_calls:
        for call in msg.tool_calls:
            fn_name = call.function.name
            fn_args = json.loads(call.function.arguments)
            print(f"  [tool call] {fn_name}({fn_args})")
            fn = functions.get(fn_name)
            result = fn(**fn_args) if fn else f"Unknown tool: {fn_name}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        response = _openai_client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, tools=tools or None
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

    return msg.content


def run_agent_turn(messages: list, tools: list, functions: dict) -> str:
    """Send messages to the active provider, resolve any tool calls, return final answer."""
    if PROVIDER == "openai":
        return _run_turn_openai(messages, tools, functions)
    return _run_turn_ollama(messages, tools, functions)


def main():
    wait_for_ollama()
    tasks_data = load_tasks()

    print(f"Local agent ready (backend: {BACKEND_LABEL})")
    print("Available tasks:")
    for t in list_tasks(tasks_data):
        print(f"  - {t['name']}: {t['description']}")

    task_name = input(f"\nWhich task? [{default_task_name(tasks_data)}]: ").strip()
    if not task_name:
        task_name = default_task_name(tasks_data)

    task = get_task(task_name, tasks_data)
    messages = [{"role": "system", "content": task["system_prompt"]}]
    print(f"\nUsing task '{task['name']}'. Type 'exit' to quit, '/task <name>' to switch.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        if user_input.startswith("/task "):
            new_name = user_input.split(" ", 1)[1].strip()
            if new_name in tasks_data["tasks"]:
                task = get_task(new_name, tasks_data)
                messages = [{"role": "system", "content": task["system_prompt"]}]
                print(f"[switched to task '{new_name}', conversation reset]\n")
            else:
                print(f"[unknown task '{new_name}']\n")
            continue

        messages.append({"role": "user", "content": user_input})
        answer = run_agent_turn(messages, task["tools"], task["functions"])
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
