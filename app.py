"""
Web UI for the local agent — Flask backend.

Reuses the exact same agent loop and task system from agent.py. The
browser can list available tasks (from instructions.yaml), switch between
them, and chat — same tool-calling loop as the terminal version underneath.
"""

from flask import Flask, render_template, request, jsonify

from agent import (
    MODEL,
    BACKEND_LABEL,
    PROVIDER,
    run_agent_turn,
    wait_for_ollama,
    check_connection,
    load_tasks,
    get_task,
    list_tasks,
    default_task_name,
)

app = Flask(__name__)

_tasks_data = load_tasks()
_current_task = get_task(default_task_name(_tasks_data), _tasks_data)
conversation = [{"role": "system", "content": _current_task["system_prompt"]}]


@app.route("/")
def index():
    return render_template(
        "index.html",
        model=BACKEND_LABEL,
        tasks=list_tasks(_tasks_data),
        current_task=_current_task["name"],
    )


@app.route("/api/status")
def status():
    result = check_connection()
    return jsonify(result), (200 if result.get("connected") else 503)


@app.route("/api/tasks")
def tasks():
    return jsonify({"tasks": list_tasks(_tasks_data), "current": _current_task["name"]})


@app.route("/api/select_task", methods=["POST"])
def select_task():
    global _current_task, conversation
    data = request.get_json(silent=True) or {}
    name = data.get("task")
    if name not in _tasks_data["tasks"]:
        return jsonify({"error": f"unknown task '{name}'"}), 400

    _current_task = get_task(name, _tasks_data)
    conversation = [{"role": "system", "content": _current_task["system_prompt"]}]
    return jsonify({"status": "ok", "task": name})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    conversation.append({"role": "user", "content": user_message})
    try:
        answer = run_agent_turn(conversation, _current_task["tools"], _current_task["functions"])
    except Exception as e:
        conversation.pop()
        return jsonify({"error": str(e)}), 500

    return jsonify({"reply": answer})


@app.route("/api/reset", methods=["POST"])
def reset():
    global conversation
    conversation = [{"role": "system", "content": _current_task["system_prompt"]}]
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    wait_for_ollama()
    app.run(host="0.0.0.0", port=5000)
