"""Route LLM calls through a logged-in CLI instead of a provider API key.

Set PAPER_ORCHESTRA_CLI to "codex" (ChatGPT plan) or "claude" (Claude plan).
When it is unset, the original Gemini/OpenAI API code paths run unchanged.

Optional settings:
  PAPER_ORCHESTRA_CLI_MODEL        model passed to the CLI (claude default: Sonnet 5.5)
  PAPER_ORCHESTRA_CLI_USAGE_LOG    token log path (default: runs/cli_usage.jsonl)
  PAPER_ORCHESTRA_CLI_CONCURRENCY  simultaneous CLI processes (default: 2)
  PAPER_ORCHESTRA_CLI_TIMEOUT      seconds per call (default: 900)

Each call runs in an empty temporary directory. Shell access is off, so the
model can only answer, plus web search when the calling agent asks for it.
"""

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional, Tuple

BACKEND_ENV = "PAPER_ORCHESTRA_CLI"
SUPPORTED_BACKENDS = ("codex", "claude")

_ANSWER_ONLY_NOTE = (
    "You are being called as a text-completion backend. Reply with the requested "
    "output only, in your final message. Do not edit files or run commands."
)

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"

_semaphore: Optional[threading.BoundedSemaphore] = None
_semaphore_lock = threading.Lock()
_usage_lock = threading.Lock()


def active_backend() -> Optional[str]:
    value = os.getenv(BACKEND_ENV, "").strip().lower()
    if not value:
        return None
    if value not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"{BACKEND_ENV}={value!r} is not supported. Use one of: {SUPPORTED_BACKENDS}"
        )
    return value


def _get_semaphore() -> threading.BoundedSemaphore:
    global _semaphore
    if _semaphore is None:
        with _semaphore_lock:
            if _semaphore is None:
                limit = int(os.getenv("PAPER_ORCHESTRA_CLI_CONCURRENCY", "2"))
                _semaphore = threading.BoundedSemaphore(max(1, limit))
    return _semaphore


def _flatten(contents: Any) -> Iterable[Any]:
    if contents is None:
        return
    if isinstance(contents, (list, tuple)):
        for item in contents:
            yield from _flatten(item)
    elif getattr(contents, "parts", None) is not None:
        yield from _flatten(contents.parts)
    else:
        yield contents


def _save_binary(data: bytes, mime_type: str, workdir: str, index: int) -> str:
    extension = mimetypes.guess_extension(mime_type or "") or ".bin"
    path = os.path.join(workdir, f"attachment_{index}{extension}")
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def _add_binary(
    data: bytes,
    mime_type: str,
    workdir: str,
    texts: List[str],
    images: List[str],
) -> None:
    path = _save_binary(data, mime_type, workdir, len(images) + len(texts))
    if (mime_type or "").lower() == "application/pdf":
        from utils.pdf_utils import load_paper

        texts.append(f"[PDF CONTENT]\n{load_paper(path, min_size=100)}\n[END PDF CONTENT]")
    elif (mime_type or "").lower().startswith("image/"):
        images.append(path)
    else:
        raise ValueError(f"Unsupported attachment type for CLI backend: {mime_type}")


def _collect_parts(contents: Any, workdir: str) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    images: List[str] = []

    for part in _flatten(contents):
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict):  # OpenAI-style content item
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                header, encoded = url.split(",", 1)
                mime_type = header.split(":", 1)[1].split(";", 1)[0]
                _add_binary(base64.b64decode(encoded), mime_type, workdir, texts, images)
            else:
                raise ValueError(f"Unsupported content item: {part.get('type')}")
        elif getattr(part, "text", None) is not None:
            texts.append(part.text)
        elif getattr(part, "inline_data", None) is not None:
            blob = part.inline_data
            _add_binary(blob.data, blob.mime_type, workdir, texts, images)
        else:
            raise ValueError(f"Unsupported content part: {type(part).__name__}")

    return texts, images


def _wants_web_search(configs: dict) -> bool:
    return any(
        getattr(tool, "google_search", None) is not None
        for tool in (configs.get("tools") or [])
    )


def _build_prompt(texts: List[str], configs: dict, images: List[str]) -> str:
    sections = [_ANSWER_ONLY_NOTE]

    system_instruction = configs.get("system_instruction")
    if system_instruction:
        system_texts, _ = _collect_parts(system_instruction, "")
        sections.append("SYSTEM INSTRUCTIONS:\n" + "\n".join(system_texts))

    if images:
        names = ", ".join(os.path.basename(path) for path in images)
        sections.append(f"Attached images, in order: {names}")

    sections.extend(texts)
    return "\n\n".join(sections)


def _child_env() -> dict:
    """Drop provider keys so the CLI uses the plan login, never paid API access."""
    env = dict(os.environ)
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(key, None)
    return env


def _run(command: List[str], prompt: str, workdir: str) -> subprocess.CompletedProcess:
    timeout = int(os.getenv("PAPER_ORCHESTRA_CLI_TIMEOUT", "900"))
    with _get_semaphore():
        return subprocess.run(
            command,
            input=prompt,  # stdin: Windows caps argv at about 32K characters
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            cwd=workdir,
            env=_child_env(),
            timeout=timeout,
        )


def _require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"'{name}' CLI was not found on PATH.")
    return path


def _call_codex(prompt: str, images: List[str], search: bool, workdir: str) -> str:
    output_path = os.path.join(workdir, "final_message.txt")
    command = [_require_executable("codex")]
    if search:
        command.append("--search")
    command += [
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--sandbox",
        "read-only",
        "--cd",
        workdir,
        "--output-last-message",
        output_path,
        "-c",
        "features.shell_tool=false",
        "-c",
        "project_doc_max_bytes=0",
    ]
    model = os.getenv("PAPER_ORCHESTRA_CLI_MODEL")
    if model:
        command += ["--model", model]
    command.append("-")  # read the prompt from stdin
    if images:
        command += ["--image", *images]  # variadic, so it must stay last

    result = _run(command, prompt, workdir)
    if result.returncode != 0:
        raise RuntimeError(f"codex failed: {(result.stderr or '')[-800:]}")
    if not os.path.exists(output_path):
        raise RuntimeError("codex finished without a final message.")
    with open(output_path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _record_usage(model: str, payload: dict, search: bool) -> None:
    """Appends token counts for one call. Prompt and response text are not logged."""
    usage = payload.get("usage") or {}
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "web_search": search,
        "turns": payload.get("num_turns"),
        "input_tokens": usage.get("input_tokens", 0),
        "cache_write_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "api_equivalent_usd": payload.get("total_cost_usd"),
    }
    default_path = os.path.join(os.path.dirname(__file__), "..", "runs", "cli_usage.jsonl")
    path = os.getenv("PAPER_ORCHESTRA_CLI_USAGE_LOG") or default_path
    with _usage_lock:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


def _call_claude(prompt: str, images: List[str], search: bool, workdir: str) -> str:
    tools = []
    if search:
        tools.append("WebSearch")
        prompt += "\n\nUse the WebSearch tool to find these papers before you answer."
    if images:
        tools.append("Read")  # file tools stay confined to the temporary directory
        prompt += "\n\nRead the attached image files from the current directory."

    model = os.getenv("PAPER_ORCHESTRA_CLI_MODEL") or DEFAULT_CLAUDE_MODEL
    command = [
        _require_executable("claude"),
        "-p",
        "--setting-sources",  # no hooks or user settings; --bare would ignore the plan login
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--output-format",
        "json",  # includes token usage
        "--model",
        model,
        "--tools",
        ",".join(tools),
    ]
    if tools:
        command += ["--allowedTools", ",".join(tools)]

    result = _run(command, prompt, workdir)
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        detail = (result.stderr or result.stdout or "")[-800:]
        raise RuntimeError(f"claude failed: {detail}")
    if result.returncode != 0 or payload.get("is_error"):
        raise RuntimeError(f"claude failed: {str(payload.get('result'))[-800:]}")
    _record_usage(model, payload, search)
    return payload.get("result") or ""


def generate_text(contents: Any, generation_configs: Optional[dict] = None) -> str:
    backend = active_backend()
    configs = generation_configs or {}

    with tempfile.TemporaryDirectory(prefix="paper_orchestra_cli_") as workdir:
        texts, images = _collect_parts(contents, workdir)
        prompt = _build_prompt(texts, configs, images)
        search = _wants_web_search(configs)
        caller = _call_codex if backend == "codex" else _call_claude
        text = caller(prompt, images, search, workdir)

    if not text or not text.strip():
        raise RuntimeError(f"{backend} returned an empty response.")
    return text


def gemini_response(contents: Any, generation_configs: Optional[dict] = None):
    """Response stub with the `.text` attribute the Gemini call path reads."""
    return SimpleNamespace(text=generate_text(contents, generation_configs))


def openai_response(messages: List[dict]):
    """Response stub with the shape the OpenAI call path reads."""
    system_parts = [m["content"] for m in messages if m["role"] in ("system", "developer")]
    user_parts = [m["content"] for m in messages if m["role"] == "user"]
    configs = {"system_instruction": system_parts} if system_parts else {}
    text = generate_text(user_parts, configs)
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])
