"""Relay one JSON-mode chat completion through the Windows host (2026-09-16).

When the WSL VM loses outbound network while Windows keeps it, `LLMClient` (set WTRL_LLM_WINPY to a Windows
python.exe) writes the request to a file and runs this script with Windows Python through WSL interop. The request
body is the one LLMClient sends (model, messages, JSON response format, reasoning_effort, max_completion_tokens);
older openai packages receive the last two through extra_body. Credentials are read from the repository's .env by
the same loader; nothing secret is written to the response file.

    python.exe llm/win_relay.py <request.json> <response.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.client import load_env  # noqa: E402


def main():
    req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    env = load_env()
    import openai
    client = openai.OpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"], timeout=float(req.get("timeout", 180)))
    kw = dict(model=env["LLM_MODEL"], messages=req["messages"], response_format={"type": "json_object"})
    extra = {"reasoning_effort": req["reasoning_effort"], "max_completion_tokens": req["max_completion_tokens"]}
    try:
        r = client.chat.completions.create(**kw, **extra)
    except TypeError:                         # openai < 1.40 does not know these keyword arguments
        r = client.chat.completions.create(**kw, extra_body=extra)
    usage = None
    if r.usage:
        usage = {"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}
    out_path.write_text(json.dumps({"content": r.choices[0].message.content or "", "usage": usage}), encoding="utf-8")


if __name__ == "__main__":
    main()
