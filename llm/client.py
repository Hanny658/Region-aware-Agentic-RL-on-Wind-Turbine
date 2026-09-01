"""Thin LLM client: reads .env (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL), JSON-mode chat
completions with retries, and a transcript log (jsonl) for reproducibility. The key is never
logged or printed.

Verified 2026-08-29 against gpt-5.6-luna: `max_tokens` and `temperature != 1` are rejected;
`max_completion_tokens`, `reasoning_effort` and `response_format={"type": "json_object"}` work.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> dict:
    p = path or PROJ / ".env"
    env = {}
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


class LLMClient:
    def __init__(self, transcript_path: str | Path | None = None, reasoning_effort: str = "medium",
                 max_completion_tokens: int = 2000, retries: int = 3, timeout: float = 180.0):
        env = load_env()
        missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if k not in env]
        if missing:
            raise RuntimeError(f"missing {missing} in .env / environment")
        import openai
        self.model = env["LLM_MODEL"]
        self.client = openai.OpenAI(base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"], timeout=timeout)
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.retries = retries
        self.transcript = Path(os.path.expanduser(transcript_path)) if transcript_path else None
        self.n_calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def ask_json(self, system: str, user: str, tag: str = "") -> dict:
        last_err = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    reasoning_effort=self.reasoning_effort,
                    max_completion_tokens=self.max_completion_tokens,
                )
                text = r.choices[0].message.content or ""
                out = json.loads(text)
                self.n_calls += 1
                if r.usage:
                    self.usage["prompt_tokens"] += r.usage.prompt_tokens or 0
                    self.usage["completion_tokens"] += r.usage.completion_tokens or 0
                self._log({"tag": tag, "attempt": attempt, "model": self.model, "elapsed_s": time.time() - t0,
                           "system": system, "user": user, "response": out,
                           "usage": r.usage.model_dump() if r.usage else None})
                return out
            except Exception as e:  # noqa: BLE001
                last_err = e
                self._log({"tag": tag, "attempt": attempt, "error": f"{type(e).__name__}: {str(e)[:500]}"})
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last_err}")

    def _log(self, rec: dict):
        if self.transcript is None:
            return
        self.transcript.parent.mkdir(parents=True, exist_ok=True)
        with open(self.transcript, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **rec}, ensure_ascii=False) + "\n")
