"""Probe the configured LLM endpoint (.env: LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).
Never prints the key. Tries the Responses API first, then Chat Completions, and reports which
parameters the model accepts (GPT-5.x rejects `max_tokens`/`temperature` on some variants)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env():
    env = {}
    for ln in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


env = load_env()
base, key, model = env["LLM_BASE_URL"], env["LLM_API_KEY"], env["LLM_MODEL"]
print(f"base={base} model={model} key_len={len(key)} key_prefix={key[:3]}***")

import openai  # noqa: E402

print("openai sdk", openai.__version__)
client = openai.OpenAI(base_url=base, api_key=key, timeout=120)

SYSTEM = "You are a control engineer. Answer with a single JSON object only."
USER = ('A wind turbine PPO residual pitch agent trains on reward r = w_P*(P/P_base-1) in Region 2 and '
        'w_w*exp(-|d_wg|/tau) in Region 3, minus lambda_L*|M_root|/M_ref. Given w_P=220, w_w=20, '
        'lambda_L=1 and the observation that blade-root DEL did not decrease after 100 episodes while '
        'speed regulation improved 20%, propose new lambda_L and one-sentence rationale. '
        'Return {"lambda_L": number, "rationale": string}.')


def try_call(name, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f"[OK ] {name} ({time.time() - t0:.1f}s): {out[:300]!r}")
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if key in msg:
            msg = msg.replace(key, "***")
        print(f"[ERR] {name} ({time.time() - t0:.1f}s): {type(e).__name__}: {msg[:400]}")
        return False


def chat(**kw):
    r = client.chat.completions.create(model=model, messages=[{"role": "system", "content": SYSTEM},
                                                              {"role": "user", "content": USER}], **kw)
    return r.choices[0].message.content or json.dumps(r.model_dump())[:300]


def responses(**kw):
    r = client.responses.create(model=model, instructions=SYSTEM, input=USER, **kw)
    return getattr(r, "output_text", None) or json.dumps(r.model_dump())[:300]


results = {}
results["chat.plain"] = try_call("chat.completions (no extra params)", lambda: chat())
results["chat.max_completion_tokens"] = try_call("chat.completions max_completion_tokens=400",
                                                 lambda: chat(max_completion_tokens=400))
results["chat.max_tokens"] = try_call("chat.completions max_tokens=400", lambda: chat(max_tokens=400))
results["chat.temperature"] = try_call("chat.completions temperature=0.2", lambda: chat(temperature=0.2))
results["chat.reasoning_effort"] = try_call("chat.completions reasoning_effort=low",
                                            lambda: chat(reasoning_effort="low"))
results["chat.json_mode"] = try_call("chat.completions response_format=json_object",
                                     lambda: chat(response_format={"type": "json_object"}))
if hasattr(client, "responses"):
    results["responses.plain"] = try_call("responses (no extra params)", lambda: responses())
    results["responses.reasoning"] = try_call("responses reasoning={'effort':'low'}",
                                              lambda: responses(reasoning={"effort": "low"}))
    results["responses.max_output_tokens"] = try_call("responses max_output_tokens=400",
                                                      lambda: responses(max_output_tokens=400))
else:
    print("responses API not in this SDK version")
print("\nSUMMARY:", json.dumps(results, indent=1))
