# llm_client.py
import os
import httpx
import json

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL") or "https://aipipe.org/openrouter/v1/chat/completions"
DEFAULT_MODEL = os.getenv("LLM_MODEL") or "gpt-5-nano"  # change to your model

HEADERS = {
    "Authorization": f"Bearer {LLM_API_KEY}" if LLM_API_KEY else "",
    "Content-Type": "application/json",
}

async def ask_llm(system_prompt: str, user_prompt: str, model: str = None, max_tokens: int = 512):
    """
    Async wrapper that calls an OpenAI-style chat completions endpoint using httpx.
    Returns assistant text (string) or raises on error.
    """
    if not LLM_API_KEY:
        # No key configured — return None so caller can fallback
        return None

    model = model or DEFAULT_MODEL
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_output_tokens": max_tokens,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(LLM_API_URL, headers=HEADERS, json=body)
        resp.raise_for_status()
        j = resp.json()
        # adapt to provider response shape
        # common OpenAI-style: j["choices"][0]["message"]["content"]
        choices = j.get("choices", [])
        if not choices:
            return None
        first = choices[0]
        msg = first.get("message", {})
        content = msg.get("content")
        # sometimes provider returns text in 'text' field
        if content is None:
            content = first.get("text")
        return content
