"""
Model access for Miss Pedant.

One function, stream_chat(), and three interchangeable backends picked by
LLM_PROVIDER in .env:
  - mlx        local mlx_lm.server (OpenAI-compatible), the default on Apple Silicon
  - anthropic  Claude API
  - openai     OpenAI API (or anything else that speaks the OpenAI protocol)

The frontend builds its own prompts, so this layer only needs to stream text.
"""
import os
from typing import Iterator

PROVIDER = os.getenv("LLM_PROVIDER", "mlx").lower()

# A light, constant system prompt. The real instructions (persona, citation
# rules, output format) travel inside the user turns the frontend sends.
SYSTEM = (
    "You are Miss Pedant, a meticulous research assistant for scientists, "
    "researchers and students. Follow the instructions in the user's message "
    "exactly, including any requested output format."
)


def model_for(tier: str) -> str:
    """Map the frontend's tier ('quick' / 'default' / 'complex') to a model name."""
    if PROVIDER == "anthropic":
        if tier == "quick":
            return os.getenv("ANTHROPIC_MODEL_QUICK", "claude-haiku-4-5-20251001")
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    if PROVIDER == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # mlx serves one model; every tier uses it
    return os.getenv("MLX_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")


def describe() -> dict:
    return {"provider": PROVIDER, "model": model_for("default")}


def _merge(messages: list[dict]) -> list[dict]:
    """Normalize turns: merge consecutive same-role turns, drop empties,
    and make sure the conversation starts on a user turn. Some APIs reject
    anything else, and the frontend intentionally sends an instructions turn
    followed by another user turn."""
    out: list[dict] = []
    for m in messages:
        role = "assistant" if m.get("role") == "assistant" else "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": role, "content": content})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def stream_chat(messages: list[dict], tier: str = "default", max_tokens: int = 1800) -> Iterator[str]:
    """Yield the answer text in small pieces as the model writes it."""
    msgs = _merge(messages)
    if not msgs:
        raise ValueError("The conversation is empty.")
    if PROVIDER == "anthropic":
        yield from _anthropic(msgs, tier, max_tokens)
    else:
        yield from _openai_compatible(msgs, tier, max_tokens)


def _openai_compatible(msgs, tier, max_tokens):
    from openai import OpenAI

    if PROVIDER == "openai":
        client = OpenAI()  # reads OPENAI_API_KEY (and OPENAI_BASE_URL if set)
    else:
        # mlx_lm.server ignores the key, but the client insists on one
        client = OpenAI(base_url=os.getenv("MLX_BASE_URL", "http://127.0.0.1:8080/v1"), api_key="local")

    stream = client.chat.completions.create(
        model=model_for(tier),
        messages=[{"role": "system", "content": SYSTEM}, *msgs],
        stream=True,
        max_tokens=max_tokens,
        temperature=0.3,  # low temperature: we want faithful, not creative
    )
    for event in stream:
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def _anthropic(msgs, tier, max_tokens):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    with client.messages.stream(
        model=model_for(tier),
        max_tokens=max_tokens,
        system=SYSTEM,
        messages=msgs,
        temperature=0.3,
    ) as stream:
        for text in stream.text_stream:
            yield text
