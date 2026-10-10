"""Thin wrapper so the analyst works with Gemini (free tier) or Claude.

Every client exposes one method:
    chat(system: str, messages: list[{"role": "user"|"assistant", "content": str}], max_tokens: int) -> str

Choose the provider with the ANALYST_PROVIDER env var ("gemini" or "claude"),
or let it be detected from whichever API key you have set.
Override models with GEMINI_MODEL / CLAUDE_MODEL.
"""
from __future__ import annotations

import os
import time

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"  # Google retires older models; override with GEMINI_MODEL
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"


class GeminiLLM:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai  # pip install google-genai

        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def chat(self, system: str, messages: list[dict], max_tokens: int = 1500) -> str:
        from google.genai import types

        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        # Gemini thinking models spend part of the output budget on internal "thinking",
        # so give them more room than the visible answer needs. Low thinking is plenty
        # for writing pandas code and is much faster than the default.
        def make_config(low_thinking: bool):
            kwargs = dict(
                system_instruction=system,
                max_output_tokens=max(max_tokens, 4096),
                temperature=0,
            )
            if low_thinking:
                try:
                    kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
                except Exception:  # older SDK without thinking_level
                    pass
            return types.GenerateContentConfig(**kwargs)

        low_thinking = True
        for attempt in range(5):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=make_config(low_thinking)
                )
                return resp.text or ""
            except Exception as e:
                msg = str(e)
                if low_thinking and "thinking" in msg.lower():
                    low_thinking = False  # this model rejects the setting: retry without it
                    continue
                # free tier is rate limited: back off and retry
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg) and attempt < 4:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        return ""


class ClaudeLLM:
    name = "claude"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        import anthropic

        self.model = model or os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def chat(self, system: str, messages: list[dict], max_tokens: int = 1500) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=messages
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def detect_provider() -> str:
    explicit = os.environ.get("ANALYST_PROVIDER", "").lower()
    if explicit in ("gemini", "claude"):
        return explicit
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return "gemini"


def env_key(provider: str) -> str | None:
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return os.environ.get("ANTHROPIC_API_KEY")


def get_client(provider: str | None = None, api_key: str | None = None, model: str | None = None):
    provider = provider or detect_provider()
    cls = GeminiLLM if provider == "gemini" else ClaudeLLM
    return cls(api_key=api_key, model=model or None)
