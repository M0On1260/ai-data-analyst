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
DEFAULT_GEMINI_FALLBACKS = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"  # lighter, faster models
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"


class GeminiLLM:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai  # pip install google-genai
        from google.genai import types

        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        fallbacks = os.environ.get("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS)
        self.fallbacks = [m.strip() for m in fallbacks.split(",") if m.strip() and m.strip() != self.model]
        self.last_model = self.model  # model that actually answered the last call
        # Cap each request so a stuck call fails over to a faster model instead of hanging.
        kwargs = {"http_options": types.HttpOptions(timeout=int(os.environ.get("GEMINI_TIMEOUT_S", "45")) * 1000)}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = genai.Client(**kwargs)

    @staticmethod
    def _is_slow_or_busy(msg: str) -> bool:
        m = msg.lower()
        return any(x in m for x in ("429", "resource_exhausted", "503", "504", "unavailable",
                                    "timed out", "timeout", "deadline", "404", "not_found"))

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

        last_error: Exception | None = None
        for model in [self.model] + self.fallbacks:
            low_thinking = True
            for attempt in range(2):  # at most one quick retry per model
                try:
                    resp = self.client.models.generate_content(
                        model=model, contents=contents, config=make_config(low_thinking)
                    )
                    self.last_model = model
                    return resp.text or ""
                except Exception as e:
                    last_error = e
                    msg = str(e)
                    if low_thinking and "thinking" in msg.lower():
                        low_thinking = False  # this model rejects the setting: retry without it
                        continue
                    if not self._is_slow_or_busy(msg):
                        raise  # a real error (bad key, bad request...): don't hide it
                    print(f"[gemini] {model} failed ({msg[:100]}); "
                          f"{'retrying' if attempt == 0 and '429' in msg else 'trying next model'}", flush=True)
                    if attempt == 0 and ("429" in msg or "RESOURCE_EXHAUSTED" in msg):
                        time.sleep(4)  # brief pause, then one retry on the same model
                        continue
                    break  # timeout / overloaded / retired: move to the next model
        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


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
