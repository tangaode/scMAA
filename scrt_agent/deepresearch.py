"""Deep Research wrapper for scMAA."""

from __future__ import annotations

import os

import openai


class DeepResearcher:
    """Small wrapper over OpenAI Deep Research models."""

    def __init__(self, openai_api_key: str) -> None:
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = os.environ.get("SCRT_DEEP_RESEARCH_MODEL", "o4-mini-deep-research")

    def _extract_output_text(self, response: object) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                ctype = getattr(content, "type", None)
                if ctype in {"output_text", "text"}:
                    value = getattr(content, "text", None)
                    if isinstance(value, str):
                        parts.append(value)
                    elif isinstance(value, dict) and "value" in value:
                        parts.append(str(value["value"]))
        return "\n".join(parts).strip()

    def research(self, prompt: str, max_output_tokens: int = 32000) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            tools=[{"type": "web_search_preview"}],
            max_output_tokens=max_output_tokens,
        )
        return self._extract_output_text(response)

