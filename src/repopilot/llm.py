"""Minimal OpenAI-compatible chat client using the standard library."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ChatClient:
    def __init__(self, model: str, api_base: str = None, api_key: str = None, timeout: int = 90):
        self.model = model
        self.api_base = (api_base or os.environ.get("REPOPILOT_API_BASE") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        if not self.model:
            raise ValueError("set --model or REPOPILOT_MODEL")
        if not self.api_key:
            raise ValueError("set OPENAI_API_KEY or pass --api-key")

    def complete(self, messages):
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": 0}).encode("utf-8")
        request = Request(
            self.api_base + "/chat/completions",
            data=payload,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError("chat API returned HTTP {}".format(exc.code))
        except URLError as exc:
            raise RuntimeError("chat API request failed: {}".format(exc.reason))
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("chat API response has no assistant message")
