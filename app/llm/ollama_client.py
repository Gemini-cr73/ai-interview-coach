from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class OllamaClient:
    """
    Lightweight client for local Ollama inference.
    Includes strict-JSON structured coaching helper.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = 60,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or OLLAMA_MODEL
        self.timeout = int(timeout)

    def health_check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def generate_text(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
        model: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if system:
            payload["system"] = system

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 900,
        model: str | None = None,
    ) -> dict[str, Any]:
        """
        Requests JSON-only output. If the model still adds text, we extract
        the first JSON object and parse it.
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            # If your Ollama build supports JSON mode, this helps.
            "format": "json",
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if system:
            payload["system"] = system

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        raw = (resp.json().get("response") or "").strip()
        if not raw:
            return {}

        # Direct parse
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Fallback: extract first JSON object
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
