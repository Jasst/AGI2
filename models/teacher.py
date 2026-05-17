"""
👨‍🏫 models/teacher.py — Async Teacher LLM (LM Studio / OpenAI-compatible API)
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("AdvancedAgent_v4")


class TeacherLLM:
    """Thin async wrapper around any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, url: str, api_key: str) -> None:
        self.url = url
        self.api_key = api_key
        self._session = None
        self.total_calls: int = 0

    # ------------------------------------------------------------------
    async def connect(self) -> None:
        if self._session is None:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Tuple[str, List[float]]:
        """Returns (response_text, token_logprobs_or_empty)."""
        if self._session is None:
            await self.connect()

        self.total_calls += 1
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._session.post(self.url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    return text, []
                logger.warning(f"Teacher API returned HTTP {resp.status}")
                return "", []
        except Exception as exc:
            logger.error(f"Teacher LLM error: {exc}")
            return "", []
