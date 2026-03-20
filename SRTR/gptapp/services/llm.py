import os
import logging
import requests
from typing import List, Dict

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 2048) -> str:
        """
        original _openai_chat
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                logger.error("OpenAI error %s: %s", resp.status_code, resp.text[:500])
                raise RuntimeError(f"OpenAI API error: {resp.status_code}")
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.exception("OpenAI request failed: %s", e)
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        original _get_embedding
        """
        url = f"{self.base_url}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.embedding_model,
            "input": text
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding API error: {resp.status_code}")
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.exception("Embedding request failed: %s", e)
            raise