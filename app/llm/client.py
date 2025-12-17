"""LLM client abstraction layer."""

import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from app.core.config import settings


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a completion."""
        pass

    @abstractmethod
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> dict:
        """Generate a JSON completion."""
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        """Generate an embedding for text."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.embedding_model = settings.embedding_model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a completion using OpenAI."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> dict:
        """Generate a JSON completion using OpenAI."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)

    async def get_embedding(self, text: str) -> list[float]:
        """Generate an embedding using OpenAI."""
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model
        # Anthropic doesn't have embeddings, fallback to OpenAI for embeddings
        self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate a completion using Anthropic."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> dict:
        """Generate a JSON completion using Anthropic."""
        enhanced_prompt = f"{user_prompt}\n\nRespond ONLY with valid JSON, no other text."
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": enhanced_prompt}],
        )
        content = response.content[0].text
        # Try to extract JSON from response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            raise

    async def get_embedding(self, text: str) -> list[float]:
        """Generate an embedding using OpenAI (Anthropic doesn't have embeddings)."""
        response = await self._openai_client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding


def get_llm_client() -> BaseLLMClient:
    """Get the configured LLM client."""
    if settings.llm_provider == "anthropic":
        return AnthropicClient()
    return OpenAIClient()
