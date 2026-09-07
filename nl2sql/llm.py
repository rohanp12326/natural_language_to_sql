"""LLM and Embedding Model Initialization Utilities.

This module provides factory functions for initializing language and embedding models.
It is configured for OpenAI-compatible providers such as Z.ai (Zhipu AI GLM series)
or OpenAI directly.

Architecture Design - Dual Model Tier:
  1. Fast Model (`glm-5.3-flash`):
     - Low latency, high throughput, cost-effective.
     - Used for: Intent classification, conversational chat, and initial SQL generation.
  2. Powerful Reasoning Model (`glm-4.5-air` / `glm-5.3`):
     - Deep reasoning capabilities for complex tasks.
     - Used for: Self-healing SQL syntax error correction and multi-table data result analysis.

Embedding Strategy:
  - Local Embeddings (`LocalEmbeddings`): Uses ChromaDB's built-in ONNX runtime
    with the lightweight `all-MiniLM-L6-v2` model. This runs 100% locally with zero API costs,
    zero rate limits, and requires no external network call.
  - Remote Embeddings (`OpenAIEmbeddings`): Fallback for cloud-based embedding endpoints.
"""

import os
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

# Default Z.ai / Zhipu AI API endpoints
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"

# Default model identifiers
DEFAULT_FAST_MODEL = "glm-5.3-flash"
DEFAULT_POWERFUL_MODEL = "glm-4.5-air"
DEFAULT_EMBEDDING_MODEL = "local"


class LocalEmbeddings(Embeddings):
    """Zero-dependency, zero-cost local embedding generator using ChromaDB's ONNX all-MiniLM-L6-v2.
    
    Why this is useful:
      Many coding plan providers (including Z.ai) do not provide specialized text embedding
      models. This class provides standard LangChain `Embeddings` compliance entirely in-process
      using an ONNX-quantized transformer model.
    """

    def __init__(self) -> None:
        """Initialize local ONNX embedding function from chromadb."""
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute vector embeddings for a list of document chunks.

        Args:
            texts: List of text strings to vectorize.

        Returns:
            list[list[float]]: List of 384-dimensional dense vectors.
        """
        embeddings = self._ef(texts)
        return [
            e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings
        ]

    def embed_query(self, text: str) -> list[float]:
        """Compute vector embedding for a single user search query.

        Args:
            text: Query string to vectorize.

        Returns:
            list[float]: 384-dimensional dense vector.
        """
        embeddings = self._ef([text])
        e = embeddings[0]
        return e.tolist() if hasattr(e, "tolist") else list(e)


def get_zai_api_key() -> str | None:
    """Retrieve API key from environment, checking common provider aliases."""
    return (
        os.getenv("ZAI_API_KEY")
        or os.getenv("ZHIPUAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def get_zai_base_url() -> str:
    """Retrieve provider base URL from environment or return Z.ai default."""
    return (
        os.getenv("ZAI_BASE_URL")
        or os.getenv("ZHIPUAI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or DEFAULT_ZAI_BASE_URL
    )


def get_chat_model(
    model: str | None = None,
    model_type: Literal["fast", "powerful"] = "fast",
    temperature: float = 0.0,
    **kwargs,
) -> ChatOpenAI:
    """Initialize a chat model instance via the OpenAI-compatible API protocol.

    Args:
        model: Explicit model name. If None, resolved automatically based on `model_type`.
        model_type: "fast" (default: glm-5.3-flash) or "powerful" (default: glm-4.5-air).
        temperature: Sampling randomness. Set to 0.0 for deterministic SQL and classification.
        **kwargs: Additional keyword arguments forwarded to `ChatOpenAI`.

    Returns:
        ChatOpenAI: Configured LangChain model instance.
    """
    api_key = get_zai_api_key()
    base_url = get_zai_base_url()

    # Determine which model to use if not explicitly specified
    if not model:
        if model_type == "powerful":
            model = os.getenv("ZAI_MODEL_POWERFUL", DEFAULT_POWERFUL_MODEL)
        else:
            model = os.getenv("ZAI_MODEL_FAST", DEFAULT_FAST_MODEL)

    logger.debug(f"Initializing LLM: {model} (base_url: {base_url})")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def get_embeddings(model: str | None = None, **kwargs) -> Embeddings:
    """Initialize embeddings instance for the knowledge base and PGVector store.

    Args:
        model: Model identifier. Defaults to 'local' for zero-cost ONNX embeddings.
        **kwargs: Additional arguments passed to OpenAIEmbeddings if remote.

    Returns:
        Embeddings: Configured LangChain Embeddings instance.
    """
    model = model or os.getenv("ZAI_MODEL_EMBEDDING", DEFAULT_EMBEDDING_MODEL)

    # Use local ONNX embeddings if requested or default
    if model.lower() in ("local", "default", "onnx"):
        logger.debug("Initializing local ONNX all-MiniLM embeddings (0 external API cost)")
        return LocalEmbeddings()

    # Otherwise, initialize remote embeddings via OpenAI-compatible endpoint
    api_key = get_zai_api_key()
    base_url = get_zai_base_url()

    logger.debug(f"Initializing remote embeddings: {model} (base_url: {base_url})")
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
        **kwargs,
    )

