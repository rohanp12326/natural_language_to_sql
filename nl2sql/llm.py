"""Z.ai GLM model initialization utilities."""

import os
from typing import Literal

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

# Default Z.ai API endpoints
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"

# Cost-effective model defaults
# glm-4-flash is free and high-speed for general routing, chat, and generation
# glm-4-air is ultra-low cost ($0.20/1M) with strong reasoning for syntax fixing and data analysis
DEFAULT_FAST_MODEL = "glm-4-flash"
DEFAULT_POWERFUL_MODEL = "glm-4-air"
DEFAULT_EMBEDDING_MODEL = "embedding-3"


def get_zai_api_key() -> str | None:
    """Retrieve Z.ai API key from environment."""
    return (
        os.getenv("ZAI_API_KEY")
        or os.getenv("ZHIPUAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def get_zai_base_url() -> str:
    """Retrieve Z.ai base URL from environment or return default."""
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
    """Initialize a Z.ai GLM chat model via the OpenAI-compatible API.

    Args:
        model: Specific model name to use. If None, resolved based on model_type.
        model_type: "fast" (default: glm-4-flash) or "powerful" (default: glm-4-air).
        temperature: Sampling temperature (0.0 for deterministic tasks).
        **kwargs: Additional kwargs passed to ChatOpenAI.

    Returns:
        Configured ChatOpenAI instance connected to Z.ai GLM endpoint.
    """
    api_key = get_zai_api_key()
    base_url = get_zai_base_url()

    if not model:
        if model_type == "powerful":
            model = os.getenv("ZAI_MODEL_POWERFUL", DEFAULT_POWERFUL_MODEL)
        else:
            model = os.getenv("ZAI_MODEL_FAST", DEFAULT_FAST_MODEL)

    logger.debug(f"Initializing Z.ai GLM model: {model} (base_url: {base_url})")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def get_embeddings(model: str | None = None, **kwargs) -> OpenAIEmbeddings:
    """Initialize Z.ai embeddings via the OpenAI-compatible API.

    Args:
        model: Specific embedding model name (default: embedding-3).
        **kwargs: Additional kwargs passed to OpenAIEmbeddings.

    Returns:
        Configured OpenAIEmbeddings instance connected to Z.ai GLM endpoint.
    """
    api_key = get_zai_api_key()
    base_url = get_zai_base_url()
    model = model or os.getenv("ZAI_MODEL_EMBEDDING", DEFAULT_EMBEDDING_MODEL)

    logger.debug(f"Initializing Z.ai embeddings: {model} (base_url: {base_url})")

    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
