"""Module for handling the open AI client library."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import openai

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client

from .const import (
    CONF_BASE_URL,
)

_LOGGER = logging.getLogger(__name__)


_TIMEOUT = 10.0
_MAX_MODELS = 100


async def async_create_client(
    hass: HomeAssistant, config_entry_data: Mapping[str, Any]
) -> openai.AsyncOpenAI:
    """Create a new OpenAI client."""

    def _create() -> openai.AsyncOpenAI:
        """Get OpenAI client."""
        _LOGGER.debug("Creating OpenAI client: %s", config_entry_data)
        return openai.AsyncOpenAI(
            api_key=config_entry_data[CONF_API_KEY],
            base_url=config_entry_data[CONF_BASE_URL],
            http_client=get_async_client(hass),
        )

    return await hass.async_add_executor_job(_create)


async def async_list_models(client: openai.AsyncOpenAI) -> list[str]:
    """Return a list of models supported by the client."""

    models = []
    async for model in client.with_options(timeout=_TIMEOUT).models.list():
        models.append(model.id)
        if len(models) >= _MAX_MODELS:
            break
    return models


_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
_TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "test_function",
            "description": "Test function.",
        },
    }
]
_TEST_MAX_TOKENS = 3  # we don't care about the response


async def async_validate_completions(
    client: openai.AsyncOpenAI,
    model: str,
    stream: bool = False,
) -> bool:
    """Validate that we can speak to the model over the completions API."""
    try:
        result = await client.chat.completions.create(
            model=model,
            messages=_TEST_MESSAGES,  # type: ignore[arg-type]
            tools=_TEST_TOOLS,  # type: ignore[arg-type]
            max_tokens=_TEST_MAX_TOKENS,
            stream=stream,
        )
    except openai.OpenAIError as err:
        _LOGGER.info("Error validating model %s (stream=%s): %s", model, stream, err)
        # Can't talk to the model either because it doesn't support the model, stream, etc
        return False

    if stream:
        try:
            async for event in result:  # type: ignore[union-attr]
                if event.choices[0].finish_reason is not None:
                    continue
        except openai.OpenAIError as err:
            _LOGGER.error("Error streaming completion result: %s", err)
            return False

    return True
