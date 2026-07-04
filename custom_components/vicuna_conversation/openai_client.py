"""Module for handling the open AI client library."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
import logging
from typing import Any, cast

import openai
from openai._streaming import AsyncStream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.httpx_client import get_async_client

from .const import (
    CONF_BASE_URL,
    DOMAIN,
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
    with api_error_handler():
        async for model in client.with_options(timeout=_TIMEOUT).models.list():
            models.append(model.id)
            if len(models) >= _MAX_MODELS:
                break
    return models


_TEST_MESSAGES: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
_TEST_TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "test_function",
            "description": "Test function.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
_TEST_MAX_TOKENS = 3  # we don't care about the response


async def async_validate_completions(
    client: openai.AsyncOpenAI,
    model: str,
    stream: bool = False,
) -> None:
    """Validate that we can speak to the model over the completions API."""
    with api_error_handler():
        result = await client.chat.completions.create(
            model=model,
            messages=_TEST_MESSAGES,
            tools=_TEST_TOOLS,
            max_tokens=_TEST_MAX_TOKENS,
            stream=stream,
        )

        if stream:
            stream_result = cast(AsyncStream[ChatCompletionChunk], result)
            async for event in stream_result:
                if not event.choices:
                    continue
                if event.choices[0].finish_reason is not None:
                    continue


def _extract_error_message(err: openai.APIStatusError) -> str:
    """Extract a clean error message from an APIStatusError response or message."""
    error_message = ""
    if err.response is not None:
        try:
            json_data = err.response.json()
            if isinstance(json_data, dict) and "error" in json_data:
                error_message = json_data["error"].get("message") or ""
        except ValueError:
            pass
    return error_message or err.message or str(err)


@contextmanager
def api_error_handler() -> Generator[None]:
    """Context manager to handle API errors and translate them to HomeAssistantErrors."""
    try:
        yield
    except openai.APITimeoutError as err:
        _LOGGER.error("Timeout talking to API: %s", err)
        error_message = err.message or str(err)
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="timeout",
            translation_placeholders={"message": error_message},
        ) from err
    except openai.APIConnectionError as err:
        _LOGGER.error("Connection error talking to API: %s", err)
        error_message = err.message or str(err)
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="cannot_connect",
            translation_placeholders={"message": error_message},
        ) from err
    except openai.AuthenticationError as err:
        _LOGGER.error("Authentication error talking to API: %s", err)
        error_message = _extract_error_message(err)
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="invalid_auth",
            translation_placeholders={"message": error_message},
        ) from err
    except openai.APIStatusError as err:
        _LOGGER.error("Status error talking to API: %s", err)
        error_message = _extract_error_message(err)

        if err.status_code == 402:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="quota_exceeded",
                translation_placeholders={"message": error_message},
            ) from err

        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="api_error",
            translation_placeholders={"message": error_message},
        ) from err
    except openai.OpenAIError as err:
        _LOGGER.error("Generic error talking to API: %s", err)
        error_message = getattr(err, "message", None) or str(err)
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="api_error",
            translation_placeholders={"message": error_message},
        ) from err
