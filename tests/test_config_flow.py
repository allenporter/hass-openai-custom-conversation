"""Tests for the config flow."""

from typing import Generator, Any
from unittest.mock import patch, Mock

import pytest
import openai

from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API


from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
)

from custom_components.vicuna_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_STREAMING,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TOP_P,
)
from custom_components.vicuna_conversation.config_flow import RECOMMENDED_OPTIONS


@pytest.fixture(name="mock_setup")
def mock_setup(hass: HomeAssistant) -> Generator[Mock]:
    """Mock the setup of the integration."""
    with (
        patch(
            f"custom_components.{DOMAIN}.async_setup_entry", return_value=True
        ) as mock_setup,
    ):
        yield mock_setup


async def test_config_flow(
    hass: HomeAssistant,
    mock_setup: Mock,
) -> None:
    """Test selecting a model in the configuration flow."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM
    assert not result.get("errors")

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "api_key": "sk-0000000000000000000",
            "base_url": "http://llama-cublas.llama:8000/v1",
        },
    )
    assert result.get("type") is FlowResultType.FORM
    assert not result.get("errors")

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_CHAT_MODEL: "gpt-4",
        },
    )
    await hass.async_block_till_done()

    assert result.get("type") is FlowResultType.CREATE_ENTRY
    assert result.get("title") == "Custom OpenAI"
    assert result.get("data") == {
        "api_key": "sk-0000000000000000000",
        "base_url": "http://llama-cublas.llama:8000/v1",
    }
    assert result["options"] == {
        **RECOMMENDED_OPTIONS,
        CONF_CHAT_MODEL: "gpt-4",
        CONF_STREAMING: True,
    }

    assert len(mock_setup.mock_calls) == 1


async def test_config_flow_fail_completion(
    hass: HomeAssistant,
    mock_setup: Mock,
    mock_completion: Mock,
) -> None:
    """Test config flow where the API does not support streaming."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM
    assert not result.get("errors")

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "api_key": "sk-0000000000000000000",
            "base_url": "http://llama-cublas.llama:8000/v1",
        },
    )
    assert result.get("type") is FlowResultType.FORM
    assert not result.get("errors")

    def fail_all(stream: bool | None, **kwargs: Any) -> None:
        """Allow first check to succeed by fail streaming."""
        raise openai.OpenAIError("Invalid request")

    mock_completion.side_effect = fail_all

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_CHAT_MODEL: "gpt-4",
        },
    )
    await hass.async_block_till_done()

    assert result.get("type") is FlowResultType.FORM
    assert result.get("errors") == {"base": "cannot_connect"}

    assert len(mock_setup.mock_calls) == 0


async def test_config_flow_no_streaming(
    hass: HomeAssistant,
    mock_setup: Mock,
    mock_completion: Mock,
) -> None:
    """Test config flow where the API does not support streaming."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM
    assert not result.get("errors")

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "api_key": "sk-0000000000000000000",
            "base_url": "http://llama-cublas.llama:8000/v1",
        },
    )
    assert result.get("type") is FlowResultType.FORM
    assert not result.get("errors")

    def fail_streaming(stream: bool | None, **kwargs: Any) -> None:
        """Allow first check to succeed by fail streaming."""
        if stream:
            raise openai.OpenAIError("Invalid request")

    mock_completion.side_effect = fail_streaming

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_CHAT_MODEL: "gpt-4",
        },
    )
    await hass.async_block_till_done()

    assert result.get("type") is FlowResultType.CREATE_ENTRY
    assert result.get("title") == "Custom OpenAI"
    assert result.get("data") == {
        "api_key": "sk-0000000000000000000",
        "base_url": "http://llama-cublas.llama:8000/v1",
    }
    assert result["options"] == {
        **RECOMMENDED_OPTIONS,
        CONF_CHAT_MODEL: "gpt-4",
        CONF_STREAMING: False,
    }

    assert len(mock_setup.mock_calls) == 1


async def test_options(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test the options form."""
    options_flow = await hass.config_entries.options.async_init(
        mock_config_entry.entry_id
    )
    options = await hass.config_entries.options.async_configure(
        options_flow["flow_id"],
        {
            "prompt": "Speak like a pirate",
            "max_tokens": 200,
        },
    )
    await hass.async_block_till_done()
    assert options["type"] is FlowResultType.CREATE_ENTRY
    assert options["data"]["prompt"] == "Speak like a pirate"
    assert options["data"]["max_tokens"] == 200
    assert options["data"][CONF_CHAT_MODEL] == RECOMMENDED_CHAT_MODEL


@pytest.mark.parametrize(
    ("current_options", "new_options", "expected_options"),
    [
        (
            {
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: "none",
                CONF_PROMPT: "bla",
            },
            {
                CONF_RECOMMENDED: False,
                CONF_PROMPT: "Speak like a pirate",
                CONF_TEMPERATURE: 0.3,
            },
            {
                CONF_RECOMMENDED: False,
                CONF_PROMPT: "Speak like a pirate",
                CONF_TEMPERATURE: 0.3,
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_TOP_P: RECOMMENDED_TOP_P,
                CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
            },
        ),
        (
            {
                CONF_RECOMMENDED: False,
                CONF_PROMPT: "Speak like a pirate",
                CONF_TEMPERATURE: 0.3,
                CONF_CHAT_MODEL: "gpt-4",
                CONF_TOP_P: RECOMMENDED_TOP_P,
                CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
            },
            {
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: "assist",
                CONF_PROMPT: "",
            },
            {
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: "assist",
                CONF_PROMPT: "",
                CONF_CHAT_MODEL: "gpt-4",
            },
        ),
    ],
)
async def test_options_switching(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    current_options: dict[str, str | bool],
    new_options: dict[str, str | bool],
    expected_options: dict[str, str | bool],
) -> None:
    """Test the options form."""
    hass.config_entries.async_update_entry(mock_config_entry, options=current_options)
    options_flow = await hass.config_entries.options.async_init(
        mock_config_entry.entry_id
    )
    if current_options.get(CONF_RECOMMENDED) != new_options.get(CONF_RECOMMENDED):
        options_flow = await hass.config_entries.options.async_configure(
            options_flow["flow_id"],
            {
                **current_options,
                CONF_RECOMMENDED: new_options[CONF_RECOMMENDED],
            },
        )
    options = await hass.config_entries.options.async_configure(
        options_flow["flow_id"],
        new_options,
    )
    await hass.async_block_till_done()
    assert options["type"] is FlowResultType.CREATE_ENTRY
    assert options["data"] == expected_options
