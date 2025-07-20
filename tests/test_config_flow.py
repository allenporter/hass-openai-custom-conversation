"""Tests for the config flow."""

from typing import Generator, Any
from unittest.mock import patch, Mock

import pytest
import openai

from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API, CONF_API_KEY


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
    DEFAULT_CONVERSATION_NAME,
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
            CONF_API_KEY: "sk-0000000000000000000",
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
        CONF_API_KEY: "sk-0000000000000000000",
        "base_url": "http://llama-cublas.llama:8000/v1",
    }
    assert result["options"] == {}
    assert result["subentries"] == [
        {
            "subentry_type": "conversation",
            "data": {
                **RECOMMENDED_OPTIONS,
                CONF_CHAT_MODEL: "gpt-4",
                CONF_STREAMING: True,
            },
            "title": DEFAULT_CONVERSATION_NAME,
            "unique_id": None,
        },
        {
            "subentry_type": "ai_task_data",
            "data": {
                CONF_RECOMMENDED: True,
                CONF_CHAT_MODEL: "gpt-4",
                CONF_STREAMING: True,
            },
            "title": "Custom OpenAI AI Task",
            "unique_id": None,
        },
    ]

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
            CONF_API_KEY: "sk-0000000000000000000",
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
            CONF_API_KEY: "sk-0000000000000000000",
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
        CONF_API_KEY: "sk-0000000000000000000",
        "base_url": "http://llama-cublas.llama:8000/v1",
    }
    assert result["subentries"] == [
        {
            "subentry_type": "conversation",
            "data": {
                **RECOMMENDED_OPTIONS,
                CONF_CHAT_MODEL: "gpt-4",
                CONF_STREAMING: False,
            },
            "title": DEFAULT_CONVERSATION_NAME,
            "unique_id": None,
        },
        {
            "subentry_type": "ai_task_data",
            "data": {
                CONF_RECOMMENDED: True,
                CONF_CHAT_MODEL: "gpt-4",
                CONF_STREAMING: False,
            },
            "title": "Custom OpenAI AI Task",
            "unique_id": None,
        },
    ]

    assert len(mock_setup.mock_calls) == 1


async def test_creating_conversation_subentry(
    hass: HomeAssistant,
    setup_integration: None,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test creating a conversation subentry."""
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, "conversation"),
        context={"source": config_entries.SOURCE_USER},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert not result["errors"]

    result2 = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {"name": "My Custom Agent", **RECOMMENDED_OPTIONS},
    )
    await hass.async_block_till_done()

    assert result2["type"] is FlowResultType.CREATE_ENTRY
    assert result2["title"] == "My Custom Agent"

    processed_options = RECOMMENDED_OPTIONS.copy()
    processed_options.update(
        {
            CONF_STREAMING: True,
        }
    )

    assert result2["data"] == processed_options


async def test_creating_conversation_subentry_not_loaded(
    hass: HomeAssistant,
    setup_integration: None,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test creating a conversation subentry when entry is not loaded."""
    await hass.config_entries.async_unload(mock_config_entry.entry_id)
    with patch(
        "homeassistant.components.openai_conversation.config_flow.openai.resources.models.AsyncModels.list",
        return_value=[],
    ):
        result = await hass.config_entries.subentries.async_init(
            (mock_config_entry.entry_id, "conversation"),
            context={"source": config_entries.SOURCE_USER},
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_not_loaded"


@pytest.mark.parametrize(
    "config_entry_options",
    [
        {CONF_RECOMMENDED: True},
    ],
)
async def test_subentry_recommended(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: None,
) -> None:
    """Test the subentry flow with recommended settings."""
    subentry = next(iter(mock_config_entry.subentries.values()))
    subentry_flow = await mock_config_entry.start_subentry_reconfigure_flow(
        hass, subentry.subentry_id
    )
    options = await hass.config_entries.subentries.async_configure(
        subentry_flow["flow_id"],
        {
            "prompt": "Speak like a pirate",
            CONF_RECOMMENDED: True,
            CONF_LLM_HASS_API: [],
        },
    )
    await hass.async_block_till_done()
    assert options["type"] is FlowResultType.ABORT
    assert options["reason"] == "reconfigure_successful"
    assert subentry.data["prompt"] == "Speak like a pirate"


@pytest.mark.parametrize(
    ("config_entry_options", "new_options", "expected_options"),
    [
        (  # Test converting single llm api format to list
            {
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: "assist",
                CONF_PROMPT: "",
                CONF_STREAMING: True,
            },
            (
                {
                    CONF_RECOMMENDED: True,
                    CONF_LLM_HASS_API: ["assist"],
                    CONF_PROMPT: "",
                },
            ),
            {
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: ["assist"],
                CONF_PROMPT: "",
                CONF_CHAT_MODEL: "gpt-3.5-turbo",
                CONF_STREAMING: True,
            },
        ),
        (  # options with no model-specific settings
            {
                CONF_RECOMMENDED: True,
                CONF_STREAMING: True,
            },
            (
                {
                    CONF_RECOMMENDED: False,
                    CONF_PROMPT: "Speak like a pirate",
                },
                {
                    CONF_PROMPT: "Speak like a pirate",
                    CONF_TEMPERATURE: 1.0,
                    CONF_CHAT_MODEL: "gpt-4.5-preview",
                    CONF_TOP_P: RECOMMENDED_TOP_P,
                    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
                },
            ),
            {
                CONF_RECOMMENDED: False,
                CONF_PROMPT: "Speak like a pirate",
                CONF_TEMPERATURE: 1.0,
                CONF_CHAT_MODEL: "gpt-4.5-preview",
                CONF_TOP_P: RECOMMENDED_TOP_P,
                CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
                CONF_STREAMING: True,
            },
        ),
        (
            {
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: "assist",
                CONF_PROMPT: "bla",
                CONF_CHAT_MODEL: "gpt-4o",
                CONF_STREAMING: True,
            },
            (
                {
                    CONF_RECOMMENDED: False,
                    CONF_LLM_HASS_API: ["assist"],
                    CONF_CHAT_MODEL: "gpt-4o",
                    CONF_PROMPT: "bla",
                },
                {
                    CONF_RECOMMENDED: False,
                    CONF_PROMPT: "Speak like a pirate",
                    CONF_CHAT_MODEL: "gpt-4o",
                    CONF_LLM_HASS_API: ["assist"],
                    CONF_TEMPERATURE: 0.3,
                },
            ),
            {
                CONF_RECOMMENDED: False,
                CONF_LLM_HASS_API: ["assist"],
                CONF_PROMPT: "Speak like a pirate",
                CONF_TEMPERATURE: 0.3,
                CONF_CHAT_MODEL: "gpt-4o",
                CONF_TOP_P: RECOMMENDED_TOP_P,
                CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
                CONF_STREAMING: True,
            },
        ),
        (
            {
                CONF_RECOMMENDED: False,
                CONF_PROMPT: "Speak like a pirate",
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_TEMPERATURE: 0.3,
                CONF_TOP_P: RECOMMENDED_TOP_P,
                CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
                CONF_STREAMING: True,
            },
            (
                {
                    CONF_RECOMMENDED: True,
                    CONF_PROMPT: "Speak like a pirate",
                    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                },
                {
                    CONF_RECOMMENDED: True,
                    CONF_LLM_HASS_API: ["assist"],
                    CONF_PROMPT: "",
                    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                },
            ),
            {
                CONF_RECOMMENDED: True,
                CONF_LLM_HASS_API: ["assist"],
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_PROMPT: "",
                CONF_STREAMING: True,
            },
        ),
    ],
)
async def test_subentry_switching(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    config_entry_options: dict[str, str | bool],
    new_options: dict[str, str | bool],
    expected_options: dict[str, str | bool],
    setup_integration: None,
) -> None:
    """Test the subentry form."""
    subentry = next(iter(mock_config_entry.subentries.values()))
    subentry_flow = await mock_config_entry.start_subentry_reconfigure_flow(
        hass, subentry.subentry_id
    )
    assert subentry_flow["step_id"] == "init"

    current_options = config_entry_options
    i = 0
    for step_options in new_options:
        assert (
            subentry_flow["type"] == FlowResultType.FORM
        ), f"Expected {i} form, got {subentry_flow}"
        i += 1

        # Test that current options are showed as suggested values:
        for key in subentry_flow["data_schema"].schema:
            if (
                isinstance(key.description, dict)
                and "suggested_value" in key.description
                and key in current_options
                and key != CONF_RECOMMENDED
            ):
                current_option = current_options[key]
                if key == CONF_LLM_HASS_API and isinstance(current_option, str):
                    current_option = [current_option]
                assert (
                    key.description["suggested_value"] == current_option
                ), f"Expected {key.description['suggested_value']} for {key}, got {current_option}"

        # Configure current step
        subentry_flow = await hass.config_entries.subentries.async_configure(
            subentry_flow["flow_id"],
            step_options,
        )
        await hass.async_block_till_done()

    assert subentry_flow["type"] is FlowResultType.ABORT
    assert subentry_flow["reason"] == "reconfigure_successful"
    assert subentry.data == expected_options
