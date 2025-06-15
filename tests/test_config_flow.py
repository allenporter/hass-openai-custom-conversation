"""Tests for the config flow."""

from unittest.mock import patch

import pytest

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


async def test_config_flow(
    hass: HomeAssistant,
) -> None:
    """Test selecting a model in the configuration flow."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result.get("type") is FlowResultType.FORM
    assert result.get("errors") is None

    with (
        patch(
            f"custom_components.{DOMAIN}.config_flow.get_streaming_support",
            return_value=False,
        ),
        patch(
            f"custom_components.{DOMAIN}.async_setup_entry", return_value=True
        ) as mock_setup,
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "api_key": "sk-0000000000000000000",
                "base_url": "http://llama-cublas.llama:8000/v1",
                "chat_model": RECOMMENDED_CHAT_MODEL,
            },
        )
        await hass.async_block_till_done()

    assert result.get("type") is FlowResultType.CREATE_ENTRY
    assert result.get("title") == "Custom OpenAI"
    assert result.get("data") == {
        "api_key": "sk-0000000000000000000",
        "base_url": "http://llama-cublas.llama:8000/v1",
        "chat_model": RECOMMENDED_CHAT_MODEL,
    }
    expected_options = RECOMMENDED_OPTIONS.copy()
    expected_options[CONF_CHAT_MODEL] = RECOMMENDED_CHAT_MODEL
    expected_options[CONF_STREAMING] = False
    assert result["options"] == expected_options
    assert len(mock_setup.mock_calls) == 1


async def test_options(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test the options form."""
    with patch(
        "custom_components.vicuna_conversation.config_flow.get_streaming_support",
        return_value=False,
    ):
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
                CONF_STREAMING: False,
            },
        ),
        (
            {
                CONF_RECOMMENDED: False,
                CONF_PROMPT: "Speak like a pirate",
                CONF_TEMPERATURE: 0.3,
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
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
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_STREAMING: False,
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
    with patch(
        "custom_components.vicuna_conversation.config_flow.get_streaming_support",
        return_value=False,
    ):
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
