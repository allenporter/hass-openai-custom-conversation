"""Test the AI Task entity."""

from unittest.mock import AsyncMock
import json

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage


from homeassistant.core import HomeAssistant
from homeassistant.const import Platform
from homeassistant.components import ai_task
import voluptuous as vol


from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
)


@pytest.fixture(name="platforms")
def mock_platforms() -> list[Platform]:
    """Set the platforms for the tests."""
    return [Platform.AI_TASK]


async def test_generate_data(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: None,
    mock_completion: AsyncMock,
) -> None:
    """Test the generate_data method."""
    mock_completion.return_value = ChatCompletion(
        id="chatcmpl-123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="mocked response",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                ),
            )
        ],
        created=1700000000,
        model="gpt-3.5-turbo",
        object="chat.completion",
        system_fingerprint=None,
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
    )
    response = await ai_task.async_generate_data(
        hass,
        task_name="Test Task",
        entity_id="ai_task.custom_openai_ai_task",
        instructions="test prompt",
    )
    assert response.data == "mocked response"
    assert len(mock_completion.call_args_list) == 1
    assert mock_completion.call_args.kwargs["messages"][1:] == [
        {"role": "user", "content": "test prompt"},
        {"role": "assistant", "content": "mocked response"},
    ]


async def test_generate_structured_data(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: None,
    mock_completion: AsyncMock,
) -> None:
    """Test the generate_data method with a structure."""
    mock_response_data = {"value": "mocked response"}
    mock_completion.return_value = ChatCompletion(
        id="chatcmpl-123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=json.dumps(mock_response_data),
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                ),
            )
        ],
        created=1700000000,
        model="gpt-3.5-turbo",
        object="chat.completion",
        system_fingerprint=None,
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
    )
    response = await ai_task.async_generate_data(
        hass,
        task_name="Test Task",
        entity_id="ai_task.custom_openai_ai_task",
        instructions="test prompt",
        structure=vol.Schema({"value": str}),
    )
    assert response.data == mock_response_data
    assert len(mock_completion.call_args_list) == 1
    assert "response_format" in mock_completion.call_args.kwargs
    assert mock_completion.call_args.kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": [],
        },
    }


async def test_generate_data_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    setup_integration: None,
    mock_completion: AsyncMock,
) -> None:
    """Test the generate_data method handles an error."""
    mock_completion.side_effect = ValueError("some error")
    with pytest.raises(ValueError, match="some error"):
        await ai_task.async_generate_data(
            hass,
            task_name="Test Task",
            entity_id="ai_task.custom_openai_ai_task",
            instructions="test prompt",
        )
