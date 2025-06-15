"""Tests for the vicuna_conversation component."""

from typing import Generator
from unittest.mock import AsyncMock, patch, Mock

from freezegun import freeze_time
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
import pytest
from syrupy.assertion import SnapshotAssertion
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

from homeassistant.const import CONF_LLM_HASS_API
from custom_components.vicuna_conversation.const import CONF_STREAMING
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent
from homeassistant.components import conversation
from homeassistant.setup import async_setup_component

from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
)
from .conftest import MockChatLog


@pytest.fixture(autouse=True)
def freeze_the_time():
    """Freeze the time."""
    with freeze_time("2024-05-24 12:00:00", tz_offset=0):
        yield


@pytest.fixture(autouse=True)
def mock_ulid() -> Generator[Mock]:
    """Mock the ulid library."""
    with patch("homeassistant.helpers.llm.ulid_now") as mock_ulid_now:
        mock_ulid_now.return_value = "mock-ulid"
        yield mock_ulid_now


@pytest.fixture(autouse=True)
async def mock_setup_integration(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> None:
    """Setup the integration"""
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    await hass.async_block_till_done()


async def test_conversation_entity(
    hass: HomeAssistant,
    mock_chat_log: MockChatLog,
    mock_config_entry: MockConfigEntry,
    snapshot: SnapshotAssertion,
) -> None:
    """Verify the conversation entity is loaded."""

    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=ChatCompletion(
            id="chatcmpl-1234567890ABCDEFGHIJKLMNOPQRS",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="Hello, how can I help you?",
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                    ),
                )
            ],
            created=1700000000,
            model="gpt-3.5-turbo-0613",
            object="chat.completion",
            system_fingerprint=None,
            usage=CompletionUsage(
                completion_tokens=9, prompt_tokens=8, total_tokens=17
            ),
        ),
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            mock_chat_log.conversation_id,
            Context(),
            agent_id="conversation.mock_title",
        )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    # Don't test the prompt, as it's not deterministic
    assert mock_chat_log.content[1:] == snapshot


async def test_function_call(
    hass: HomeAssistant,
    mock_chat_log: MockChatLog,  # noqa: F811
    mock_config_entry_with_assist: MockConfigEntry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test function call from the assistant."""

    mock_chat_log.mock_tool_results(
        {
            "call_call_1": "value1",
            "call_call_2": "value2",
        }
    )

    def completion_result(*args, messages, **kwargs):
        for message in messages:
            role = message["role"] if isinstance(message, dict) else message.role
            if role == "tool":
                return ChatCompletion(
                    id="chatcmpl-1234567890ZYXWVUTSRQPONMLKJIH",
                    choices=[
                        Choice(
                            finish_reason="stop",
                            index=0,
                            message=ChatCompletionMessage(
                                content="I have successfully called the function",
                                role="assistant",
                                function_call=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                    created=1700000000,
                    model="gpt-4-1106-preview",
                    object="chat.completion",
                    system_fingerprint=None,
                    usage=CompletionUsage(
                        completion_tokens=9, prompt_tokens=8, total_tokens=17
                    ),
                )

        return ChatCompletion(
            id="chatcmpl-1234567890ABCDEFGHIJKLMNOPQRS",
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    index=0,
                    message=ChatCompletionMessage(
                        content=None,
                        role="assistant",
                        function_call=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_call_1",
                                function=Function(
                                    arguments='{"param1":"call1"}',
                                    name="test_tool",
                                ),
                                type="function",
                            )
                        ],
                    ),
                )
            ],
            created=1700000000,
            model="gpt-4-1106-preview",
            object="chat.completion",
            system_fingerprint=None,
            usage=CompletionUsage(
                completion_tokens=9, prompt_tokens=8, total_tokens=17
            ),
        )

    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
        side_effect=completion_result,
    ):
        result = await conversation.async_converse(
            hass,
            "Please call the test function",
            mock_chat_log.conversation_id,
            Context(),
            agent_id="conversation.mock_title",
        )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    # Don't test the prompt, as it's not deterministic
    assert mock_chat_log.content[1:] == snapshot


@pytest.mark.parametrize(
    "tool_arguments",
    [
        (""),
        ('{"para'),
    ],
)
async def test_function_exception(
    hass: HomeAssistant,
    mock_chat_log: MockChatLog,  # noqa: F811
    mock_config_entry_with_assist: MockConfigEntry,
    tool_arguments: str,
    snapshot: SnapshotAssertion,
) -> None:
    """Test function call with exception."""

    def completion_result(*args, messages, **kwargs):
        for message in messages:
            role = message["role"] if isinstance(message, dict) else message.role
            if role == "tool":
                return ChatCompletion(
                    id="chatcmpl-1234567890ZYXWVUTSRQPONMLKJIH",
                    choices=[
                        Choice(
                            finish_reason="stop",
                            index=0,
                            message=ChatCompletionMessage(
                                content="There was an error calling the function",
                                role="assistant",
                                function_call=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                    created=1700000000,
                    model="gpt-4-1106-preview",
                    object="chat.completion",
                    system_fingerprint=None,
                    usage=CompletionUsage(
                        completion_tokens=9, prompt_tokens=8, total_tokens=17
                    ),
                )

        return ChatCompletion(
            id="chatcmpl-1234567890ABCDEFGHIJKLMNOPQRS",
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    index=0,
                    message=ChatCompletionMessage(
                        content=None,
                        role="assistant",
                        function_call=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_AbCdEfGhIjKlMnOpQrStUvWx",
                                function=Function(
                                    arguments=tool_arguments,
                                    name="test_tool",
                                ),
                                type="function",
                            )
                        ],
                    ),
                )
            ],
            created=1700000000,
            model="gpt-4-1106-preview",
            object="chat.completion",
            system_fingerprint=None,
            usage=CompletionUsage(
                completion_tokens=9, prompt_tokens=8, total_tokens=17
            ),
        )

    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
        side_effect=completion_result,
    ):
        result = await conversation.async_converse(
            hass,
            "Please call the test function",
            "conversation-id",
            Context(),
            agent_id="conversation.mock_title",
        )

    assert result.response.response_type == intent.IntentResponseType.ERROR, result
    assert result.response.speech["plain"]["speech"] == snapshot


async def test_assist_api_tools_conversion(
    hass: HomeAssistant,
    mock_config_entry_with_assist: MockConfigEntry,
) -> None:
    """Test that we are able to convert actual tools from Assist API."""
    for component in [
        "intent",
        "todo",
        "light",
        "shopping_list",
        "humidifier",
        "climate",
        "media_player",
        "vacuum",
        "cover",
        "weather",
    ]:
        assert await async_setup_component(hass, component, {})

    agent_id = mock_config_entry_with_assist.entry_id
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=ChatCompletion(
            id="chatcmpl-1234567890ABCDEFGHIJKLMNOPQRS",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="Hello, how can I help you?",
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                    ),
                )
            ],
            created=1700000000,
            model="gpt-3.5-turbo-0613",
            object="chat.completion",
            system_fingerprint=None,
            usage=CompletionUsage(
                completion_tokens=9, prompt_tokens=8, total_tokens=17
            ),
        ),
    ) as mock_create:
        await conversation.async_converse(hass, "hello", None, None, agent_id=agent_id)

    tools = mock_create.mock_calls[0][2]["tools"]
    assert tools


async def test_streaming_response(
    hass: HomeAssistant,
    mock_chat_log: MockChatLog,
    mock_config_entry: MockConfigEntry,
) -> None:
    """Test streaming response from the assistant."""
    # Enable streaming in config
    hass.config_entries.async_update_entry(
        mock_config_entry,
        options={
            **mock_config_entry.options,
            CONF_STREAMING: True,
        },
    )
    await hass.async_block_till_done()

    # Create mock streaming response
    async def mock_stream():
        # Use model_construct to bypass validation for non-final chunks
        yield ChatCompletionChunk.model_construct(
            id="chatcmpl-1234567890ABCDEFGHIJKLMNOPQRS",
            choices=[
                ChunkChoice.model_construct(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
            created=1700000000,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        )
        yield ChatCompletionChunk.model_construct(
            id="chatcmpl-1234567890ABCDEFGHIJKLMNOPQRS",
            choices=[
                ChunkChoice.model_construct(
                    index=0,
                    delta=ChoiceDelta(content=" world"),
                    finish_reason=None,
                )
            ],
            created=1700000000,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        )
        # Final chunk uses regular constructor
        yield ChatCompletionChunk(
            id="chatcmpl-1234567890ABCDEFGHIJKLMNOPQRS",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )
            ],
            created=1700000000,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        )

    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
        return_value=mock_stream(),
    ):
        result = await conversation.async_converse(
            hass,
            "hello",
            mock_chat_log.conversation_id,
            Context(),
            agent_id="conversation.mock_title",
        )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Hello world"

    # Direct assertions instead of snapshot
    content = mock_chat_log.content[1:]
    assert len(content) == 2
    assert content[0].role == "user"
    assert content[0].content == "hello"
    assert content[1].role == "assistant"
    assert content[1].content == "Hello world"


async def test_unknown_hass_api(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    snapshot: SnapshotAssertion,
) -> None:
    """Test when we reference an API that no longer exists."""
    hass.config_entries.async_update_entry(
        mock_config_entry,
        options={
            **mock_config_entry.options,
            CONF_LLM_HASS_API: "non-existing",
        },
    )
    await hass.async_block_till_done()  # Integration may reload

    result = await conversation.async_converse(
        hass, "hello", "conversation-id", Context(), agent_id=mock_config_entry.entry_id
    )

    assert result.as_dict() == snapshot
