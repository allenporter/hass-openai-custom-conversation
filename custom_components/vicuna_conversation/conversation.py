"""The OpenAI Conversation integration."""

from collections.abc import Callable, AsyncGenerator
import json
import logging
from typing import Literal, Any, cast

import openai
from openai._streaming import AsyncStream
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition
from voluptuous_openapi import convert

from homeassistant.components import assist_pipeline, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_STREAMING,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    LOGGER,
)


_LOGGER = logging.getLogger(__name__)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = OpenAIConversationEntity(config_entry)
    async_add_entities([agent])


def _format_tool(
    tool: llm.Tool,
    custom_serializer: Callable[[Any], Any] | None,
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


def _convert_content_to_chat_message(
    content: conversation.Content,
) -> ChatCompletionMessageParam | None:
    """Convert any native chat message for this agent to the native format."""
    _LOGGER.debug("_convert_content_to_chat_message=%s", content)
    if isinstance(content, conversation.ToolResultContent):
        return ChatCompletionToolMessageParam(
            # Note: The functionary 'tool' role expects a name which is
            # not supported in llama cpp python and the openai protos.
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json.dumps(content.tool_result),
        )

    role: Literal["user", "assistant", "system"] = content.role
    if role == "system" and content.content:
        return ChatCompletionSystemMessageParam(role="system", content=content.content)

    if role == "user" and content.content:
        return ChatCompletionUserMessageParam(role="user", content=content.content)

    if role == "assistant":
        param = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=content.content,
        )
        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            param["tool_calls"] = [
                ChatCompletionMessageToolCallParam(
                    type="function",
                    id=tool_call.id,
                    function=Function(
                        arguments=json.dumps(tool_call.tool_args),
                        name=tool_call.tool_name,
                    ),
                )
                for tool_call in content.tool_calls
            ]
        return param
    LOGGER.warning("Could not convert message to OpenAI API: %s", content)
    return None


def _decode_tool_arguments(arguments: str) -> Any:
    """Decode tool call arguments."""
    try:
        return json.loads(arguments)
    except json.JSONDecodeError as err:
        raise HomeAssistantError(f"Unexpected tool argument response: {err}") from err


async def _transform_response(
    message: ChatCompletionMessage,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform the OpenAI API message to a ChatLog format."""
    data: conversation.AssistantContentDeltaDict = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_calls:
        data["tool_calls"] = [
            llm.ToolInput(
                id=tool_call.id,
                tool_name=tool_call.function.name,
                tool_args=_decode_tool_arguments(tool_call.function.arguments),
            )
            for tool_call in message.tool_calls
        ]
    yield data


def _convert_content_to_param(
    content: conversation.Content,
) -> ChatCompletionMessageParam:
    """Convert any native chat message for this agent to the native format."""
    if content.role == "tool_result":
        assert type(content) is conversation.ToolResultContent
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json.dumps(content.tool_result),
        )
    if content.role != "assistant" or not content.tool_calls:
        role: Literal["system", "user", "assistant", "developer"] = content.role
        if role == "system":
            role = "developer"
        return cast(
            ChatCompletionMessageParam,
            {"role": content.role, "content": content.content},
        )

    # Handle the Assistant content including tool calls.
    assert type(content) is conversation.AssistantContent
    return ChatCompletionAssistantMessageParam(
        role="assistant",
        content=content.content,
        tool_calls=[
            ChatCompletionMessageToolCallParam(
                id=tool_call.id,
                function=Function(
                    arguments=json.dumps(tool_call.tool_args),
                    name=tool_call.tool_name,
                ),
                type="function",
            )
            for tool_call in content.tool_calls
        ],
    )


async def _transform_stream(
    result: AsyncStream[ChatCompletionChunk],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform an OpenAI delta stream into HA format."""
    current_tool_call: dict | None = None

    async for chunk in result:
        LOGGER.debug("Received chunk: %s", chunk)
        choice = chunk.choices[0]

        if choice.finish_reason:
            if current_tool_call:
                tool_args = []
                if current_tool_call["tool_args"]:
                    tool_args = json.loads(current_tool_call["tool_args"])
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=current_tool_call["id"],
                            tool_name=current_tool_call["tool_name"],
                            tool_args=tool_args,
                        )
                    ]
                }

            break

        delta = chunk.choices[0].delta

        # We can yield delta messages not continuing or starting tool calls
        if current_tool_call is None and not delta.tool_calls:
            yield {  # type: ignore[misc]
                key: value
                for key in ("role", "content")
                if (value := getattr(delta, key)) is not None
            }
            continue

        # When doing tool calls, we should always have a tool call
        # object or we have gotten stopped above with a finish_reason set.
        if (
            not delta.tool_calls
            or not (delta_tool_call := delta.tool_calls[0])
            or not delta_tool_call.function
        ):
            continue

        if current_tool_call and delta_tool_call.index == current_tool_call["index"]:
            current_tool_call["tool_args"] += delta_tool_call.function.arguments or ""
            continue

        # We got tool call with new index, so we need to yield the previous
        if current_tool_call:
            yield {
                "tool_calls": [
                    llm.ToolInput(
                        id=current_tool_call["id"],
                        tool_name=current_tool_call["tool_name"],
                        tool_args=json.loads(current_tool_call["tool_args"]),
                    )
                ]
            }

        current_tool_call = {
            "index": delta_tool_call.index,
            "id": delta_tool_call.id,
            "tool_name": delta_tool_call.function.name,
            "tool_args": delta_tool_call.function.arguments or "",
        }


class OpenAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """OpenAI conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="OpenAI",
            model="Custom OpenAI",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def attribution(self) -> str | None:
        """Return the attribution."""
        return "Powered by Custom LLM"

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        options = self.entry.options

        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[ChatCompletionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        messages = [
            m
            for content in chat_log.content
            if (m := _convert_content_to_chat_message(content))
        ]

        client = self.entry.runtime_data

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                result = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools or NOT_GIVEN,
                    max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                    temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                    user=chat_log.conversation_id,
                    stream=options.get(CONF_STREAMING),
                )
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to API: %s", err)
                raise HomeAssistantError("Error talking to API") from err

            if options.get(CONF_STREAMING):
                convert_message = _convert_content_to_param
                convert_stream = _transform_stream
            else:
                convert_message = _convert_content_to_chat_message
                convert_stream = _transform_response
                result = result.choices[0].message

            messages.extend(
                [
                    msg
                    async for content in chat_log.async_add_delta_content_stream(
                        user_input.agent_id, convert_stream(result)
                    )
                    if (msg := convert_message(content))
                ]
            )

            if not chat_log.unresponded_tool_results:
                break

        intent_response = intent.IntentResponse(language=user_input.language)
        assert type(chat_log.content[-1]) is conversation.AssistantContent
        intent_response.async_set_speech(chat_log.content[-1].content or "")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)
