"""Conversation support for Custom OpenAI."""

from typing import Literal

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

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import MATCH_ALL, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_PROMPT, DOMAIN
from .entity import CustomOpenAIBaseLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue
        async_add_entities(
            [OpenAIConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,  # type: ignore[call-arg]
        )


class OpenAIConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    CustomOpenAIBaseLLMEntity,
):
    """OpenAI conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
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
        conversation.async_set_agent(self.hass, self.entry, self)

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
        options = self.subentry.data

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
