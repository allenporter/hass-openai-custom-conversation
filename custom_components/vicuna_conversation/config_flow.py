"""Config flow for OpenAI Conversation integration."""

from __future__ import annotations

import logging
from typing import Any, cast

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigSubentryFlow,
    SubentryFlowResult,
    ConfigFlow,
    ConfigFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    TemplateSelector,
    SelectSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
)
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_BASE_URL,
    CONF_STREAMING,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_AI_TASK_NAME,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CHAT_MODELS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    LOGGER,
)
from .openai_client import (
    async_create_client,
    async_list_models,
    async_validate_completions,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY, default=DEFAULT_API_KEY): str,
        vol.Required(CONF_BASE_URL, default=DEFAULT_BASE_URL): str,
    }
)
STEP_MODELS_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_CHAT_MODEL, default=RECOMMENDED_CHAT_MODEL): str,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_CHAT_MODEL: "gpt-3.5-turbo",
}

RECOMMENDED_OPTIONS_AI_TASK = {
    CONF_RECOMMENDED: True,
    CONF_CHAT_MODEL: "gpt-3.5-turbo",
}


def _recommended_model(models: list[str] | None) -> str:
    """Return the selected model from user input."""
    # Don't use the recommended model if there is a valid list of other
    # models to choose from.
    if not models:
        return RECOMMENDED_CHAT_MODEL
    for model in RECOMMENDED_CHAT_MODELS:
        if model in models:
            return model
    return models[0]


class OpenAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation."""

    VERSION = 2
    MINOR_VERSION = 2

    data: dict[str, Any] | None = None
    client: openai.AsyncOpenAI | None = None
    models: list[str] | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors = {}
        if user_input is not None:
            self._async_abort_entries_match(user_input)
            try:
                self.client = await async_create_client(self.hass, user_input)
                self.models = await async_list_models(self.client)
            except openai.APIConnectionError:
                errors["base"] = "cannot_connect"
            except openai.AuthenticationError:
                errors["base"] = "invalid_api_key"
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                self.data = user_input
                return await self.async_step_model()

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle selecting a model."""
        assert self.client is not None
        assert self.models is not None
        errors = {}
        if user_input is not None:
            model = user_input[CONF_CHAT_MODEL]
            success = await async_validate_completions(
                self.client,
                model=model,
                stream=False,
            )
            if not success:
                errors["base"] = "cannot_connect"
            else:
                stream_support = await async_validate_completions(
                    self.client,
                    model=model,
                    stream=True,
                )
                base_options = {
                    **user_input,
                    CONF_STREAMING: stream_support,
                }
                return self.async_create_entry(
                    title="Custom OpenAI",
                    data=self.data,  # type: ignore[arg-type]
                    subentries=[
                        {
                            "subentry_type": "conversation",
                            "data": {
                                **RECOMMENDED_OPTIONS,
                                **base_options,
                            },
                            "title": DEFAULT_CONVERSATION_NAME,
                            "unique_id": None,
                        },
                        {
                            "subentry_type": "ai_task_data",
                            "data": {
                                **RECOMMENDED_OPTIONS_AI_TASK,
                                **base_options,
                            },
                            "title": DEFAULT_AI_TASK_NAME,
                            "unique_id": None,
                        },
                    ],
                )

        return self.async_show_form(
            step_id="model",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_CHAT_MODEL,
                        ): SelectSelector(
                            SelectSelectorConfig(
                                options=self.models,
                                translation_key=CONF_CHAT_MODEL,
                                mode=SelectSelectorMode.DROPDOWN,
                                custom_value=True,  # Allow manual model entry
                            ),
                        ),
                    }
                ),
                {
                    CONF_CHAT_MODEL: (user_input or {}).get(
                        CONF_CHAT_MODEL, _recommended_model(self.models)
                    ),
                },
            ),
            errors=errors,
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": ConversationSubentryFlowHandler,
            "ai_task": ConversationSubentryFlowHandler,
        }


class ConversationSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing conversation subentries."""

    last_rendered_recommended = False
    options: dict[str, Any]
    models: list[str] | None = None

    @property
    def _openai_client(self) -> openai.AsyncOpenAI:
        """Return the OpenAI client."""
        return cast(openai.AsyncOpenAI, self._get_entry().runtime_data)

    async def _get_models(self) -> list[str] | None:
        """Return the list of models."""
        if self.models is None:
            self.models = await async_list_models(self._openai_client)
        return self.models

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        if self._subentry_type == "ai_task":
            self.options = RECOMMENDED_OPTIONS_AI_TASK.copy()
        else:
            self.options = RECOMMENDED_OPTIONS.copy()
        self.last_rendered_recommended = cast(
            bool, self.options.get(CONF_RECOMMENDED, False)
        )
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of a subentry."""
        LOGGER.debug("Reconfiguring subentry: %s", self._get_entry().title)
        self.options = self._get_reconfigure_subentry().data.copy()
        self.last_rendered_recommended = self.options.get(CONF_RECOMMENDED, False)
        LOGGER.debug("Last rendered recommended: %s", self.last_rendered_recommended)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage initial options."""
        # abort if entry is not loaded
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        options = self.options

        if user_input is not None:
            model = user_input[CONF_CHAT_MODEL]
            stream_support = await async_validate_completions(
                self._openai_client,
                model=model,
                stream=True,
            )
            user_input[CONF_STREAMING] = stream_support
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if self._is_new:
                    return self.async_create_entry(
                        title=user_input.pop(CONF_NAME),
                        data=user_input,
                    )

                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=user_input,
                )

            # Re-render the options again, now with the recommended options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input[CONF_PROMPT],
                CONF_CHAT_MODEL: user_input[CONF_CHAT_MODEL],
            }
            if self._subentry_type == "conversation":
                options[CONF_LLM_HASS_API] = user_input.get(CONF_LLM_HASS_API, [])

        models = await self._get_models()
        schema = openai_config_option_schema(
            self.hass, self._subentry_type, self._is_new, options, models
        )
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), options
            ),
        )


def openai_config_option_schema(
    hass: HomeAssistant,
    subentry_type: str,
    is_new: bool,
    options: dict[str, Any],
    models: list[str] | None = None,
) -> dict:
    """Return a schema for OpenAI completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    ]
    LOGGER.debug("Available LLM APIs: %s", hass_apis)
    if (suggested_llm_apis := options.get(CONF_LLM_HASS_API)) and isinstance(
        suggested_llm_apis, str
    ):
        options[CONF_LLM_HASS_API] = [suggested_llm_apis]

    if is_new:
        schema: dict[vol.Required | vol.Optional, Any] = {
            vol.Required(
                CONF_NAME,
                default=(
                    DEFAULT_AI_TASK_NAME
                    if subentry_type == "ai_task"
                    else DEFAULT_CONVERSATION_NAME
                ),
            ): str,
        }
    else:
        schema = {}

    schema.update(
        {
            vol.Optional(
                CONF_PROMPT,
                description={
                    "suggested_value": options.get(
                        CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                    )
                },
            ): TemplateSelector(),
        }
    )
    if subentry_type == "conversation":
        schema.update(
            {
                vol.Optional(
                    CONF_LLM_HASS_API,
                    # description={"suggested_value": suggested_llm_apis},
                    # default="none",
                ): SelectSelector(
                    SelectSelectorConfig(options=hass_apis, multiple=True)
                ),
            }
        )
    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=options.get(CONF_CHAT_MODEL, _recommended_model(models)),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=models or [],
                    translation_key=CONF_CHAT_MODEL,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,  # Allow manual model entry
                ),
            ),
            vol.Required(
                CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
            ): bool,
        }
    )

    if options.get(CONF_RECOMMENDED):
        return schema

    schema.update(
        {
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
        }
    )
    return schema
