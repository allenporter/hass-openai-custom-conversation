"""Config flow for OpenAI Conversation integration."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    TemplateSelector,
)
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
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
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    LOGGER,
)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY, default=DEFAULT_API_KEY): str,
        vol.Required(CONF_BASE_URL, default=DEFAULT_BASE_URL): str,
        vol.Required(CONF_CHAT_MODEL, default=RECOMMENDED_CHAT_MODEL): str,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """

    def list_models() -> None:
        """Get OpenAI client."""
        client = openai.AsyncOpenAI(
            api_key=data[CONF_API_KEY], base_url=data[CONF_BASE_URL]
        )
        client = client.with_options(timeout=10.0)
        client.models.list()  # Ignore
        return

    await hass.async_add_executor_job(list_models)


def get_streaming_support(base_url: str, api_key: str, model_name: str) -> bool:
    """Validate streaming and tool calling support on the remote API.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    client = client.with_options(timeout=10.0)
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "description": "Test function.",
                    },
                }
            ],
            temperature=0,
            max_tokens=3,  # we don't care about the response
            stream=True,
        )
        for event in stream:
            if event.choices[0].finish_reason is not None:
                continue
    except openai.OpenAIError:
        # If the model doesn't support streaming, we can just fall back to non-streaming
        return False
    # If we get here, we got a full response and streaming is supported
    return True


class OpenAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}

        try:
            await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            config_options = RECOMMENDED_OPTIONS.copy()
            config_options[CONF_CHAT_MODEL] = user_input[CONF_CHAT_MODEL]
            config_options[CONF_STREAMING] = get_streaming_support(
                user_input[CONF_BASE_URL],
                user_input[CONF_API_KEY],
                user_input[CONF_CHAT_MODEL],
            )
            return self.async_create_entry(
                title="Custom OpenAI",
                data=user_input,
                options=config_options,
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return OpenAIOptionsFlow(config_entry)


class OpenAIOptionsFlow(OptionsFlow):
    """OpenAI config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if user_input[CONF_LLM_HASS_API] == "none":
                    user_input.pop(CONF_LLM_HASS_API)
                base_url = self.config_entry.data[CONF_BASE_URL]
                api_key = self.config_entry.data[CONF_API_KEY]
                user_input[CONF_STREAMING] = get_streaming_support(
                    base_url, api_key, user_input[CONF_CHAT_MODEL]
                )
                return self.async_create_entry(title="", data=user_input)

            # Re-render the options again, now with the recommended options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input[CONF_PROMPT],
                CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
            }

        schema = openai_config_option_schema(self.hass, options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def openai_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
) -> dict:
    """Return a schema for OpenAI completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    schema = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Optional(
            CONF_CHAT_MODEL,
            description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            default=RECOMMENDED_CHAT_MODEL,
        ): str,
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

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
