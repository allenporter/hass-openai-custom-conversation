"""Constants for the Custom OpenAI Conversation integration."""

import logging
from typing import Final


DOMAIN = "vicuna_conversation"
LOGGER = logging.getLogger(__package__)

DEFAULT_CONVERSATION_NAME = "Custom OpenAI Conversation"

CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_PROMPT = "prompt"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_BASE_URL = "base_url"
CONF_RECOMMENDED = "recommended"
CONF_STREAMING = "streaming"

RECOMMENDED_CHAT_MODEL = "gpt-3.5-turbo"
RECOMMENDED_CHAT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
]
RECOMMENDED_MAX_TOKENS = 3000
RECOMMENDED_TEMPERATURE = 0.7
RECOMMENDED_TOP_P = 1.0

DEFAULT_AI_TASK_NAME: Final = "Custom OpenAI AI Task"
RECOMMENDED_AI_TASK_OPTIONS = {
    CONF_RECOMMENDED: True,
}
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY = "sk-0000000000000000000"
