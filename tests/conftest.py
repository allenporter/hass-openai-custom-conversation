"""Fixtures for the custom component."""

from collections.abc import Generator
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
import pathlib
import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from syrupy import SnapshotAssertion
from syrupy.extensions.amber import AmberSnapshotExtension
from syrupy.location import PyTestLocation

from homeassistant.const import CONF_LLM_HASS_API, Platform
from homeassistant.helpers import llm
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from homeassistant.components import conversation
from homeassistant.helpers import chat_session

from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
)

from custom_components.vicuna_conversation.const import (
    DOMAIN,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_AI_TASK_NAME,
)


_LOGGER = logging.getLogger(__name__)


DIFFERENT_DIRECTORY = "snapshots"
CONFIG_ENTRY_DATA = {
    "api_key": "sk-0000000000000000000",
    "base_url": "http://llama-cublas.llama:8000/v1",
}
ASSIST_OPTIONS = {CONF_LLM_HASS_API: llm.LLM_API_ASSIST}


class DifferentDirectoryExtension(AmberSnapshotExtension):
    """Extension to set a different snapshot directory."""

    @classmethod
    def dirname(cls, *, test_location: "PyTestLocation") -> str:
        """Override the snapshot directory name."""
        return str(
            pathlib.Path(test_location.filepath).parent.joinpath(DIFFERENT_DIRECTORY)
        )


@pytest.fixture
def snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Fixture to override the snapshot directory."""
    return snapshot.use_extension(DifferentDirectoryExtension)


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> Generator[None, None, None]:
    """Enable custom integration."""
    _ = enable_custom_integrations  # unused
    yield


@pytest.fixture(autouse=True)
async def setup_home_assistant(hass: HomeAssistant) -> None:
    """Enable dependencies."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture(name="platforms")
def mock_platforms() -> list[Platform]:
    """Fixture for platforms loaded by the integration."""
    return []


@pytest.fixture(name="setup_integration")
async def mock_setup_integration(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    platforms: list[Platform],
) -> None:
    """Set up the integration."""

    with patch(f"custom_components.{DOMAIN}.PLATFORMS", platforms):
        assert await async_setup_component(hass, DOMAIN, {})
        await hass.async_block_till_done()
        yield


@pytest.fixture(name="config_entry_data")
async def config_entry_data_fixture() -> dict[str, Any]:
    """Fixture to add data to the config entry."""
    return {}


@pytest.fixture(name="config_entry_options")
async def config_entry_options_fixture() -> dict[str, Any]:
    """Fixture to add options to the config entry."""
    return {}


@pytest.fixture(name="mock_config_entry")
async def mock_config_entry_fixture(
    hass: HomeAssistant,
    config_entry_data: dict[str, Any],
    config_entry_options: dict[str, Any],
) -> MockConfigEntry:
    """Fixture to create a configuration entry."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title="OpenAI Custom Conversation",
        data={
            **CONFIG_ENTRY_DATA,
            **config_entry_data,
        },
        version=2,
        minor_version=2,
        subentries_data=[
            {
                "data": {**config_entry_options},
                "subentry_type": "conversation",
                "title": DEFAULT_CONVERSATION_NAME,
                "unique_id": None,
            },
            {
                "data": {**config_entry_options},
                "subentry_type": "ai_task_data",
                "title": DEFAULT_AI_TASK_NAME,
                "unique_id": None,
            },
        ],
    )
    config_entry.add_to_hass(hass)
    return config_entry


@dataclass
class MockChatLog(conversation.ChatLog):
    """Mock chat log."""

    _mock_tool_results: dict = field(default_factory=dict)

    def mock_tool_results(self, results: dict) -> None:
        """Set tool results."""
        self._mock_tool_results = results

    @property
    def llm_api(self):
        """Return LLM API."""
        return self._llm_api

    @llm_api.setter
    def llm_api(self, value):
        """Set LLM API."""
        self._llm_api = value

        if not value:
            return

        async def async_call_tool(tool_input):
            """Call tool."""
            if tool_input.id not in self._mock_tool_results:
                raise ValueError(
                    f"Tool {tool_input.id} not found ({self._mock_tool_results})"
                )
            return self._mock_tool_results[tool_input.id]

        self._llm_api.async_call_tool = async_call_tool


@pytest.fixture
async def mock_chat_log(hass: HomeAssistant) -> AsyncGenerator[MockChatLog]:
    """Return mock chat logs."""
    # pylint: disable-next=contextmanager-generator-missing-cleanup
    with (
        patch(
            "homeassistant.components.conversation.chat_log.ChatLog",
            MockChatLog,
        ),
        chat_session.async_get_chat_session(hass, "mock-conversation-id") as session,
        conversation.async_get_chat_log(hass, session) as chat_log,
    ):
        yield chat_log


@pytest.fixture(autouse=True)
async def mock_models_list() -> None:
    """Initialize integration."""
    with patch(
        "openai.resources.models.AsyncModels.list",
    ):
        yield


@pytest.fixture(name="mock_completion", autouse=True)
async def mock_openai_client_fixture() -> AsyncMock:
    """Fixture to mock the OpenAI client."""
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
    ) as mock_create:
        yield mock_create
