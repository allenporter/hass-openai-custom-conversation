"""The OpenAI Conversation integration."""

from __future__ import annotations

import logging
from types import MappingProxyType

import openai

from homeassistant.config_entries import ConfigEntry, ConfigSubentry, ConfigType
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.const import Platform, CONF_API_KEY
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    config_validation as cv,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_STREAMING,
    DEFAULT_AI_TASK_NAME,
    DOMAIN,
    RECOMMENDED_AI_TASK_OPTIONS,
)
from .openai_client import async_create_client, async_list_models

__all__ = [
    DOMAIN,
]

_LOGGER = logging.getLogger(__name__)
PLATFORMS = (Platform.CONVERSATION, Platform.AI_TASK)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Conversation."""
    await async_migrate_integration(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""

    try:
        client = await async_create_client(hass, entry.data)
        await async_list_models(client)
    except openai.AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_migrate_integration(hass: HomeAssistant) -> None:
    """Migrate integration entry structure."""

    entries = hass.config_entries.async_entries(DOMAIN)
    if not any(entry.version == 1 for entry in entries):
        return

    api_keys_entries: dict[str, ConfigEntry] = {}
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)

    for entry in entries:
        use_existing = False
        subentry = ConfigSubentry(
            data=entry.options,
            subentry_type="conversation",
            title=entry.title,
            unique_id=None,
        )
        if entry.data[CONF_API_KEY] not in api_keys_entries:
            use_existing = True
            api_keys_entries[entry.data[CONF_API_KEY]] = entry

        parent_entry = api_keys_entries[entry.data[CONF_API_KEY]]

        hass.config_entries.async_add_subentry(parent_entry, subentry)
        conversation_entity = entity_registry.async_get_entity_id(
            "conversation",
            DOMAIN,
            entry.entry_id,
        )
        if conversation_entity is not None:
            entity_registry.async_update_entity(
                conversation_entity,
                config_entry_id=parent_entry.entry_id,
                config_subentry_id=subentry.subentry_id,
                new_unique_id=subentry.subentry_id,
            )

        device = device_registry.async_get_device(
            identifiers={(DOMAIN, entry.entry_id)}
        )
        if device is not None:
            device_registry.async_update_device(
                device.id,
                new_identifiers={(DOMAIN, subentry.subentry_id)},
                add_config_subentry_id=subentry.subentry_id,
                add_config_entry_id=parent_entry.entry_id,
            )
            if parent_entry.entry_id != entry.entry_id:
                device_registry.async_update_device(
                    device.id,
                    remove_config_entry_id=entry.entry_id,
                )
            else:
                device_registry.async_update_device(
                    device.id,
                    remove_config_entry_id=entry.entry_id,
                    remove_config_subentry_id=None,
                )

        if not use_existing:
            await hass.config_entries.async_remove(entry.entry_id)
        else:
            hass.config_entries.async_update_entry(
                entry,
                options={},
                version=2,
            )


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    if entry.version > 2:
        # New version, don't know how to migrate
        _LOGGER.debug("Cannot migrate from version %s", entry.version)
        return False

    if entry.version == 2 and entry.minor_version >= 2:
        _LOGGER.debug(
            "No migration needed for version %s.%s", entry.version, entry.minor_version
        )
        return True

    _LOGGER.debug("Migrating from version %s.%s", entry.version, entry.minor_version)

    existing_subentry = next(iter(entry.subentries.values()))

    # The streaming support is not known, but it is not used by the AI task entity.
    # The config flow will update it when the user reconfigures the entity.
    hass.config_entries.async_add_subentry(
        entry,
        ConfigSubentry(
            data=MappingProxyType(
                {
                    **RECOMMENDED_AI_TASK_OPTIONS,
                    CONF_CHAT_MODEL: existing_subentry.data[CONF_CHAT_MODEL],
                    CONF_STREAMING: existing_subentry.data.get(CONF_STREAMING),
                }
            ),
            subentry_type="ai_task_data",
            title=DEFAULT_AI_TASK_NAME,
            unique_id=None,
        ),
    )
    hass.config_entries.async_update_entry(entry, version=2, minor_version=2)

    _LOGGER.debug(
        "Migration to version %s.%s successful", entry.version, entry.minor_version
    )

    return True
