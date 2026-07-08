# coding=utf-8
"""Internationalization support for the CellProfiler GUI."""

from .manager import (
    DEFAULT_LANGUAGE,
    FALLBACK_LANGUAGE,
    LocaleManager,
    _,
    add_language_change_listener,
    get_available_languages,
    get_language,
    get_manager,
    init_from_preferences,
    remove_language_change_listener,
    set_language,
)

__all__ = [
    "DEFAULT_LANGUAGE",
    "FALLBACK_LANGUAGE",
    "LocaleManager",
    "_",
    "add_language_change_listener",
    "get_available_languages",
    "get_language",
    "get_manager",
    "init_from_preferences",
    "remove_language_change_listener",
    "set_language",
]
