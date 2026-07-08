# coding=utf-8
"""Localization manager for CellProfiler GUI strings."""

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "en"
FALLBACK_LANGUAGE = "en"


class LocaleManager:
    """Loads JSON translation catalogs and resolves UI strings."""

    def __init__(self):
        self._language = DEFAULT_LANGUAGE
        self._catalogs = {}
        self._listeners = []
        self._translations_dir = Path(__file__).parent / "translations"
        self._discover_and_load_catalogs()

    def _discover_and_load_catalogs(self):
        if not self._translations_dir.is_dir():
            LOGGER.warning(
                "Translation directory not found: %s", self._translations_dir
            )
            self._catalogs[DEFAULT_LANGUAGE] = {}
            return

        for path in sorted(self._translations_dir.glob("*.json")):
            language_code = path.stem
            try:
                with path.open(encoding="utf-8") as handle:
                    self._catalogs[language_code] = json.load(handle)
            except (OSError, json.JSONDecodeError):
                LOGGER.error("Failed to load translation catalog: %s", path, exc_info=True)
                self._catalogs[language_code] = {}

        if DEFAULT_LANGUAGE not in self._catalogs:
            self._catalogs[DEFAULT_LANGUAGE] = {}

    def translate(self, message):
        if message is None:
            return ""
        if not isinstance(message, str):
            return message

        catalog = self._catalogs.get(self._language, {})
        if message in catalog:
            return catalog[message]

        if self._language != FALLBACK_LANGUAGE:
            fallback_catalog = self._catalogs.get(FALLBACK_LANGUAGE, {})
            if message in fallback_catalog:
                return fallback_catalog[message]

        return message

    def get_language(self):
        return self._language

    def get_available_languages(self):
        return [
            {"code": "en", "name": "English"},
            {"code": "zh", "name": "简体中文"},
        ]

    def get_available_language_codes(self):
        return [language["code"] for language in self.get_available_languages()]

    def set_language(self, language, persist=True):
        if language not in self.get_available_language_codes():
            language = DEFAULT_LANGUAGE

        if language == self._language:
            return False

        self._language = language

        if persist:
            from cellprofiler_core.preferences import set_ui_language

            set_ui_language(language)

        self._notify_listeners()
        return True

    def load_language_from_preferences(self):
        from cellprofiler_core.preferences import get_ui_language

        language = get_ui_language()
        if language not in self.get_available_language_codes():
            language = DEFAULT_LANGUAGE
        self._language = language

    def add_listener(self, callback):
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self):
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:
                LOGGER.error("Language change listener failed", exc_info=True)


_manager = LocaleManager()


def get_manager():
    return _manager


def _(message):
    """Translate a user-facing string using the active locale."""
    return _manager.translate(message)


def get_language():
    return _manager.get_language()


def get_available_languages():
    return _manager.get_available_languages()


def set_language(language):
    return _manager.set_language(language)


def init_from_preferences():
    _manager.load_language_from_preferences()


def add_language_change_listener(callback):
    _manager.add_listener(callback)


def remove_language_change_listener(callback):
    _manager.remove_listener(callback)
