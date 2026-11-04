from importlib.util import find_spec


def pytest_configure(config):
    if find_spec("cellprofiler_core") is None:
        return

    from cellprofiler_core import preferences, reader

    preferences.set_headless()
    reader.fill_readers(check_config=False)


def pytest_sessionfinish(session, exitstatus):
    if find_spec("cellprofiler_core") is None:
        return

    from cellprofiler_core.constants.reader import ALL_READERS
    from cellprofiler_core.utilities.java import stop_java

    for reader in ALL_READERS.values():
        reader.clear_cached_readers()
    stop_java()
