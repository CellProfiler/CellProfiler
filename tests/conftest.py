import cellprofiler_core.preferences
import cellprofiler_core.reader
import cellprofiler_core.utilities.java
from cellprofiler_core.constants.reader import ALL_READERS


def pytest_configure(config):
    cellprofiler_core.preferences.set_headless()
    cellprofiler_core.reader.fill_readers(check_config=False)


def pytest_sessionfinish(session, exitstatus):
    for reader in ALL_READERS.values():
        reader.clear_cached_readers()
    cellprofiler_core.utilities.java.stop_java()
