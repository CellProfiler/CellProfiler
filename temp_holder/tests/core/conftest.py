import cellprofiler_core.preferences
import cellprofiler_core.reader
import cellprofiler_core.utilities.java


def pytest_sessionstart(session):
    cellprofiler_core.preferences.set_headless()
    cellprofiler_core.reader.fill_readers(check_config=True)


def pytest_sessionfinish(session, exitstatus):
    cellprofiler_core.utilities.java.stop_java()

    return exitstatus
