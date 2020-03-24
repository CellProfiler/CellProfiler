import bioformats.formatreader

import cellprofiler_core.preferences
import cellprofiler_core.utilities.java


def pytest_sessionstart(session):
    cellprofiler_core.preferences.set_headless()

    cellprofiler_core.utilities.java.start_java()


def pytest_sessionfinish(session, exitstatus):
    bioformats.formatreader.clear_image_reader_cache()

    cellprofiler_core.utilities.java.stop_java()

    return exitstatus
