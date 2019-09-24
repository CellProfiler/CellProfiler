import bioformats

import nucleus.preferences
import nucleus.utilities.java


def pytest_sessionstart(session):
    nucleus.preferences.set_headless()

    nucleus.utilities.java.start_java()


def pytest_sessionfinish(session, exitstatus):
    bioformats.formatreader.clear_image_reader_cache()

    nucleus.utilities.java.stop_java()

    return exitstatus
