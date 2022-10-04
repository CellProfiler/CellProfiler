"""
java.py - CellProfiler-specific JVM utilities
"""

import logging
import os
import sys
import threading

import cellprofiler_core.preferences


LOGGER = logging.getLogger(__name__)


def get_jars():
    """
    Get the final list of JAR files passed to Java
    """

    class_path = []
    if "CLASSPATH" in os.environ:
        class_path += os.environ["CLASSPATH"].split(os.pathsep)
        LOGGER.debug(
            "Adding Java class path from environment variable, " "CLASSPATH" ""
        )
        LOGGER.debug("    CLASSPATH=" + os.environ["CLASSPATH"])

    pathname = os.path.dirname(prokaryote.__file__)

    return [
        os.path.join(pathname, f)
        for f in os.listdir(pathname)
        if f.lower().endswith(".jar")
    ]


def find_logback_xml():
    """Find the location of the logback.xml file for Java logging config

    Paths to search are the current directory, the utilities directory
    and ../../java/src/main/resources
    """
    paths = [
        os.curdir,
        os.path.dirname(__file__),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "java",
            "src",
            "main",
            "resources",
        ),
    ]
    for path in paths:
        target = os.path.join(path, "logback.xml")
        if os.path.isfile(target):
            return target


def start_java():
    """Start CellProfiler's JVM via Javabridge

    JVM parameters are harvested from preferences and
    the environment variables:

    CP_JDWP_PORT - port # for debugging Java within the JVM
    cpprefs.get_awt_headless() - controls java.awt.headless to prevent
        awt from being invoked
    """
    thread_id = threading.get_ident()
    LOGGER.info("Initializing Java Virtual Machine")
    args = [
        "-Dloci.bioformats.loaded=true",
        "-Djava.util.prefs.PreferencesFactory="
        + "org.cellprofiler.headlesspreferences.HeadlessPreferencesFactory",
    ]

    logback_path = find_logback_xml()

    if logback_path is not None:
        if sys.platform.startswith("win"):
            logback_path = logback_path.replace("\\", "/")
            if logback_path[1] == ":":
                # \\localhost\x$ is same as x:
                logback_path = "//localhost/" + logback_path[0] + "$" + logback_path[2:]
        args.append("-Dlogback.configurationFile=%s" % logback_path)

    class_path = get_jars()
    awt_headless = cellprofiler_core.preferences.get_awt_headless()
    if awt_headless:
        LOGGER.debug("JVM will be started with AWT in headless mode")
        args.append("-Djava.awt.headless=true")

    if "CP_JDWP_PORT" in os.environ:
        args.append(
            (
                "-agentlib:jdwp=transport=dt_socket,address=127.0.0.1:%s"
                ",server=y,suspend=n"
            )
            % os.environ["CP_JDWP_PORT"]
        )
    #CTR FIXME
    #scyjava.start_jvm(args=args, class_path=class_path)
    #
    # Enable Bio-Formats directory cacheing
    #
    #Location = scyjava.jimport("loci.common.Location")
    #Location.cacheDirectoryListings(True)
    LOGGER.debug("Enabled Bio-formats directory cacheing")


def stop_java():
    LOGGER.info("Shutting down Java Virtual Machine")
    #CTR FIXME: scyjava.shutdown_jvm()
