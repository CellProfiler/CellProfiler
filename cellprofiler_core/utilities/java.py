"""
java.py - CellProfiler-specific JVM utilities
"""

import logging
import os
import scyjava

import cellprofiler_core.preferences


LOGGER = logging.getLogger(__name__)

def start_java():
    """Start CellProfiler's JVM via Javabridge

    JVM parameters are harvested from preferences and
    the environment variables:

    CP_JDWP_PORT - port # for debugging Java within the JVM
    cpprefs.get_awt_headless() - controls java.awt.headless to prevent
        awt from being invoked
    """
    if scyjava.jvm_started():
        return

    # Add Bio-Formats Java dependency.
    scyjava.config.endpoints.append("ome:formats-gpl")
    scyjava.config.endpoints.append("org.scijava:scijava-config")

    LOGGER.info("Initializing Java Virtual Machine")
    args = [
        "-Dloci.bioformats.loaded=true",
    ]

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
    scyjava.start_jvm(options=args)
    #
    # Enable Bio-Formats directory cacheing
    #
    Location = scyjava.jimport("loci.common.Location")
    Location.cacheDirectoryListings(True)
    LOGGER.debug("Enabled Bio-formats directory cacheing")


def stop_java():
    LOGGER.info("Shutting down Java Virtual Machine")
    scyjava.shutdown_jvm()

def jimport(package):
    start_java()
    return scyjava.jimport(package)
