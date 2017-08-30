"""
cpjvm.py - CellProfiler-specific JVM utilities
"""

import bioformats.formatreader
import cellprofiler.preferences
import javabridge
import logging
import os
import pkg_resources
import sys
import prokaryote
import tempfile

logger = logging.getLogger(__name__)


def get_jars():
    """
    Get the final list of JAR files passed to javabridge
    """

    class_path = []
    if os.environ.has_key("CLASSPATH"):
        class_path += os.environ["CLASSPATH"].split(os.pathsep)
        logging.debug(
                "Adding Java class path from environment variable, ""CLASSPATH""")
        logging.debug("    CLASSPATH=" + os.environ["CLASSPATH"])

    # prokaryote_resources = pkg_resources.get_distribution("prokaryote")

    # filename = "prokaryote-{}.jar".format(prokaryote_resources.version)

    # os.path.join(prokaryote_resources.location, prokaryote)

    pathname = os.path.dirname(prokaryote.__file__)

    jar_files = [os.path.join(pathname, f) for f in os.listdir(pathname) if f.lower().endswith(".jar")]

    class_path += javabridge.JARS + jar_files

    if sys.platform.startswith("win") and not hasattr(sys, 'frozen'):
        # Have to find tools.jar
        from javabridge.locate import find_jdk
        jdk_path = find_jdk()
        if jdk_path is not None:
            tools_jar = os.path.join(jdk_path, "lib", "tools.jar")
            class_path.append(tools_jar)
        else:
            logger.warning("Failed to find tools.jar")
    return class_path


def find_logback_xml():
    '''Find the location of the logback.xml file for Java logging config

    Paths to search are the current directory, the utilities directory
    and ../../java/src/main/resources
    '''
    paths = [os.curdir,
             os.path.dirname(__file__),
             os.path.join(
                     os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                     "java", "src", "main", "resources")]
    for path in paths:
        target = os.path.join(path, "logback.xml")
        if os.path.isfile(target):
            return target


def cp_start_vm():
    '''Start CellProfiler's JVM via Javabridge

    JVM parameters are harvested from preferences and
    the environment variables:

    CP_JDWP_PORT - port # for debugging Java within the JVM
    cpprefs.get_awt_headless() - controls java.awt.headless to prevent
        awt from being invoked
    '''
    args = ["-Dloci.bioformats.loaded=true",
            "-Djava.util.prefs.PreferencesFactory=" +
            "org.cellprofiler.headlesspreferences.HeadlessPreferencesFactory"]

    logback_path = find_logback_xml()

    if logback_path is not None:
        if sys.platform.startswith("win"):
            logback_path = logback_path.replace("\\", "/")
            if logback_path[1] == ':':
                # \\localhost\x$ is same as x:
                logback_path = "//localhost/" + logback_path[0] + "$" + \
                               logback_path[2:]
        args.append("-Dlogback.configurationFile=%s" % logback_path)

    class_path = get_jars()
    awt_headless = cellprofiler.preferences.get_awt_headless()
    if awt_headless:
        logger.debug("JVM will be started with AWT in headless mode")
        args.append("-Djava.awt.headless=true")

    heap_size = str(cellprofiler.preferences.get_jvm_heap_mb()) + "m"
    if os.environ.has_key("CP_JDWP_PORT"):
        args.append(
                ("-agentlib:jdwp=transport=dt_socket,address=127.0.0.1:%s"
                 ",server=y,suspend=n") % os.environ["CP_JDWP_PORT"])
    javabridge.start_vm(args=args,
                        class_path=class_path,
                        max_heap_size=heap_size)
    #
    # Enable Bio-Formats directory cacheing
    #
    try:
        c_location = javabridge.JClassWrapper("loci.common.Location")
        c_location.cacheDirectoryListings(True)
        logger.debug("Enabled Bio-formats directory cacheing")
    except:
        logger.warning("Bioformats version does not support directory cacheing")
    #
    # Monkey-patch bioformats.formatreader.get_class_list to add
    # the classes we added to loci.formats.in
    #
    old_get_class_list = bioformats.formatreader.get_class_list

    def get_class_list():
        "Return a wrapped instance of loci.formats.ClassList"

        env = javabridge.get_env()
        class_list = old_get_class_list()
        rais_classname = 'loci/common/RandomAccessInputStream'
        #
        # Move any class to the back that thinks it can read garbage
        #
        fd, path = tempfile.mkstemp(suffix=".garbage",
                                    dir=cellprofiler.preferences.get_temporary_directory())
        stream = None
        try:
            os.write(fd, "This is not an image file")
            os.fsync(fd)
            stream = javabridge.make_instance(
                    rais_classname, '(Ljava/lang/String;)V', path)
            problem_classes = []
            for klass in env.get_object_array_elements(class_list.get_classes()):
                try:
                    instance = javabridge.call(
                            klass, "newInstance", "()Ljava/lang/Object;")
                    can_read_garbage = javabridge.call(
                            instance, "isThisType",
                            "(L%s;)Z" % rais_classname, stream)
                    if can_read_garbage:
                        problem_classes.append(klass)
                        class_list.remove_class(klass)
                except:
                    logger.info("Caught exception in %s.isThisType",
                                javabridge.to_string(klass))
        finally:
            os.close(fd)
            javabridge.call(stream, "close", "()V")
            os.remove(path)

        for classname in ("loci.formats.in.FlowSightReader",
                          "loci.formats.in.IM3Reader"):
            try:
                klass = javabridge.class_for_name(classname)
                class_list.add_class(klass)
            except:
                logger.warn("Could not find Bio-formats reader %s" % classname,
                            exc_info=1)
        for klass in problem_classes:
            class_list.add_class(klass)
        return class_list

    bioformats.formatreader.get_class_list = get_class_list


def cp_stop_vm(kill=True):
    """
    Shut down the Java VM
    """
    javabridge.kill_vm()
