Configuring Logging
===================

CellProfiler prints diagnostic messages to the console by default. You
can change this behavior for most messages by configuring logging. The
simplest way to do this is to use the command-line switch, “-L”, to
set the log level. For instance, to show error messages or more
critical events, start CellProfiler like this:
``CellProfiler -L ERROR``
The following is a list of log levels that can be used:

-  **DEBUG:** Detailed diagnostic information
-  **INFO:** Informational messages that confirm normal progress
-  **WARNING:** Messages that report problems that might need attention
-  **ERROR:** Messages that report unrecoverable errors that result in
   data loss or termination of the current operation.
-  **CRITICAL:** Messages indicating that CellProfiler should be
   restarted or is incapable of running.

You can tailor CellProfiler’s logging with much more control using a
logging configuration file. You specify the file name in place of the
log level on the command line, like this:

``CellProfiler -L ~/CellProfiler/my_log_config.cfg``

Files are in the Microsoft .ini format which is grouped into
categories enclosed in square brackets and the key/value pairs for
each category. Here is an example file:

::

    [loggers]
    keys=root,pipelineStatistics

    [handlers]
    keys=console,logfile

    [formatters]
    keys=detailed

    [logger_root]
    level=WARNING
    handlers=console

    [logger_pipelineStatistics]
    level=INFO
    handlers=logfile
    qualname=PipelineStatistics
    propagate=0

    [handler_console]
    class=StreamHandler
    level=WARNING
    formatter=detailed
    args=(sys.stderr,)

    [handler_logfile]
    class=FileHandler
    level=INFO
    formatter=detailed
    args=('cellprofiler.log', 'w')

    [formatter_detailed]
    format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
    datefmt=

The above file would print warnings and errors to the console for all
messages but “pipeline statistics” which are configured using the
*PipelineStatistics* logger are written to the file `cellprofiler.log`
instead. The *PipelineStatistics* logger is the logger that is used to
print progress messages when the pipeline is run. You can find out which
loggers are being used to write particular messages by printing all
messages with a formatter that prints the logger name (“%(name)s”).
The format of the file is described in greater detail `here`_.

.. _here: http://docs.python.org/2.7/howto/logging.html#configuring-logging
