Batch Profiler is the tool we use at the Broad Institute for running
CellProfiler jobs. The tool requires a build of CellProfiler and its
dependencies using the Makefile.CP2 in the root directory. It also uses
GridEngine or compatible to dispatch jobs. You can tailor this by
monkey-patching or replacing bputilities' job dispatching functions.

SEE NOTES ABOUT APACHE CONFIGURATION BELOW!!!

You also need a MySQL database. Configure using the environment variables:

BATCHPROFILER_MYSQL_HOST=<DNS name of host>
BATCHPROFILER_MYSQL_PORT=<Port # of host> (optional)
BATCHPROFILER_MYSQL_USER=<User name on host>
BATCHPROFILER_MYSQL_PASSWORD=<Password for DB> (optional)
BATCHPROFILER_MYSQL_DATABASE=<MySQL database name> (optional, defaults to "batchprofiler")

The script, batchprofiler.sh, configures environment variables and runs Python
for each of the CGI scripts. It sources, "$(HOME)/.batchprofiler.sh" if it
exists - you can use this file to configure your environment.

Additional environment variables (which can be included in .batchprofiler.sh)

PREFIX - the prefix variable used when building. Defaults to ../../../..
JAVA_HOME - the location of your Java installation
LC_ALL - your locale (but will use UTF-8 encoding)

BATCHPROFILER_CPCHECKOUT - Versions of CellProfiler will be checked out into
                           this root directory.

The defaults for populating the fields in NewBatch.py:

BATCHPROFILER_DATA_DIR - the initial directory for searching for Batch_data.h5 files
BATCHPROFILER_EMAIL - the default email to use for responses
BATCHPROFILER_QUEUE - the grid engine queue to use when submitting a job
BATCHPROFILER_PROJECT - for fairshare queues, who to charge the job to
BATCHPROFILER_REVISION - the git hash or tag to use when checking out CP from GIT
BATCHPROFILER_BATCH_SIZE - default # of image sets / batch
BATCHPROFILER_MEMORY_LIMIT - job memory limit in MB
BATCHPROFILER_WRITE_DATA - defined if the default is to write the measurements files

Apache configuration:

You should set up a configuration file that points to the BatchProfiler
directory using an alias. !!! You should deny access to .sh files so that
someone can't run batchprofiler.sh !!!

Example batchprofiler.conf:

ScriptAlias /batchprofiler/cgi-bin/ <path-to>/CellProfiler/BatchProfiler/

<Files ~ "\.sh$">
    Order allow,deny
    Deny from all
</Files>
