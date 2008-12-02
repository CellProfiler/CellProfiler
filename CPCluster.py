#!/util/bin/python

# This is a wrapper around the compiled Matlab CPCluster program,
# which has a tendency to hang on the cluster.  We wrap it in a script
# that watches the matlab program, and if it fails to report back (via
# a HUP signal after each module), it kills it and restarts it (up to
# three times).

import os
import sys
from subprocess import Popen
import signal
import copy

# set up the environment for running that compiled matlab version.
cpcluster_home = sys.path[0]
os.environ['CPCLUSTERHOME'] = cpcluster_home
mcr_path = '/imaging/analysis/CPCluster/MCR/v78'
os.environ['LD_LIBRARY_PATH'] = '%(mcr_path)s/runtime/glnxa64:%(mcr_path)s/sys/os/glnxa64:%(mcr_path)s/bin/glnxa64'%(locals())
os.environ['HOME'] = '/imaging/analysis/CPCluster'

def run_job_with_timeout(full_command, timeout):
    '''
    run a command with a timeout, and return whether it should be retried because of early failure.
    '''

    # necessary for nested scoping
    status = {'ever_had_heartbeat' : False, 'was_killed' : False}
    
    # set up the HUP signal handler before calling the subprocess
    def hup_handler(signum, frame):
        # record if the job has ever reported back
        status['ever_had_heartbeat'] = True
        
        # reset the timeout (no need to reset the HUP handler, python handles that)
        # this removes the previously running alarm
        remaining_time = signal.alarm(timeout)
        # give it credit for running fast
        signal.alarm(remaining_time + timeout)

    # register the HUP handler
    signal.signal(signal.SIGHUP, hup_handler)

    subproc = Popen(full_command, env=os.environ)

    # set up the alarm handler
    def alarm_handler(sig, frame):
        status['was_killed'] = True
        os.kill(subproc.pid, signal.SIGKILL)

    # register the alarm handler
    signal.signal(signal.SIGALRM, alarm_handler)

    # start the timer
    signal.alarm(timeout)

    while subproc.poll() == None:
        # the call to wait() will be interrupted by the HUP from the
        # child, so need to catch that.
        try:
            subproc.wait()
        except OSError: 
            pass

    # If the job ever reported back, then if it was killed, it was for
    # taking too long, and shouldn't be retried.  Otherwise, it
    # probably crashed before starting (intermitted Matlab failure),
    # and should be given another chance.  It also may have exited
    # before ever reporting back.

    if status['was_killed'] and status['ever_had_heartbeat']:
        print >>sys.stderr, "CPCluster.py: Matlab process timed out, killed."

    return status['was_killed'] and (not status['ever_had_heartbeat'])
    

# This script is called in two ways:
# no arguments -> unpack the .ctf
# many arguments -> run in "watchdog" mode, last argument is the timeout,
#   which will be replaced with a "heartbeat" command for the child to run.

command = cpcluster_home + '/CPCluster'

if len(sys.argv) == 1:
    # no arguments, just run the command
    subproc = Popen(command, env=os.environ)
    subproc.wait()
else:
    # arguments, run with timeout

    # get the timeout
    timeout = int(sys.argv[-1])

    # replace last argument with the keepalive command
    arguments = copy.copy(sys.argv[1:])
    arguments[-1] = "kill -HUP " + str(os.getpid())

    # try three times to run the job
    for retries in range(3):
        should_rerun = run_job_with_timeout([command] + arguments, timeout)
        if not should_rerun:
            break
