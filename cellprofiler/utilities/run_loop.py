'''run_loop.py - support for running the OS/X event loop

OS/X needs to run an event loop on the main thread in order to run a user
interface. This is handled by WX when running in 32-bit mode, but needs to
be done explicitly in 64-bit mode if Java loads AWT (e.g. ImageJ 1.0 or
badly-written ImageJ 2.0 plugin).

The run loop stalls the main thread on other platforms without doing anything
else.

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org

'''
import sys
if sys.platform == "darwin":
    from cellprofiler.utilities.javabridge \
         import mac_enter_run_loop as enter_run_loop
    from cellprofiler.utilities.javabridge \
         import mac_stop_run_loop as stop_run_loop
else:
    import threading
    
    #
    # A mutex and condition variable for a dummy run loop
    #
    run_loop_lock = threading.Lock()
    run_loop_cv = threading.Condition(run_loop_lock)
    run_loop_state = 0
    
    def enter_run_loop():
	global run_loop_state
	with run_loop_lock:
	    if run_loop_state == 0:
		run_loop_state = 1
		run_loop_cv.notify_all()
	    while run_loop_state == 1:
		run_loop_cv.wait()
	    run_loop_state = 3
	    run_loop_cv.notify_all()
		
    def stop_run_loop():
	global run_loop_state
	with run_loop_lock:
	    while run_loop_state == 0:
		run_loop_cv.wait()
	    if run_loop_state == 1:
		run_loop_state = 2
		run_loop_cv.notify_all()
		while run_loop_state == 2:
		    run_loop_cv.wait()
