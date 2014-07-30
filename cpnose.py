"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

from unittest.case import SkipTest
from pkg_resources import load_entry_point
import javabridge
import nose
import nose.plugins
import os
import sys

from cellprofiler.utilities.cpjvm import \
     get_path_to_jars, get_cellprofiler_jars, get_patcher_args

import numpy as np
np.seterr(all='ignore')

class CPShutdownPlugin(nose.plugins.Plugin):
    '''CellProfiler shutdown plugin
    
    This plugin shuts down threads that might happen to have been
    left open.
    '''
    name = "cpshutdown"
    score = 100
    
    def begin(self):
        pass
    def finalize(self, result):
        try:
            from ilastik.core.jobMachine import GLOBAL_WM
            GLOBAL_WM.stopWorkers()
        except:
            logging.root.warn("Failed to stop Ilastik")
        try:
            from cellprofiler.utilities.zmqrequest import join_to_the_boundary
            join_to_the_boundary()
        except:
            logging.root.warn("Failed to stop zmq boundary")
    
class MockModule(object):
    def __getattr__(self, *args, **kwargs):
        raise SkipTest

    def __getitem__(self, *args, **kwargs):
        raise SkipTest


if '--noguitests' in sys.argv:
    sys.argv.remove('--noguitests')
    import cellprofiler.preferences as cpprefs
    cpprefs.set_headless()
    sys.modules['wx'] = MockModule()
    sys.modules['wx.html'] = MockModule()
    import matplotlib
    matplotlib.use('agg')
    with_guitests = False
else:
    with_guitests = True

def mock_start_vm(*args, **kwargs):
    raise SkipTest

if '--nojavatests' in sys.argv:
    sys.argv.remove('--nojavatests')
    javabridge.start_vm = mock_start_vm
elif not with_guitests:
    javabridge.activate_awt = mock_start_vm
    javabridge.execute_future_in_main_thread = mock_start_vm
    javabridge.execute_runnable_in_main_thread = mock_start_vm

jar_directory = get_path_to_jars()
class_path = os.pathsep.join(
    [os.path.join(jar_directory, jarfile) 
     for jarfile in get_cellprofiler_jars()])

addplugins = [CPShutdownPlugin()]

if not "--with-cpshutdown" in sys.argv:
    sys.argv.append("--with-cpshutdown")
if '--with-kill-vm' in sys.argv:
    sys.argv[sys.argv.index('--with-kill-vm')] = '--with-javabridge'
if '--with-javabridge' in sys.argv:
    javabridge_plugin_class = load_entry_point(
        'javabridge', 'nose.plugins.0.10', 'javabridge')
    javabridge_plugin_class.extra_jvm_args += get_patcher_args(class_path)
    if "CP_JDWP_PORT" in os.environ:
        javabridge_plugin_class.extra_jvm_args.append(
            ("-agentlib:jdwp=transport=dt_socket,address=127.0.0.1:%s"
             ",server=y,suspend=n") % os.environ["CP_JDWP_PORT"])
    addplugins.append(javabridge_plugin_class())

if len(sys.argv) == 0:
    args = ['--testmatch=(?:^)test_.*']
else:
    args = sys.argv

nose.main(argv=args + ['--exe', '--classpath=%s' % class_path], 
          addplugins=addplugins)
