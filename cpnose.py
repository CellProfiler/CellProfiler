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
import nose
import sys

import numpy as np
np.seterr(all='ignore')


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
    import cellprofiler.utilities.jutil as jutil
    jutil.start_vm = mock_start_vm
elif not with_guitests:
    import cellprofiler.utilities.jutil as jutil
    jutil.activate_awt = mock_start_vm
    jutil.execute_future_in_main_thread = mock_start_vm
    jutil.execute_runnable_in_main_thread = mock_start_vm

addplugins = []
if '--with-kill-vm' in sys.argv:
    from killjavabridge import KillVMPlugin
    addplugins.append(KillVMPlugin())

if len(sys.argv) == 0:
    args = ['--testmatch=(?:^)test_.*']
else:
    args = sys.argv

nose.main(argv=args + ['--exe'], addplugins=addplugins)
