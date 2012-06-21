import nose
import sys

import numpy as np
np.seterr(all='ignore')

if '--noguitests' in sys.argv:
    sys.argv.remove('--noguitests')
    import cellprofiler.preferences as cpprefs
    cpprefs.set_headless()
    sys.modules['wx'] = None
    import matplotlib
    matplotlib.use('agg')

if '--nojavatests' in sys.argv:
    sys.argv.remove('--nojavatests')
    import cellprofiler.utilities.jutil as jutil
    jutil.start_vm = None

if len(sys.argv) == 0:
    args = ['--testmatch=(?:^)test_.*']
else:
    args = sys.argv

nose.main(argv=args + ['--exe'])
