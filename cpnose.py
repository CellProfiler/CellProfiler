import nose
import sys

from killjavabridge import KillVMPlugin

import numpy as np
np.seterr(all='ignore')

if '--noguitests' in sys.argv:
    sys.argv.remove('--noguitests')
    import wx
    for s in dir(wx):
        del wx.__dict__[s]

if len(sys.argv) == 0:
    args = ['--testmatch=(?:^)test_.*']
else:
    args = sys.argv

nose.main(argv=args + ['--with-kill-vm', '--exe'],
          addplugins=[KillVMPlugin()])
