import nose
import sys

from killjavabridge import KillVMPlugin

if len(sys.argv) == 0:
    args = ['--testmatch=(?:^)test_.*']
else:
    args = sys.argv[1:]

nose.main(argv=args + ['--with-kill-vm', '--exe'],
          addplugins=[KillVMPlugin()])
