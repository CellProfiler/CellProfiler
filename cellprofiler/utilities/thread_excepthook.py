# No CellProfiler copyright notice here.

# From http://bugs.python.org/issue1230540 - The threading module uses its own
# exception handler by default, preventing sys.excepthook from being called for
# uncaught exceptions.  This function overrides that behavior.
# Code by Ian Beaver.

import sys
import threading


def install_thread_sys_excepthook():
    "Workaround for threading module ignoring sys.excepthook()."

    if hasattr(threading, '__sys_excepthook_reinstalled__'):
        return
    threading.__sys_excepthook_reinstalled__ = True

    init_old = threading.Thread.__init__

    def init(self, *args, **kwargs):
        init_old(self, *args, **kwargs)
        run_old = self.run

        def run_with_except_hook(*args, **kw):
            try:
                run_old(*args, **kw)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                sys.excepthook(*sys.exc_info())
        self.run = run_with_except_hook

    threading.Thread.__init__ = init
