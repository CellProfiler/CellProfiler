"""
Run pipelines on imagesets to produce measurements.
"""

import sys


use_analysis = True

DEBUG = "DEBUG"
ANNOUNCE_DONE = "DONE"


###############################
# Request, Replies, Events
###############################


if sys.platform == "darwin":
    pass

if __name__ == "__main__":
    # This is an ugly hack, but it's necesary to unify the Request/Reply
    # classes above, so that regardless of whether this is the current module,
    # or a separately imported one, they see the same classes.
    import cellprofiler_core.analysis
    from ._runner import Runner

    globals().update(cellprofiler_core.analysis.__dict__)

    Runner.start_workers(2)
    Runner.stop_workers()
