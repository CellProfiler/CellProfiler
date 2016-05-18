'''walk_in_background.py - walk a directory tree from a background thread

This module walks a directory tree, incrementally reporting the
results to the UI thread.
'''

import logging

logger = logging.getLogger(__name__)


#
# Some file types won't open with BioFormats unless BioFormats is allowed
# to look at the file contents while determining the appropriate file reader.
# Others will try too hard and will look at associated files, even with
# the grouping option turned off. So here's the list of those that
# absolutely need it.
#


