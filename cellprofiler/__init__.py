from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import os.path


__test__ = False

__version_file__ = open(os.path.join(os.path.dirname(__file__), 'VERSION'))
__version__ = __version_file__.read().strip()