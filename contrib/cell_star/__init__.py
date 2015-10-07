# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz, Szymon Stoma'

__all__ = ["core", "process", "test", "utils"]

import sys
import os

# Adds this package internals (PIL versions) into current sys.path (search in during import)
current_module = sys.modules[__name__]
python_paths = []
variable = os.environ.get('PYTHONPATH')
if variable is not None:
    python_paths = variable.split(os.pathsep)
sys.path = sys.path + python_paths + [os.path.dirname(current_module.__file__)]
