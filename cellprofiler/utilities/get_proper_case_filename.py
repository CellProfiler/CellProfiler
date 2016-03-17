'''get_proper_case_filename.py - convert filename to proper case for Windows
'''

import os
import sys

if sys.platform.startswith("win"):
    import _get_proper_case_filename
    from _get_proper_case_filename \
         import get_file_attributes, set_file_attributes, \
         FILE_ATTRIBUTE_ARCHIVE, FILE_ATTRIBUTE_HIDDEN, \
         FILE_ATTRIBUTE_NOT_CONTENT_INDEXED, FILE_ATTRIBUTE_OFFLINE, \
         FILE_ATTRIBUTE_READONLY, FILE_ATTRIBUTE_SYSTEM, \
         FILE_ATTRIBUTE_TEMPORARY

    def get_proper_case_filename(path):
        if False:
            result = _get_proper_case_filename.get_proper_case_filename(unicode(path))
            if result is None:
                return path
            if (len(result) and len(path) and
                path[-1] == os.path.sep and result[-1] != os.path.sep):
                result += os.path.sep
            return result
        else:
            return path
else:
    def get_proper_case_filename(path):
        return path
