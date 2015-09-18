"""
the workspace for an imageset
"""

import logging
import StringIO
import numpy as np
import h5py
import os
import cellprofiler.grid
import cellprofiler.utilities.hdf5_dict

logger = logging.getLogger(__name__)


'''Continue to run the pipeline

Set workspace.disposition to DISPOSITION_CONTINUE to go to the next module.
This is the default.
'''
DISPOSITION_CONTINUE = "Continue"
'''Skip remaining modules

Set workspace.disposition to DISPOSITION_SKIP to skip to the next image set
in the pipeline.
'''
DISPOSITION_SKIP = "Skip"
'''Pause and let the UI run

Set workspace.disposition to DISPOSITION_PAUSE to pause the UI. Set
it back to DISPOSITION_CONTINUE to resume.
'''
DISPOSITION_PAUSE = "Pause"
'''Cancel running the pipeline'''
DISPOSITION_CANCEL = "Cancel"

def is_workspace_file(path):
    '''Return True if the file along the given path is a workspace file'''
    if not h5py.is_hdf5(path):
        return False
    h5file = h5py.File(path, mode="r")
    try:
        if not cellprofiler.utilities.hdf5_dict.HDF5FileList.has_file_list(h5file):
            return False
        return cellprofiler.utilities.hdf5_dict.HDF5Dict.has_hdf5_dict(h5file)
    finally:
        h5file.close()
