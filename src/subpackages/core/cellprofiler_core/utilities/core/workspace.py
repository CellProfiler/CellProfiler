import h5py

from ..hdf5_dict import HDF5FileList
from ..hdf5_dict import HDF5Dict


def is_workspace_file(path):
    """Return True if the file along the given path is a workspace file"""
    if not h5py.is_hdf5(path):
        return False
    h5file = h5py.File(path, mode="r")
    try:
        if not HDF5FileList.has_file_list(h5file):
            return False
        return HDF5Dict.has_hdf5_dict(h5file)
    finally:
        h5file.close()
