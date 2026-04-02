import h5py

from ..hdf5_dict import HDF5FileList
from ..hdf5_dict import HDF5Dict
from cellprofiler_library.measurement_model import LibraryMeasurements, R_FIRST_OBJECT_NUMBER, R_SECOND_OBJECT_NUMBER
import numpy as np
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

def add_library_measurements_to_workspace(library_measurements: LibraryMeasurements, workspace, module_num):
    """Add the library measurements to the workspace

    library_measurements - the library measurements to be added
    workspace - the workspace to which the measurements will be added
    module_num - the module number of the module that generated the library measurements
    """
    #
    # Record the measurements
    #
    # assume isinstance(workspace, Workspace)
    m = workspace.measurements
    # assume isinstance(m, Measurements)
    
    # Record Image Measurements
    for feature_name, value in library_measurements.image.items():
        m.add_image_measurement(feature_name, value)
    
    # Record Object Measurements
    for object_name, features in library_measurements.objects.items():
        for feature_name, data in features.items():
            m.add_measurement(object_name, feature_name, data)

    for relationship in library_measurements.get_relationship_groups():
        data = library_measurements.get_relationships(
            relationship.relationship,
            relationship.object_name1,
            relationship.object_name2
        )
        n_records = len(data)
        img_nums = np.ones(n_records, int) * m.image_set_number

        m.add_relate_measurement(
            module_num,
            relationship.relationship,
            relationship.object_name1,
            relationship.object_name2,
            img_nums,
            data[R_FIRST_OBJECT_NUMBER],
            img_nums,
            data[R_SECOND_OBJECT_NUMBER], 
        )
