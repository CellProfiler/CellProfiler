import h5py

from ..hdf5_dict import HDF5FileList
from ..hdf5_dict import HDF5Dict
from cellprofiler_library.measurements.measurement_model import LibraryMeasurements, R_FIRST_OBJECT_NUMBER, R_SECOND_OBJECT_NUMBER
import numpy


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

def add_library_measurements_to_workspace_measurements(workspace_measurements, library_measurements: LibraryMeasurements, module_num=None):
    """Add the library measurements to the workspace measurements

    workspace_measurements - the Measurements instance to add to
    library_measurements - the library measurements to be added
    module_num - the module number of the module that generated the library measurements
    """
    #
    # Record the measurements
    #
    # assume isinstance(workspace, Workspace)
    m = workspace_measurements
    # assume isinstance(m, Measurements)
    
    # Record Image Measurements
    for feature_name, value in library_measurements.image.items():
        m.add_image_measurement(feature_name, value)
    
    # Record Object Measurements
    for object_name, features in library_measurements.objects.items():
        for feature_name, data in features.items():
            m.add_measurement(object_name, feature_name, data)

    relationship_groups = library_measurements.get_relationship_groups()
    if relationship_groups and module_num is None:
        raise ValueError(
            "module_num must be provided to add_library_measurements_to_workspace_measurements "
            "when library_measurements contains relationships"
        )
    for relationship in relationship_groups:
        data = library_measurements.get_relationships(
            relationship.relationship,
            relationship.object_name1,
            relationship.object_name2
        )
        n_records = len(data)
        img_nums = numpy.ones(n_records, int) * m.image_set_number

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
