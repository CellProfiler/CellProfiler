"""_workspace.py - the workspace for an imageset
"""

import io
import logging
import os

import h5py

import cellprofiler_core.utilities.measurement
from ._disposition_changed_event import DispositionChangedEvent
from ..constants.measurement import EXPERIMENT
from ..constants.workspace import DISPOSITION_CONTINUE
from ..utilities.hdf5_dict import HDF5FileList


LOGGER = logging.getLogger(__name__)

class Workspace:
    """The workspace contains the processing information and state for
    a pipeline run on an image set
    """

    def __init__(
        self,
        pipeline,
        module,
        image_set,
        object_set,
        measurements,
        image_set_list,
        frame=None,
        create_new_window=False,
        outlines=None,
    ):
        """Workspace constructor

        pipeline          - the pipeline of modules being run
        module            - the current module to run (a CPModule instance)
        image_set         - the set of images available for this iteration
                            (a cpimage.ImageSet instance)
        object_set        - an object.ObjectSet instance
        image_set_list    - the list of all images
        frame             - the application's frame, or None for no display
        create_new_window - True to create another frame, even if one is open
                            False to reuse the current frame.
        """
        if outlines is None:
            outlines = {}
        self.__pipeline = pipeline
        self.__module = module
        self.__image_set = image_set
        self.__object_set = object_set
        self.__measurements = measurements
        self.__image_set_list = image_set_list
        self.__frame = frame
        self.__do_show = frame is not None
        self.__outlines = outlines
        self.__windows_used = []
        self.__create_new_window = create_new_window
        self.__grid = {}
        self.__disposition = DISPOSITION_CONTINUE
        self.__disposition_listeners = []
        self.__in_background = (
            False  # controls checks for calls to create_or_find_figure()
        )
        self.__filename = None
        self.__file_list = None
        self.__loading = False
        if measurements is not None:
            self.set_file_list(HDF5FileList(measurements.hdf5_dict.hdf5_file))
        self.__notification_callbacks = []

        self.interaction_handler = None
        self.post_run_display_handler = None
        self.post_group_display_handler = None
        self.cancel_handler = None

        class DisplayData(object):
            pass

        self.display_data = DisplayData()
        """Object into which the module's run() method can stuff items
        that must be available later for display()."""

    def refresh(self):
        """Refresh any windows created during use"""
        for window in self.__windows_used:
            window.figure.canvas.draw()

    def get_windows_used(self):
        return self.__windows_used

    def get_pipeline(self):
        """Get the pipeline being run"""
        return self.__pipeline

    pipeline = property(get_pipeline)

    def get_image_set(self):
        """The image set is the set of images currently being processed
        """
        return self.__image_set

    def set_image_set_for_testing_only(self, image_set_number):
        self.__image_set = self.image_set_list.get_image_set(image_set_number)

    image_set = property(get_image_set)

    def get_image_set_list(self):
        """The list of all image sets"""
        return self.__image_set_list

    image_set_list = property(get_image_set_list)

    def get_object_set(self):
        """The object set is the set of image labels for the current image set
        """
        return self.__object_set

    object_set = property(get_object_set)

    def get_objects(self, objects_name):
        """Return the objects.Objects instance for the given name.

        objects_name - the name of the objects to retrieve
        """
        return self.object_set.get_objects(objects_name)

    def get_measurements(self):
        """The measurements contain measurements made on images and objects
        """
        return self.__measurements

    measurements = property(get_measurements)

    def add_measurement(self, object_name, feature_name, data):
        """Add a measurement to the workspace's measurements

        object_name - name of the objects measured or 'Image'
        feature_name - name of the feature measured
        data - the result of the measurement
        """
        self.measurements.add_measurement(object_name, feature_name, data)

    def get_file_list(self):
        """The user-curated list of files"""
        return self.__file_list

    def set_file_list(self, file_list):
        """Set the file list

        A caller can set the file list to the file list in some other
        workspace. This lets a single, sometimes very bulky file list be
        used without copying it to a measurements file.
        """
        if self.__file_list is not None:
            self.__file_list.remove_notification_callback(self.__on_file_list_changed)
        self.__file_list = file_list
        if file_list is not None:
            self.__file_list.add_notification_callback(self.__on_file_list_changed)

    file_list = property(get_file_list)

    def get_grid(self, grid_name):
        """Return a grid with the given name"""
        if grid_name not in self.__grid:
            raise ValueError("Could not find grid %s" % grid_name)
        return self.__grid[grid_name]

    def set_grids(self, last=None):
        """Initialize the grids for an image set

        last - none if first in image set or the return value from
               this method.
        returns a grid dictionary
        """
        if last is None:
            last = {}
        self.__grid = last
        return self.__grid

    def set_grid(self, grid_name, grid_info):
        """Add a grid to the workspace"""
        self.__grid[grid_name] = grid_info

    def get_frame(self):
        """The frame is CellProfiler's gui window

        If the frame is present, a module should do its display
        """
        if self.__do_show:
            return self.__frame
        return None

    frame = property(get_frame)

    def show_frame(self, do_show):
        self.__do_show = do_show

    def get_display(self):
        """True to provide a gui display"""
        return self.__frame is not None

    display = property(get_display)

    def get_in_background(self):
        return self.__in_background

    def set_in_background(self, val):
        self.__in_background = val

    in_background = property(get_in_background, set_in_background)

    def get_outline_names(self):
        """The names of outlines of objects"""
        return list(self.__outlines.keys())

    def add_outline(self, name, outline):
        """Add an object outline to the workspace"""
        self.__outlines[name] = outline

    def get_outline(self, name):
        """Get a named outline"""
        return self.__outlines[name]

    def get_module(self):
        """Get the module currently being run"""
        return self.__module

    module = property(get_module)

    def set_module(self, module):
        """Set the module currently being run"""
        self.__module = module

    def interaction_request(self, module, *args, **kwargs):
        """make a request for GUI interaction via a pipeline event

        module - target module for interaction request

        headless_ok - True if the interaction request can be made in
                      a headless context. An example is synchronized access to
                      a shared resource which must be coordinated among all
                      workers.
        """
        # See also:
        # main().interaction_handler() in __init__.py
        # PipelineController.module_interaction_request() in pipelinecontroller.py
        import cellprofiler_core.preferences

        if "headless_ok" in kwargs:
            tmp = kwargs.copy()
            del tmp["headless_ok"]
            headless_ok = kwargs["headless_ok"]
            kwargs = tmp
        else:
            headless_ok = False
        if self.interaction_handler is None:
            if cellprofiler_core.preferences.get_headless() and not headless_ok:
                raise self.NoInteractionException()
            else:
                return module.handle_interaction(*args, **kwargs)
        else:
            return self.interaction_handler(module, *args, **kwargs)

    def cancel_request(self):
        """Make a request to cancel an ongoing analysis"""
        if self.cancel_handler is None:
            raise self.NoInteractionException()
        self.cancel_handler()

    def post_group_display(self, module):
        """Perform whatever post-group module display is necessary

        module - module being run
        """
        if self.post_group_display_handler is not None:
            self.post_group_display_handler(
                module, self.display_data, self.measurements.image_set_number
            )
        elif self.frame is not None:
            figure = self.get_module_figure(
                module, self.measurements.image_set_number, self.frame
            )
            module.display_post_group(self, figure)

    def post_run_display(self, module):
        """Perform whatever post-run module display is necessary

        module - module being run
        """
        if self.post_run_display_handler is not None:
            self.post_run_display_handler(self, module)
        elif self.frame is not None:
            figure = self.get_module_figure(
                module, self.measurements.image_set_count + 1, self.frame
            )
            module.display_post_run(self, figure)

    @property
    def is_last_image_set(self):
        return self.measurements.image_set_number == self.image_set_list.count() - 1

    def get_disposition(self):
        """How to proceed with the pipeline

        One of the following values:
        DISPOSITION_CONTINUE - continue to execute the pipeline
        DISPOSITION_PAUSE - wait until the status changes before executing
                            the next module
        DISPOSITION_CANCEL - stop running the pipeline
        DISPOSITION_SKIP - skip the rest of this image set
        """
        return self.__disposition

    def set_disposition(self, disposition):
        self.__disposition = disposition
        event = DispositionChangedEvent(disposition)
        for listener in self.__disposition_listeners:
            listener(event)

    disposition = property(get_disposition, set_disposition)

    def add_disposition_listener(self, listener):
        self.__disposition_listeners.append(listener)

    class NoInteractionException(Exception):
        pass

    def load(self, filename, load_pipeline):
        """Load a workspace from a .cpi file

        filename - path to file to load

        load_pipeline - true to load the pipeline from the file, false to
                        use the current pipeline.
        """
        import shutil
        from ..constants.pipeline import M_DEFAULT_OUTPUT_FOLDER
        from ..constants.pipeline import M_DEFAULT_INPUT_FOLDER
        from ..constants.pipeline import M_PIPELINE
        import cellprofiler_core.measurement
        from cellprofiler_core.preferences import (
            set_default_image_directory,
            set_default_output_directory,
        )

        image_set_and_measurements_are_same = False
        if self.__measurements is not None:
            image_set_and_measurements_are_same = id(self.__measurements) == id(
                self.__image_set
            )
            self.close()
        self.__loading = True
        try:
            #
            # Copy the file to a temporary location before opening
            #
            (
                fd,
                self.__filename,
            ) = cellprofiler_core.utilities.measurement.make_temporary_file()
            os.close(fd)

            shutil.copyfile(filename, self.__filename)

            self.__measurements = cellprofiler_core.measurement.Measurements(
                filename=self.__filename, mode="r+"
            )
            if self.__file_list is not None:
                self.__file_list.remove_notification_callback(
                    self.__on_file_list_changed
                )
            self.__file_list = HDF5FileList(self.measurements.hdf5_dict.hdf5_file)
            self.__file_list.add_notification_callback(self.__on_file_list_changed)
            if load_pipeline and self.__measurements.has_feature(
                EXPERIMENT, M_PIPELINE
            ):
                pipeline_txt = self.__measurements.get_experiment_measurement(
                    M_PIPELINE
                )
                # CP 3.1.8 cpproj (and possibly before) saved info in bytes; converting to python 3 string
                if type(pipeline_txt) == bytes:
                    pipeline_txt = (
                        pipeline_txt.decode("unicode_escape")
                        .encode("utf-8")
                        .replace(b"\\x00", b"")
                        .decode("unicode_escape")
                        .replace("ÿþ", "")
                    )
                if "\\n" in pipeline_txt:
                    # Loaded pipeline text from a pre-h5py 3 hdf5 will have escaped characters in need of decoding.
                    try:
                        pipeline_txt = pipeline_txt.encode(
                            "latin-1", "backslashreplace"
                        ).decode("unicode-escape")
                    except Exception as e:
                        print(
                            f"Unable to fully decode pipeline, you may encounter some errors. Issue was: {e}"
                        )
                self.pipeline.load(io.StringIO(pipeline_txt))
            elif load_pipeline:
                self.pipeline.clear()
            else:
                fd = io.StringIO()
                self.pipeline.dump(fd, save_image_plane_details=False)
                self.__measurements.add_experiment_measurement(
                    M_PIPELINE, fd.getvalue()
                )

            for feature, function in (
                (M_DEFAULT_INPUT_FOLDER, set_default_image_directory),
                (M_DEFAULT_OUTPUT_FOLDER, set_default_output_directory),
            ):
                if self.measurements.has_feature(EXPERIMENT, feature):
                    path = self.measurements[EXPERIMENT, feature]
                    if os.path.isdir(path):
                        function(path)
            if image_set_and_measurements_are_same:
                self.__image_set = self.__measurements

        finally:
            self.__loading = False
        self.notify(self.WorkspaceLoadedEvent(self))

    def create(self):
        """Create a new workspace file

        filename - name of the workspace file
        """
        from ..utilities.measurement import make_temporary_file
        from cellprofiler_core.measurement import Measurements

        if isinstance(self.measurements, Measurements):
            self.close()

        fd, self.__filename = make_temporary_file()
        self.__measurements = Measurements(filename=self.__filename, mode="w")
        os.close(fd)
        if self.__file_list is not None:
            self.__file_list.remove_notification_callback(self.__on_file_list_changed)
        self.__file_list = HDF5FileList(self.measurements.hdf5_dict.hdf5_file)
        self.__file_list.add_notification_callback(self.__on_file_list_changed)
        self.notify(self.WorkspaceCreatedEvent(self))

    def save(self, path):
        """Save the current workspace to the given path

        path - path to file to save

        Note: "saving" means copying the temporary workspace file
        """
        self.save_default_folders_to_measurements()
        self.measurements.flush()
        #
        # Note: shutil.copy and similar don't seem to work under Windows.
        #       I suspect that there's some file mapping magic that's
        #       causing problems because I found some internet postings
        #       where people tried to copy database files and failed.
        #       If you're thinking, "He didn't close the file", I did.
        #       shutil.copy creates a truncated file if you use it.
        #
        hdf5src = self.measurements.hdf5_dict.hdf5_file
        hdf5dest = h5py.File(path, mode="w")
        for key in hdf5src:
            obj = hdf5src[key]
            if isinstance(obj, h5py.Dataset):
                hdf5dest[key] = obj[()]
            else:
                hdf5src.copy(hdf5src[key], hdf5dest, key)
        for key in hdf5src.attrs:
            hdf5dest.attrs[key] = hdf5src.attrs[key]
        hdf5dest.close()

    def close(self):
        """Close the workspace and delete the temporary measurements file"""
        if self.measurements is not None and self.__filename is not None:
            self.measurements.close()
            os.unlink(self.__filename)

    def save_pipeline_to_measurements(self):
        from ..constants.pipeline import M_PIPELINE

        fd = io.StringIO()
        self.pipeline.dump(fd, save_image_plane_details=False)
        self.measurements.add_experiment_measurement(M_PIPELINE, fd.getvalue())
        self.measurements.flush()

    def save_default_folders_to_measurements(self):
        from ..constants.pipeline import M_DEFAULT_INPUT_FOLDER
        from ..constants.pipeline import M_DEFAULT_OUTPUT_FOLDER
        from cellprofiler_core.preferences import get_default_image_directory
        from cellprofiler_core.preferences import get_default_output_directory

        self.measurements.add_experiment_measurement(
            M_DEFAULT_INPUT_FOLDER, get_default_image_directory()
        )
        self.measurements.add_experiment_measurement(
            M_DEFAULT_OUTPUT_FOLDER, get_default_output_directory()
        )

    def invalidate_image_set(self):
        if not self.__loading:
            self.measurements.clear()
            self.save_pipeline_to_measurements()

    def refresh_image_set(self, force=False):
        """Refresh the image set if not present

        force - force a rewrite, even if the image set is cached

        This method executes Pipeline.prepare_run in order on self in order
        to write the image set image measurements to our internal
        measurements. If image set measurements are present, then we
        assume that the cache reflects pipeline + file list unless "force"
        is true.
        """
        if len(self.measurements.get_image_numbers()) == 0 or force:
            self.measurements.clear()
            self.save_pipeline_to_measurements()
            modules = self.pipeline.modules()
            stop_module = None
            if len(modules) > 1 and not modules[-1].is_load_module():
                for module, next_module in zip(
                    self.pipeline.modules()[:-1], self.pipeline.modules()[1:]
                ):
                    if module.is_load_module():
                        stop_module = next_module

            # TODO: Get rid of image_set_list
            no_image_set_list = self.image_set_list is None
            if no_image_set_list:
                from cellprofiler_core.image import ImageSetList

                self.__image_set_list = ImageSetList()
            try:
                result = self.pipeline.prepare_run(self, stop_module)
                return result
            except:
                LOGGER.error("Failed during prepare_run", exc_info=1)
                return False
            finally:
                if no_image_set_list:
                    self.__image_set_list = None
        return True

    def add_notification_callback(self, callback):
        """Add a callback that will be called on a workspace event

        Workspace events are load events, and file list events.

        callback - a function to be called when an event occurs. The signature
        is: callback(event)
        """
        self.__notification_callbacks.append(callback)

    def remove_notification_callback(self, callback):
        self.__notification_callbacks.remove(callback)

    def notify(self, event):
        for callback in self.__notification_callbacks:
            try:
                callback(event)
            except:
                LOGGER.error("Notification callback threw an exception", exc_info=1)

    def __on_file_list_changed(self):
        self.notify(self.WorkspaceFileListNotification(self))

    class WorkspaceEvent(object):
        """The base for any event sent to a workspace callback via Workspace.notify

        """

        def __init__(self, workspace):
            self.workspace = workspace

    class WorkspaceLoadedEvent(WorkspaceEvent):
        """Indicates that a workspace has been loaded

        When a workspace loads, the file list changes.
        """

        def __init__(self, workspace):
            super(self.__class__, self).__init__(workspace)

    class WorkspaceCreatedEvent(WorkspaceEvent):
        """Indicates that a blank workspace has been created"""

        def __init__(self, workspace):
            super(self.__class__, self).__init__(workspace)

    class WorkspaceFileListNotification(WorkspaceEvent):
        """Indicates that the workspace's file list changed"""

        def __init__(self, workspace):
            super(self.__class__, self).__init__(workspace)
