import logging
import sys
import uuid

import docutils.core
import numpy

from ..constants.pipeline import BATCH_STATE
from ..constants.pipeline import MODULE_NAMES
from ..constants.pipeline import MODULE_NOTES
from ..constants.pipeline import MODULE_REVISION_NUMBERS
from ..constants.pipeline import NUMBERS_OF_VARIABLES
from ..constants.pipeline import SETTINGS
from ..constants.pipeline import SHOW_WINDOW
from ..constants.pipeline import VARIABLE_INFO_TYPES
from ..constants.pipeline import VARIABLE_REVISION_NUMBERS
from ..constants.pipeline import VARIABLE_VALUES
from ..setting import HiddenCount
from ..setting import Setting
from ..setting import ValidationError


LOGGER = logging.getLogger(__name__)

class Module:
    """ Derive from the abstract module class to create your own module in Python

    You need to implement the following in the derived class:
    create_settings - create the settings that configure the module.
    settings - return the settings that will be loaded or saved from/to the
               pipeline.
    run - to run the module, producing measurements, etc.

    These methods are optional:
    prepare_settings - adjust the internal state of a module to accept a set of
               stored settings, (e.g., to create the right number of image
               entries in the module's settings.)
    visible_settings - return the settings that will be displayed on the UI
               (default is to use the output of settings())
    upgrade_settings - rewrite a group of settings from a previous version to
               be compatible with the latest version of a module.

    Implement these if you produce measurements:
    get_categories - The category of measurement produced, for instance AreaShape
    get_measurements - The measurements produced by a category
    get_measurement_images - The images measured for a particular measurement
    get_measurement_scales - the scale at which a measurement was taken
    get_measurement_columns - the measurements stored in the database

    The pipeline calls hooks in the module before and after runs and groups.
    The hooks are:
    prepare_run - before run: useful for setting up image sets
    prepare_group - before group: useful for initializing aggregation
    post_group - after group: useful for calculating final aggregation steps
               and for writing out results.
    post_run - use this to perform operations on the results of the experiment,
               for instance on all measurements
    post_pipeline_load - use this to update any settings that require the pipeline
                to be available before they can be adjusted.

    If your module requires state across image_sets, think of storing
    information in the module shared_state dictionary (fetched by
    get_dictionary()).
    """

    def __init__(self):
        if self.__doc__ is None:
            self.__doc__ = sys.modules[self.__module__].__doc__

        self.function = None
        self.__module_num = -1
        self.__settings = []
        self.__notes = []
        self.__show_window = False
        self.__wants_pause = False
        self.__svn_version = "Unknown"
        self.__enabled = True
        self.__variable_revision_number = 0
        self.shared_state = (
            {}
        )  # used for maintaining state between modules, see get_dictionary()
        self.id = uuid.uuid4()
        self.batch_state = numpy.zeros((0,), numpy.uint8)
        # A subclass can override bellow two either by declaring a module_name
        # attribute in the class definition or by assigning to it in
        # the create_settings method.
        # used for citation generation
        if not hasattr(self, "doi"):
            self.doi = []
        # Set the name of the module based on the class name.
        if not hasattr(self, "module_name"):
            self.module_name = self.__class__.__name__
        self.create_settings()

    def to_dict(self) -> dict:
        return {"module_num": self.module_num,
                "notes": self.notes,
                "show_window": self.show_window,
                "wants_pause": self.wants_pause,
                "svn_version": self.svn_version,
                "enabled": self.enabled,
                "variable_revision_number": self.variable_revision_number,
                "batch_state": repr(self.batch_state),
                "module_name": self.__class__.__qualname__,
                "module_path": ".".join([self.__module__, self.__class__.__qualname__])
                }

    def from_dict(self, settings: list, attributes: dict):
        self.module_num = attributes["module_num"]
        self.show_window = attributes["show_window"]
        self.wants_pause = attributes["wants_pause"]
        self.svn_version = attributes["svn_version"]
        self.enabled = attributes["enabled"]
        self.module_path = attributes["module_path"]
        self.module_name = attributes["module_name"]
        setting_values = [setting["value"] for setting in settings]
        if attributes["variable_revision_number"] > self.variable_revision_number:
            LOGGER.warning(f"Loaded module '{self.module_name}' is from a newer version of CellProfiler - "
                            f"v{attributes['variable_revision_number']}, current version is "
                            f"v{self.variable_revision_number}. Settings may load incorrectly.")
        self.set_settings_from_values(
            setting_values, attributes['variable_revision_number'], self.module_path
        )

    def __setattr__(self, slot, value):
        if hasattr(self, slot) and isinstance(getattr(self, slot), Setting):
            assert isinstance(value, Setting), (
                    "Overwriting %s's %s existing Setting with value of type %s.\nUse __dict__['%s'] = ... to override."
                    % (self.module_name, slot, type(value), slot)
            )
        object.__setattr__(self, slot, value)

    def create_settings(self):
        """Create your settings by subclassing this function

        create_settings is called at the end of initialization.

        You should create the setting variables for your module here:
            # Ask the user for the input image
            self.image_name = cellprofiler_core.settings.ImageNameSubscriber(...)
            # Ask the user for the name of the output image
            self.output_image = cellprofiler_core.settings.ImageNameProvider(...)
            # Ask the user for a parameter
            self.smoothing_size = cellprofiler_core.settings.Float(...)
        """
        pass

    def create_from_handles(self, handles, module_num):
        """Fill a module with the information stored in the handles structure for module # ModuleNum

        Returns a module with the settings decanted from the handles.
        If the revision is old, a different and compatible module can be returned.
        """
        self.__module_num = module_num
        idx = module_num - 1
        settings = handles[SETTINGS][0, 0]
        setting_values = []
        self.__notes = []
        if (
                MODULE_NOTES in settings.dtype.fields
                and settings[MODULE_NOTES].shape[1] > idx
        ):
            n = settings[MODULE_NOTES][0, idx].flatten()
            for x in n:
                if isinstance(x, numpy.ndarray):
                    if len(x) == 0:
                        x = ""
                    else:
                        x = x[0]
                self.__notes.append(x)
        if SHOW_WINDOW in settings.dtype.fields:
            self.__show_window = settings[SHOW_WINDOW][0, idx] != 0
        if BATCH_STATE in settings.dtype.fields:
            # convert from uint8 to array of one string to avoid long
            # arrays, which get truncated by numpy repr()
            self.batch_state = numpy.array(settings[BATCH_STATE][0, idx].tostring())
        setting_count = settings[NUMBERS_OF_VARIABLES][0, idx]
        variable_revision_number = settings[VARIABLE_REVISION_NUMBERS][0, idx]
        module_name = settings[MODULE_NAMES][0, idx][0]
        for i in range(0, setting_count):
            value_cell = settings[VARIABLE_VALUES][idx, i]
            if isinstance(value_cell, numpy.ndarray):
                if numpy.product(value_cell.shape) == 0:
                    setting_values.append("")
                else:
                    setting_values.append(str(value_cell[0]))
            else:
                setting_values.append(value_cell)
        self.set_settings_from_values(
            setting_values, variable_revision_number, module_name
        )

    def prepare_settings(self, setting_values):
        """Do any sort of adjustment to the settings required for the given values

        setting_values - the values for the settings just prior to mapping
                         as done by set_settings_from_values
        This method allows a module to specialize itself according to
        the number of settings and their value. For instance, a module that
        takes a variable number of images or objects can increase or decrease
        the number of relevant settings so they map correctly to the values.

        See cellprofiler_core.modules.measureobjectareashape for an example.
        """
        pass

    def set_settings_from_values(
            self, setting_values, variable_revision_number, module_name
    ):
        """Set the settings in a module, given a list of values

        The default implementation gets all the settings and then
        sets their values using the string passed. A more modern
        module may want to tailor the particular settings set to
        whatever values are in the list or however many values
        are in the list.
        """
        setting_values, variable_revision_number = self.upgrade_settings(
            setting_values, int(variable_revision_number), module_name
        )
        self.prepare_settings(setting_values)
        for v, value in zip(self.settings(), setting_values):
            v.value = value

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust setting values if they came from a previous revision

        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module

        Overriding modules should return a tuple of setting_values,
        variable_revision_number and True if upgraded to CP 2.0, otherwise
        they should leave things as-is so that the caller can report
        an error.
        """
        return setting_values, variable_revision_number

    def post_pipeline_load(self, pipeline):
        """This is a convenient place to do things to your module after the
           settings have been loaded or initialized"""
        pass

    # https://wiki.python.org/moin/reStructuredText
    @staticmethod
    def _rst_to_html_fragment(source):
        if not isinstance(source, str):
            return ""
        parts = docutils.core.publish_parts(source=source, writer_name="html")

        return parts["body_pre_docinfo"] + parts["fragment"]

    def _get_setting_help(self, setting):
        if isinstance(setting, HiddenCount):
            return ""

        return """\
<div>
    <h4>{SETTING_NAME}</h4>
    <p>{SETTING_DOC}</p>
</div>
        """.format(
            **{
                "SETTING_DOC": self._rst_to_html_fragment(setting.doc),
                "SETTING_NAME": self._rst_to_html_fragment(setting.text),
            }
        )

    def get_help_text(self):
        """Return help text for a module as unformatted strings
        """
        settings_list = self.help_settings()
        if len(settings_list):
            settings_doc = [(setting.text, setting.doc) for setting in settings_list]
        else:
            settings_doc = ""
        mod_doc = self.__doc__ if self.__doc__ else ""
        return mod_doc, settings_doc

    def get_help(self):
        """Return help text for the module

        The default help is taken from your modules docstring and from
        the settings.
        """
        settings_help = (
            """\
<div>
    <h2>Settings:</h2>
    {SETTINGS_DOC}
</div>
""".format(
                **{
                    "SETTINGS_DOC": "\n".join(
                        [
                            self._get_setting_help(setting)
                            for setting in self.help_settings()
                        ]
                    )
                }
            )
            if len(self.help_settings())
            else ""
        )

        return """\
<html style="font-family:arial">
    <body>
        <div>
            {MODULE_DOC}
        </div>
        {SETTINGS_HELP}
    </body>
</html>
""".format(
            **{
                "MODULE_DOC": self._rst_to_html_fragment(self.__doc__),
                "SETTINGS_HELP": settings_help,
                "TITLE": self.module_name,
            }
        )

    def save_to_handles(self, handles):
        from ..setting.text import Name
        from ..setting.subscriber import Subscriber

        module_idx = self.module_num - 1
        setting = handles[SETTINGS][0, 0]
        setting[MODULE_NAMES][0, module_idx] = str(self.module_class())
        setting[MODULE_NOTES][0, module_idx] = numpy.ndarray(
            shape=(len(self.notes), 1), dtype="object"
        )
        for i in range(0, len(self.notes)):
            setting[MODULE_NOTES][0, module_idx][i, 0] = self.notes[i]
        setting[NUMBERS_OF_VARIABLES][0, module_idx] = len(self.settings())
        for i in range(0, len(self.settings())):
            variable = self.settings()[i]
            if len(str(variable)) > 0:
                setting[VARIABLE_VALUES][module_idx, i] = variable.get_unicode_value()
            if isinstance(variable, Name):
                setting[VARIABLE_INFO_TYPES][module_idx, i] = str(
                    "%s indep" % variable.group
                )
            elif isinstance(variable, Subscriber):
                setting[VARIABLE_INFO_TYPES][module_idx, i] = str(variable.group)
        setting[VARIABLE_REVISION_NUMBERS][
            0, module_idx
        ] = self.variable_revision_number
        setting[MODULE_REVISION_NUMBERS][0, module_idx] = 0
        setting[SHOW_WINDOW][0, module_idx] = 1 if self.show_window else 0
        # convert from single-element array with a long string to an
        # array of uint8, to avoid string encoding isues in .MAT
        # format.
        setting[BATCH_STATE][0, module_idx] = numpy.fromstring(
            self.batch_state.tostring(), numpy.uint8
        )

    @staticmethod
    def in_batch_mode():
        """Return True if the module knows that the pipeline is in batch mode"""
        return None

    def change_causes_prepare_run(self, setting):
        """Check to see if changing the given setting means you have to restart

        Some settings, esp in modules like LoadImages, affect more than
        the current image set when changed. For instance, if you change
        the name specification for files, you have to reload your image_set_list.
        Override this and return True if changing the given setting means
        that you'll have to call "prepare_run".
        """
        return False

    def turn_off_batch_mode(self):
        """Reset the module to an editable state if batch mode is on

        A module is allowed to create hidden information that it uses
        to turn batch mode on or to save state to be used in batch mode.
        This call signals that the pipeline has been opened for editing,
        even if it is a batch pipeline; all modules should be restored
        to a state that's appropriate for creating a batch file, not
        for running a batch file
        """
        pass

    def test_valid(self, pipeline):
        """Test to see if the module is in a valid state to run

        Throw a ValidationError exception with an explanation if a module is not valid.
        """
        try:
            for setting in self.visible_settings():
                setting.test_valid(pipeline)
            self.validate_module(pipeline)
        except ValidationError as ve:
            raise ve
        except Exception as e:
            raise ValidationError(
                "Exception in cpmodule.test_valid %s" % e, self.visible_settings()[0]
            )

    def test_module_warnings(self, pipeline):
        """Test to see if there are any troublesome setting values in the module

        Throw a ValidationError exception with an explanation if a module
        is likely to be misconfigured. An example is if ExportToDatabase is
        not the last module.
        """
        try:
            for setting in self.visible_settings():
                setting.test_setting_warnings(pipeline)
            self.validate_module_warnings(pipeline)
        except ValidationError as instance:
            raise instance
        except Exception as e:
            raise ValidationError(
                "Exception in cpmodule.test_valid %s" % e, self.visible_settings()[0]
            )

    def validate_module(self, pipeline):
        """Implement this to validate module settings

        Module implementers should implement validate_module to
        further validate a module's settings. For instance, load_data
        checks the .csv file that it uses in validate_module to ensure
        that the user has chosen a valid .csv file.

        Throw a cps.ValidationError, selecting the most egregiously offending
        setting to indicate failure.
        """
        pass

    def validate_module_warnings(self, pipeline):
        """Implement this to flag potentially dangerous settings

        Module implementers should implement validate_module_warnings to
        find setting combinations that can cause unexpected results.
        Implementers should throw a cps.ValidationError, selecting the
        most egregiously offending setting to indicate failure.
        """
        pass

    def other_providers(self, group):
        """Return a list of hidden name/object/etc. providers supplied by the module for this group

        group - a group supported by a subclass of NameProvider

        This routine returns additional providers beyond those that
        are listed by the module's visible_settings.
        """
        return []

    def get_module_num(self):
        """Get the module's index number

        The module's index number or ModuleNum is a one-based index of its
        execution position in the pipeline. It can be used to predict what
        modules have been run (creating whatever images and measurements
        those modules create) previous to a given module.
        """
        return self.__module_num

    def set_module_num(self, module_num):
        """Change the module's one-based index number in the pipeline

        """
        self.__module_num = module_num

    module_num = property(get_module_num, set_module_num)

    def module_class(self):
        """The class to instantiate, except for the special case of matlab modules.

        """
        return self.__module__ + "." + self.module_name

    def get_enabled(self):
        """True if the module should be executed, False if it should be ignored.

        """
        return self.__enabled

    def set_enabled(self, enable):
        self.__enabled = enable

    enabled = property(get_enabled, set_enabled)

    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline

        These are the settings (from cellprofiler_core.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return self.__settings

    def help_settings(self):
        """Override this if you want the settings for help to be in a different order"""
        return self.settings()

    def setting(self, setting_num):
        """Reference a setting by its one-based setting number
        """
        return self.settings()[setting_num - 1]

    def set_settings(self, settings):
        self.__settings = settings

    def visible_settings(self):
        """The settings that are visible in the UI
        """
        return self.settings()

    def get_show_window(self):
        """True if the user wants to see the figure for this module"""
        return self.__show_window

    def set_show_window(self, show_window):
        self.__show_window = show_window

    show_window = property(get_show_window, set_show_window)

    def get_wants_pause(self):
        """True if the user wants to pause at this module while debugging"""
        return self.__wants_pause

    def set_wants_pause(self, wants_pause):
        self.__wants_pause = wants_pause

    wants_pause = property(get_wants_pause, set_wants_pause)

    def get_notes(self):
        """The user-entered notes for a module
        """
        return self.__notes

    def set_notes(self, notes):
        """Give the module new user-entered notes

        """
        if isinstance(notes, str):
            notes = [notes]
        sanitization_dict = {"“": "\"", "”": "\""}
        self.__notes = [''.join(sanitization_dict.get(x,x) for x in note) for note in notes]

    notes = property(get_notes, set_notes)

    def get_svn_version(self):
        return self.__svn_version

    def set_svn_version(self, version):
        self.__svn_version = version

    svn_version = property(get_svn_version, set_svn_version)

    def write_to_handles(self, handles):
        """Write out the module's state to the handles

        """
        pass

    def write_to_text(self, file):
        """Write the module's state, informally, to a text file
        """
        pass

    def prepare_run(self, workspace):
        """Prepare the image set list for a run (& whatever else you want to do)

        workspace - holds the following crucial structures:

            pipeline - the pipeline being run

            module - this module

            measurements - measurements structure that can be populated with
                          a image set file names and metadata.

            image_set_list - add any image sets to the image set list

            frame - parent frame of application if GUI enabled, None if GUI
                    disabled

        return True if operation completed, False if aborted
        """
        return True

    def is_load_module(self):
        """If true, the module will load files and make image sets"""
        return False

    @classmethod
    def is_input_module(cls):
        """If true, the module is one of the input modules

        The input modules are "Images", "Metadata", "NamesAndTypes" and "Groups"
        """
        return False

    @staticmethod
    def is_create_batch_module():
        """If true, the module will pickle the pipeline into a batch file and exit

        This is needed by modules which can't properly operate in a batch
        mode (e.g., do all their work post_run or don't work so well if
        run in parallel)
        """
        return False

    @staticmethod
    def is_aggregation_module():
        """If true, the module uses data from other imagesets in a group

        Aggregation modules perform operations that require access to
        all image sets in a group, generally resulting in an aggregation
        operation during the last image set or in post_group. Examples are
        TrackObjects, MakeProjection and CorrectIllumination_Calculate.
        """
        return False

    def convert(self, pipeline, metadata, namesandtypes, groups):
        """Convert the input processing of this module from the legacy format

        Legacy modules like LoadImages should copy their settings into
        the Metadata, NamesAndTypes and Groups modules when this call is made.

        pipeline - the pipeline being converted

        metadata - the pipeline's Metadata module

        namesandtypes - the pipeline's NamesAndTypes module

        groups - the pipeline's Groups module

        """
        pass

    @staticmethod
    def is_object_identification_module():
        """If true, the module will identify primary, secondary or tertiary objects"""
        return False

    def run(self, workspace):
        """Run the module (abstract method)

        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created.
                           None means don't draw.
            display_data - the run() module should store anything to be
                           displayed in this attribute, which will be used in
                           display()

        run() should not attempt to display any data, but should communicate it
        to display() via the workspace.
        """
        pass

    def post_run(self, workspace):
        """Do post-processing after the run completes

        workspace - the workspace at the end of the run
        """
        pass

    def display(self, workspace, figure):
        """Display the results, and possibly intermediate results, as
        appropriate for this module.  This method will be called after
        run() is finished if self.show_window is True.

        The run() method should store whatever data display() needs in
        workspace.display_data.  The module is given a CPFigure to use for
        display in the third argument.
        """
        figure.Close()  # modules that don't override display() shouldn't
        # display anything

    def display_post_group(self, workspace, figure):
        """Display the results of work done post-group

        This method is only called if self.show_window is True

        workspace - the current workspace. workspace.display_data should have
                    whatever information is needed for the display. Numpy arrays
                    lists, tuples, dictionaries and Python builtin objects are
                    allowed.

        figure - the figure to use for the display.
        """
        pass

    def display_post_run(self, workspace, figure):
        """Display results after post_run completes

        workspace - a workspace with pipeline, module and measurements valid

        figure - display results in this CPFigure
        """
        pass

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        """Prepare to create a batch file

        This function is called when CellProfiler is about to create a
        file for batch processing. It gives a module an opportunity to
        change its settings and measurements to adapt to file mount differences
        between the machine that created the pipeline and the machines that
        will run the pipeline. You should implement prepare_to_create_batch
        if your module stores paths in settings or measurements. You should
        call fn_alter_path(path) to update any paths to those of the target
        machine.

        workspace - the workspace including the pipeline, the image_set_list
                    and the measurements that need to be modified.

        fn_alter_path - this is a function that takes a pathname on the local
                        host and returns a pathname on the remote host. It
                        handles issues such as replacing backslashes and
                        mapping mountpoints. It should be called for every
                        pathname stored in the settings or legacy fields.

        Returns True if it succeeds.
        """
        return True

    def get_groupings(self, workspace):
        """Return the image groupings of the image sets in an image set list

        get_groupings is called after prepare_run

        workspace - a workspace with an image_set_list and measurements
                    as prepared by prepare_run.

        returns a tuple of key_names and group_list:
        key_names - the names of the keys that identify the groupings
        group_list - a sequence composed of two-tuples.
                     the first element of the tuple is a dictionary giving
                     the metadata values for the metadata keys
                     the second element of the tuple is a sequence of
                     image numbers comprising the image sets of the group
        For instance, an experiment might have key_names of 'Metadata_Row'
        and 'Metadata_Column' and a group_list of:
        [ ({'Metadata_Row':'A','Metadata_Column':'01'}, [1,97,193]),
          ({'Metadata_Row':'A','Metadata_Column':'02'), [2,98,194]),... ]

        Returns None to indicate that the module does not contribute any
        groupings.
        """
        return None

    def prepare_group(self, workspace, grouping, image_numbers):
        """Prepare to start processing a new grouping

        workspace - the workspace for the group. The pipeline, measurements
                    and image_set_list are valid at this point and you can
                    fill in image_sets at this point.
        grouping - a dictionary that describes the key for the grouping.
                   For instance, { 'Metadata_Row':'A','Metadata_Column':'01'}
        image_numbers - a sequence of the image numbers within the
                   group (image sets can be retreved as
                   image_set_list.get_image_set(image_numbers[i]-1)

        prepare_group is called once after prepare_run if there are no
        groups.
        """
        pass

    def post_group(self, workspace, grouping):
        """Do post-processing after a group completes

        workspace - the workspace at the end of the group
        grouping - the group that's being run
        """
        pass

    def get_measurement_columns(self, pipeline):
        """Return a sequence describing the measurement columns needed by this module

        This call should return one element per image or object measurement
        made by the module during image set analysis. The element itself
        is a 3-tuple:
        first entry: either one of the predefined measurement categories,
                     {"Image", "Experiment" or "Neighbors" or the name of one
                     of the objects.}
        second entry: the measurement name (as would be used in a call
                      to add_measurement)
        third entry: the column data type (for instance, "varchar(255)" or
                     "float")
        """
        return []

    @staticmethod
    def get_object_relationships(pipeline):
        """Return a sequence describing the relationships recorded in measurements

        This method reports the relationships recorded in the measurements
        using add_relate_measurement. Modules that add relationships should
        return one 4-tuple of
        (<relationship-name>, <object-name-1>, <object-name-2>, <when>)
        for every combination of relationship and parent / child objects
        that will be produced during the course of a run.

        <when> is one of cpmeas.MCA_AVAILABILE_EACH_CYCLE or
        cpmeas.MCA_AVAILABLE_POST_GROUP. cpmeas.MCA_AVAILABLE_EACH_CYCLE
        promises that the relationships will be available after each cycle.
        Any relationship with that cycle's image number (as either the
        parent or child) will be inserted if not already present in the database.

        MCA_AVAILABLE_POST_GROUP indicates that the relationship is not available
        until the group has completed - all relationships with a group's
        image number will be written in that case.
        """
        return []

    def get_dictionary(self, ignore=None):
        """Get the dictionary for this module
        """
        return self.shared_state

    def get_dictionary_for_worker(self):
        """Get the dictionary that should be shared between analysis workers

        A module might use the dictionary for cacheing information stored on
        disk or that's difficult to compute. It might also use it to store
        aggregate data, but this data may not be useful to other workers.

        Finally, a module might store Python objects that aren't JSON serializable
        in its dictionary. In these cases, the module should create a dictionary
        that can be JSON serialized in get_dictionary_for_worker and then
        reconstruct the result of JSON deserialization in set_dictionary_in_worker.
        """
        return self.get_dictionary()

    def set_dictionary_for_worker(self, d):
        """Initialize this worker's dictionary using results from first worker

        see get_dictionary_for_worker for details.
        """
        self.get_dictionary().clear()
        self.get_dictionary().update(d)

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        return []

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        """Return a list of image names used as a basis for a particular measure
        """
        return []

    @staticmethod
    def get_measurement_objects(pipeline, object_name, category, measurement):
        """Return a list of secondary object names used as a basis for a particular measure

        object_name - either "Image" or the name of the primary object
        category - the category being measured, for instance "Threshold"
        measurement - the name of the measurement being done

        Some modules output image-wide aggregate measurements in addition to
        object measurements. These must be stored using the "Image" object name
        in order to save a single value. A module can override
        get_measurement_objects to tell the user about the object name in
        those situations.

        In addition, some modules may make use of two segmentations, for instance
        when measuring the total value of secondary objects related to primary
        ones. This mechanism can be used to identify the secondary objects used.
        """
        return []

    def get_measurement_scales(
            self, pipeline, object_name, category, measurement, image_name
    ):
        """Return a list of scales (eg for texture) at which a measurement was taken
        """
        return []

    def is_image_from_file(self, image_name):
        """Return True if this module loads this image name from a file."""
        from ..setting.subscriber import FileImageSubscriber

        for setting in self.settings():
            if (
                    isinstance(setting, FileImageSubscriber, )
                    and setting.value == image_name
            ):
                return True
        return False

    @staticmethod
    def should_stop_writing_measurements():
        """Returns True if measurements should not be taken after this module

        The ExportToDatabase and ExportToExcel modules expect that no
        measurements will be recorded in latter modules. This function
        returns False in the default, indicating that measurements should
        keep being made, but returns True for these modules, indicating
        that any subsequent modules will lose their measurements and should
        not write any.
        """
        return False

    def needs_default_image_folder(self, pipeline):
        """Returns True if the module needs the default image folder

        pipeline - pipeline being run

        Legacy modules might need the default image folder as does any module
        that uses the DirectoryPath setting.
        """
        from ..setting.text import Directory

        for setting in self.visible_settings():
            if isinstance(setting, Directory):
                return True
        return False

    def obfuscate(self):
        """Erase any sensitive information in a module's settings

        You should implement "obfuscate" to erase information like
        passwords or file names so that the pipeline can be uploaded
        for error reporting without revealing that information.
        """
        pass

    def on_activated(self, workspace):
        """Called when the module is activated in the GUI

        workspace - the workspace that's currently running

        on_activated is here to give modules the chance to modify other
        elements of the pipeline, such as the image plane details or image
        set list. You're allowed to modify these parts of the pipeline
        in the UI thread until on_deactivated is called.
        """
        pass

    def on_deactivated(self):
        """Called when the module is deactivated in the GUI

        This is the signal that the settings have been unhooked from the
        GUI and can't be used to edit the pipeline
        """
        pass

    def on_setting_changed(self, setting, pipeline):
        """Called when a setting has been changed in the GUI"""
        pass

    def volumetric(self):
        return False
