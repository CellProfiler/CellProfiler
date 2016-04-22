'''<b>ilastik</b> classifiy image pixels as belonging to different 
classes using the machine-learning tool, ilastik.
<hr>


IlastikPixelClassification performs per-pixel classification using the
<a href="http://www.ilastik.org/">ilastik</a> application.
Ilastik is now bundled with the CellProfiler distribution; it applies
supervised machine learning techniques to images to learn their features.
A user trains a classifier with Ilastik and then saves the classifier.
The user then uses the IlastikPixelClassification module to classify the pixels in an
image.

IlastikPixelClassification produces an "image" consisting of probabilities that
the pixel belongs to the chosen class; this image is similar to
an intensity image that would be produced by fluorescence imaging.
Provided that the classifier is sufficiently accurate, the image is
well-suited for input into one of the <b>Identify</b> modules for
object detection. More instructions on using the interface may be found
<a href="http://ilastik.org/index.php?cat=20_Documentation&page=03_Cellprofiler">here</a>.
Please note that you must use the same image format for classification
as for the initial learning phase.

Currently, ilastik is only available for Windows, and is accessible from
in the CellProfiler folder under the Start Menu. A 64-bit system is
recommended for running ilastik.
'''

import logging
import urllib

import cellprofiler.cpimage  as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
from cellprofiler.preferences import standardize_default_folder_names, \
    DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME, NO_FOLDER_NAME, \
    ABSOLUTE_FOLDER_NAME, IO_FOLDER_CHOICE_HELP_TEXT, \
    DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME, \
    URL_FOLDER_NAME

logger = logging.getLogger(__name__)
import numpy as np
import sys, os

# Import vigra
try:
    import vigra

    has_ilastik = True
    # TODO Version check
except ImportError, vigraImport:
    logger.warning("""vigra import: failed to import the vigra library. Please follow the instructions on
"http://hci.iwr.uni-heidelberg.de/vigra/" to install vigra""", exc_info=True)
    has_ilastik = False

# Import h5py
try:
    import h5py
except ImportError, h5pyImport:
    logger.warning("""h5py import: failed to import the h5py library.""",
                   exc_info=True)
    raise h5pyImport

# Import ilastik

old_stdout = sys.stdout
if has_ilastik:
    try:
        # TODO Version check
        sys.stdout = sys.stderr = open(os.devnull, "w")
        import ilastik_main
        import ilastik

        print ilastik.__file__
        from ilastik.workflows.pixelClassification import PixelClassificationWorkflow

        sys.stdout = old_stdout

    except ImportError, ilastikImport:
        sys.stdout = old_stdout
        logger.warning("""ilastik import: failed to import the ilastik. Please follow the instructions on
                          "http://www.ilastik.org" to install ilastik""", exc_info=True)
        has_ilastik = False

SI_PROBABILITY_MAP_COUNT = 3


class IlastikPixelClassification(cpm.CPModule):
    module_name = 'ilastik_pixel_classification'
    variable_revision_number = 2
    category = "Image Processing"

    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
                "Select the input image", cps.NONE)

        self.probability_maps = []

        self.probability_map_count = cps.HiddenCount(
                self.probability_maps, "Probability map count")

        self.add_probability_map(False)

        self.add_probability_button = cps.DoSomething(
                "Add another probability map", "Add", self.add_probability_map, doc="""
            Press the <i>Add</i> button to output another
            probability map image from the classifier. Ilastik can be trained
            to recognize any number of classes of pixels. You can generate
            probability maps for any or all of them simultaneously by adding
            more images.""")

        self.h5_directory = cps.DirectoryPath(
                "Ilastik project file location",
                dir_choices=[
                    DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_INPUT_FOLDER_NAME,
                    ABSOLUTE_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME,
                    DEFAULT_OUTPUT_SUBFOLDER_NAME, URL_FOLDER_NAME],
                allow_metadata=False, doc="""
                Select the folder containing the ilastik project file to be loaded.
            %(IO_FOLDER_CHOICE_HELP_TEXT)s""" % globals())

        def get_directory_fn():
            '''Get the directory for the CSV file name'''
            return self.h5_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.h5_directory.get_parts_from_path(path)
            self.h5_directory.join_parts(dir_choice, custom_path)

        self.classifier_file_name = cps.FilenameText(
                "Ilastik project file name",
                cps.NONE,
                doc="""This is the name of the ilastik project file.""",
                get_directory_fn=get_directory_fn,
                set_directory_fn=set_directory_fn,
                browse_msg="Choose ilastik project file",
                exts=[("ilastik project file (*.ilp)", "*.ilp"), ("All files (*.*)", "*.*")]
        )
        self.no_ilastik_msg = cps.HTMLText(
                "",
                content="""
            IlastikPixelClassification cannot run on this platform because
            the necessary libraries are not available. IlastikPixelClassification is
            supported on 64-bit versions of Windows Vista, Windows 7 and
            Windows 8 and on Linux.""", size=(-1, 50))

    def add_probability_map(self, can_remove=True):
        group = cps.SettingsGroup()
        group.can_remove = can_remove
        self.probability_maps.append(group)

        # The following settings are used for the combine option
        group.output_image = cps.ImageNameProvider(
                "Name the output probability map", "ProbabilityMap")

        group.class_sel = cps.Integer(
                "Select the class",
                0, 0, 42, doc=
                '''Select the class you want to use. The class number
                corresponds to the label-class in ilastik''')

        if can_remove:
            group.remover = cps.RemoveSettingButton(
                    "Remove probability map",
                    "Remove", self.probability_maps, group, doc="""
                Press the <i>Remove</i> button to remove the
                probability map image from the list of images produced by this
                module""")

    def settings(self):
        result = [self.image_name, self.h5_directory, self.classifier_file_name,
                  self.probability_map_count]
        for group in self.probability_maps:
            result += [group.output_image, group.class_sel]
        return result

    def visible_settings(self):
        if has_ilastik:
            result = [self.image_name]
            for group in self.probability_maps:
                result += [group.output_image, group.class_sel]
                if group.can_remove:
                    result += [group.remover]
            result += [self.add_probability_button, self.h5_directory,
                       self.classifier_file_name]
            return result
        else:
            return [self.no_ilastik_msg]

    def run(self, workspace):
        if not has_ilastik:
            raise ImportError("The Vigra and Ilastik packages are not available or installed on this platform")
        # get input image
        image = workspace.image_set.get_image(self.image_name.value, must_be_color=False)

        # recover raw image domain
        image_ = image.pixel_data
        if image.get_scale() is not None:
            image_ = image_ * image.get_scale()
        else:
            # Best guess for derived images
            image_ = image_ * 255.0
        #
        # Apply a rescaling that's done similarly in ilastik's dataImpex
        #
        image_max = np.max(image_)
        if (image_max > 255) and (image_max < 4096):
            image_ = image_ / 4095. * 255.0

        # Check if image_ has channels, if not add singelton dimension
        if len(image_.shape) == 2:
            image_.shape = image_.shape + (1,)

        probMaps = self._classify_with_ilastik(image_)

        workspace.display_data.source_image = image.pixel_data
        workspace.display_data.dest_images = []
        for group in self.probability_maps:
            # Produce output image and select the probability map
            probMap = probMaps[:, :, int(group.class_sel.value)]
            temp_image = cpi.Image(probMap, parent_image=image)
            workspace.image_set.add(group.output_image.value, temp_image)
            workspace.display_data.dest_images.append(probMap)

    def _classify_with_ilastik(self, image):

        args = ilastik_main.parser.parse_args([])
        args.headless = True
        args.project = os.path.join(
                self.h5_directory.get_absolute_path(),
                self.classifier_file_name.value).encode("utf-8")

        input_data = image
        print input_data.shape
        input_data = vigra.taggedView(input_data, 'yxc')

        shell = ilastik_main.main(args)
        assert isinstance(shell.workflow, PixelClassificationWorkflow)

        # The training operator
        opPixelClassification = shell.workflow.pcApplet.topLevelOperator

        # Sanity checks
        assert len(opPixelClassification.InputImages) > 0
        assert opPixelClassification.Classifier.ready()

        # print opPixelClassification.

        label_names = opPixelClassification.LabelNames.value
        label_colors = opPixelClassification.LabelColors.value
        probability_colors = opPixelClassification.PmapColors.value

        print label_names, label_colors, probability_colors

        # Change the connections of the batch prediction pipeline so we can supply our own data.
        opBatchFeatures = shell.workflow.opBatchFeatures
        opBatchPredictionPipeline = shell.workflow.opBatchPredictionPipeline

        opBatchFeatures.InputImage.disconnect()
        opBatchFeatures.InputImage.resize(1)
        opBatchFeatures.InputImage[0].setValue(input_data)

        # Run prediction.
        assert len(opBatchPredictionPipeline.HeadlessPredictionProbabilities) == 1
        assert opBatchPredictionPipeline.HeadlessPredictionProbabilities[0].ready()
        predictions = opBatchPredictionPipeline.HeadlessPredictionProbabilities[0][:].wait()
        return predictions

    def display(self, workspace, figure):
        figure.set_subplots((len(workspace.display_data.dest_images) + 1, 1))
        source_image = workspace.display_data.source_image
        if source_image.ndim == 3:
            src_plot = figure.subplot_imshow_color(
                    0, 0, source_image, title=self.image_name.value)
        else:
            src_plot = figure.subplot_imshow_grayscale(
                    0, 0, source_image, title=self.image_name.value)
        for i, dest_image in enumerate(workspace.display_data.dest_images):
            figure.subplot_imshow_grayscale(
                    i + 1, 0, dest_image,
                    title=self.probability_maps[i].output_image.value,
                    sharexy=src_plot)

    def validate_module(self, pipeline):
        '''Mark IlastikPixelClassification as invalid if Ilastik is not properly installed

        '''
        if not has_ilastik:
            raise cps.ValidationError(
                    "IlastikPixelClassification is not available on this platform.",
                    self.no_ilastik_msg)
        if self.h5_directory.dir_choice != URL_FOLDER_NAME:
            fileName = os.path.join(
                    self.h5_directory.get_absolute_path(),
                    self.classifier_file_name.value)
            if not os.path.isfile(fileName):
                if len(self.classifier_file_name.value) == 0:
                    msg = "Please select a classifier file"
                else:
                    msg = "Could not find the classifier file, \"%s\"." % \
                          fileName

                raise cps.ValidationError(msg, self.classifier_file_name)

    def prepare_settings(self, setting_values):
        '''Prepare the module to receive the settings'''
        n_maps = int(setting_values[SI_PROBABILITY_MAP_COUNT])
        if len(self.probability_maps) > n_maps:
            del self.probability_maps[n_maps:]
        elif len(self.probability_maps) < n_maps:
            for _ in range(len(self.probability_maps), n_maps):
                self.add_probability_map()

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''Prepare the module's settings for the batch target environment

        workspace - workspace / measurements / pipeline for batch file

        fn_alter_path - call this to alter any file path to target the
                        batch environment.
        '''
        self.h5_directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        '''Upgrade settings to maintain backwards compatibility

        setting_values - list of setting strings
        variable_revision_number - version number used to save the settings
        module_name - original module name used to save the settings
        from_matlab - true if CellProfiler 1.0 pipeline
        '''
        if variable_revision_number == 1:
            setting_values = [
                setting_values[0],  # image_name
                setting_values[3],  # h5_directory
                setting_values[4],  # classifier_file_name
                "1",  # probability map count = 1
                setting_values[1],  # output_image
                setting_values[2]]  # class_sel
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
