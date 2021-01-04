# coding=utf-8

#################################
#
# Imports from useful Python libraries
#
#################################

import csv
import numpy
import os
import re
from urllib.request import urlretrieve

from io import StringIO

#################################
#
# Imports from CellProfiler
#
##################################

from cellprofiler.modules import _help
import cellprofiler_core.image
import cellprofiler_core.module
import cellprofiler_core.measurement
import cellprofiler_core.object
import cellprofiler_core.setting

__doc__ = """\
CallBarcodes
============

**CallBarcodes** - This module calls barcodes.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          Yes           YES
============ ============ ===============


What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

Are there any assumptions about input data someone using this module
should be made aware of? For example, is there a strict requirement that
image data be single-channel, or that the foreground is brighter than
the background? Describe any assumptions here.

This section can be omitted if there is no requirement on the input.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

Describe the output of this module. This is necessary if the output is
more complex than a single image. For example, if there is data displayed
over the image then describe what the data represents.

This section can be omitted if there is no specialized output.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Describe the measurements made by this module. Typically, measurements
are described in the following format:

**Measurement category:**

-  *MeasurementName*: A brief description of the measurement.
-  *MeasurementName*: A brief description of the measurement.

**Measurement category:**

-  *MeasurementName*: A brief description of the measurement.
-  *MeasurementName*: A brief description of the measurement.

This module makes the following measurements:

**MT** (the MeasurementTemplate category):

-  *Intensity_[IMAGE_NAME]_N[Ni]_M[Mj]*: the Zernike feature of the
   IMAGE_NAME image with radial degree Ni and Azimuthal degree Mj,
   Mj >= 0.
-  *Intensity_[IMAGE_NAME]_N[Ni]_MM[Mj]*: the Zernike feature of
   the IMAGE_NAME image with radial degree Ni and Azimuthal degree
   Mj, Mj < 0.

Technical notes
^^^^^^^^^^^^^^^

Include implementation details or notes here. Additionally provide any
other background information about this module, including definitions
or adopted conventions. Information which may be too specific to fit into
the general description should be provided here.

Omit this section if there is no technical information to mention.

The Zernike features measured here are themselves interesting. You can
reconstruct the image of a cell, approximately, by constructing the Zernike
functions on a unit circle, multiplying the real parts by the corresponding
features for positive M, multiplying the imaginary parts by the corresponding
features for negative M and adding real and imaginary parts.

References
^^^^^^^^^^

Provide citations here, if appropriate. Citations are formatted as a list and,
wherever possible, include a link to the original work. For example,

-  Meyer F, Beucher S (1990) “Morphological segmentation.” *J Visual
   Communication and Image Representation* 1, 21-46.
   (`link <http://dx.doi.org/10.1016/1047-3203(90)90014-M>`__)
"""

#
# Constants
#
# It's good programming practice to replace things like strings with
# constants if they will appear more than once in your program. That way,
# if someone wants to change the text, that text will change everywhere.
# Also, you can't misspell it by accident.
#
"""This is the measurement template category"""
C_CALL_BARCODES = "Barcode"


#
# The module class
#
# Your module should "inherit" from cellprofiler_core.module.Module.
# This means that your module will use the methods from Module unless
# you re-implement them. You can let Module do most of the work and
# implement only what you need.
#
class CallBarcodes(cellprofiler_core.module.Module):
    #
    # The module starts by declaring the name that's used for display,
    # the category under which it is stored and the variable revision
    # number which can be used to provide backwards compatibility if
    # you add user-interface functionality later.
    #
    module_name = "CallBarcodes"
    category = "Data Tools"
    variable_revision_number = 1

    #
    # "create_settings" is where you declare the user interface elements
    # (the "settings") which the user will use to customize your module.
    #
    # You can look at other modules and in cellprofiler_core.settings for
    # settings you can use.
    #
    def create_settings(self):
        self.csv_directory = cellprofiler_core.setting.text.Directory(
            "Input data file location",
            allow_metadata=False,
            doc="""\
Select the folder containing the CSV file to be loaded. {IO_FOLDER_CHOICE_HELP_TEXT}
""".format(
                **{"IO_FOLDER_CHOICE_HELP_TEXT": _help.IO_FOLDER_CHOICE_HELP_TEXT}
            ),
        )

        def get_directory_fn():
            """Get the directory for the CSV file name"""
            return self.csv_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.csv_directory.get_parts_from_path(path)
            self.csv_directory.join_parts(dir_choice, custom_path)

        self.csv_file_name = cellprofiler_core.setting.text.Filename(
            "Name of the file",
            "None",
            doc="""Provide the file name of the CSV file containing the data you want to load.""",
            get_directory_fn=get_directory_fn,
            set_directory_fn=set_directory_fn,
            browse_msg="Choose CSV file",
            exts=[("Data file (*.csv)", "*.csv"), ("All files (*.*)", "*.*")],
        )

        #
        # The ObjectNameSubscriber is similar to the ImageNameSubscriber.
        # It will ask the user which object to pick from the list of
        # objects provided by upstream modules.
        #
        self.input_object_name = cellprofiler_core.setting.subscriber.LabelSubscriber(
            text="Input object name",
            doc="These are the objects that the module operates on.",
        )

        self.ncycles = cellprofiler_core.setting.text.Integer(
            doc="""\
Enter the number of cycles present in the data.
""",
            text="Number of cycles",
            value=8,
        )
        self.cycle1measure = cellprofiler_core.setting.Measurement(
            "Select one of the measures from Cycle 1 to use for calling",
            self.input_object_name.get_value,
            "AreaShape_Area",
            doc="""\
This measurement should be """,
        )

        self.metadata_field_barcode = cellprofiler_core.setting.choice.Choice(
            "Select the column of barcodes to match against",
            ["No CSV file"],
            choices_fn=self.get_choices,
            doc="""\
""",
        )

        self.metadata_field_tag = cellprofiler_core.setting.choice.Choice(
            "Select the column with gene/transcript barcode names",
            ["No CSV file"],
            choices_fn=self.get_choices,
            doc="""\
""",
        )

        self.wants_call_image = cellprofiler_core.setting.Binary(
            "Retain an image of the barcodes color coded by call?",
            False,
            doc="""\
Select "Yes" to retain the image of the objects color-coded
according to which line of the CSV their barcode call matches to,
for use later in the pipeline (for example, to be saved by a **SaveImages**
module).""",
        )

        self.outimage_calls_name = cellprofiler_core.setting.text.ImageName(
            "Enter the called barcode image name",
            "None",
            doc="""\
*(Used only if the called barcode image is to be retained for later use in the pipeline)*

Enter the name to be given to the called barcode image.""",
        )

        self.wants_score_image = cellprofiler_core.setting.Binary(
            "Retain an image of the barcodes color coded by score match?",
            False,
            doc="""\
Select "Yes" to retain the image of the objects where the intensity of the spot matches
indicates the match score between the called barcode and its closest match,
for use later in the pipeline (for example, to be saved by a **SaveImages**
module).""",
        )

        self.outimage_score_name = cellprofiler_core.setting.text.ImageName(
            "Enter the barcode score image name",
            "None",
            doc="""\
*(Used only if the barcode score image is to be retained for later use in the pipeline)*

Enter the name to be given to the barcode score image.""",
        )

    #
    # The "settings" method tells CellProfiler about the settings you
    # have in your module. CellProfiler uses the list for saving
    # and restoring values for your module when it saves or loads a
    # pipeline file.
    #
    # This module does not have a "visible_settings" method. CellProfiler
    # will use "settings" to make the list of user-interface elements
    # that let the user configure the module. See imagetemplate.py for
    # a template for visible_settings that you can cut and paste here.
    #
    def settings(self):
        return [
            self.ncycles,
            self.input_object_name,
            self.cycle1measure,
            self.csv_directory,
            self.csv_file_name,
            self.metadata_field_barcode,
            self.metadata_field_tag,
            self.wants_call_image,
            self.outimage_calls_name,
            self.wants_score_image,
            self.outimage_score_name,
        ]

    def visible_settings(self):
        result = [
            self.ncycles,
            self.input_object_name,
            self.cycle1measure,
            self.csv_directory,
            self.csv_file_name,
            self.metadata_field_barcode,
            self.metadata_field_tag,
            self.wants_call_image,
            self.wants_score_image,
        ]

        if self.wants_call_image:
            result += [self.outimage_calls_name]

        if self.wants_score_image:
            result += [self.outimage_score_name]

        return result

    def validate_module(self, pipeline):
        csv_path = self.csv_path

        if not os.path.isfile(csv_path):
            raise cellprofiler_core.setting.ValidationError(
                "No such CSV file: %s" % csv_path, self.csv_file_name
            )

        try:
            self.open_csv()
        except IOError as e:
            import errno

            if e.errno == errno.EWOULDBLOCK:
                raise cellprofiler_core.setting.ValidationError(
                    "Another program (Excel?) is locking the CSV file %s."
                    % self.csv_path,
                    self.csv_file_name,
                )
            else:
                raise cellprofiler_core.setting.ValidationError(
                    "Could not open CSV file %s (error: %s)" % (self.csv_path, e),
                    self.csv_file_name,
                )

        try:
            self.get_header()
        except Exception as e:
            raise cellprofiler_core.setting.ValidationError(
                "The CSV file, %s, is not in the proper format."
                " See this module's help for details on CSV format. (error: %s)"
                % (self.csv_path, e),
                self.csv_file_name,
            )

    @property
    def csv_path(self):
        """The path and file name of the CSV file to be loaded"""
        path = self.csv_directory.get_absolute_path()
        return os.path.join(path, self.csv_file_name.value)

    def open_csv(self, do_not_cache=False):
        """Open the csv file or URL, returning a file descriptor"""
        global header_cache

        if cellprofiler.preferences.is_url_path(self.csv_path):
            if self.csv_path not in header_cache:
                header_cache[self.csv_path] = {}
            entry = header_cache[self.csv_path]
            if "URLEXCEPTION" in entry:
                raise entry["URLEXCEPTION"]
            if "URLDATA" in entry:
                fd = StringIO(entry["URLDATA"])
            else:
                if do_not_cache:
                    raise RuntimeError("Need to fetch URL manually.")
                try:
                    url = cellprofiler.misc.generate_presigned_url(self.csv_path)
                    url_fd, headers = urlretrieve(url)
                except Exception as e:
                    entry["URLEXCEPTION"] = e
                    raise e
                fd = StringIO()
                while True:
                    text = url_fd.read()
                    if len(text) == 0:
                        break
                    fd.write(text)
                fd.seek(0)
                entry["URLDATA"] = fd.getvalue()
            return fd
        else:
            return open(self.csv_path, "rb")

    def get_header(self, do_not_cache=False):
        """Read the header fields from the csv file

        Open the csv file indicated by the settings and read the fields
        of its first line. These should be the measurement columns.
        """
        fd = self.open_csv(do_not_cache=do_not_cache)
        reader = csv.reader(fd)
        header = next(reader)
        fd.close()
        return header

    def get_choices(self, pipeline):
        try:
            choices = self.get_header()
        except:
            choices = ["No CSV file"]
        return choices

    #
    # CellProfiler calls "run" on each image set in your pipeline.
    #
    def run(self, workspace):
        #
        # Get the measurements object - we put the measurements we
        # make in here
        #
        measurements = workspace.measurements
        listofmeasurements = measurements.get_feature_names(
            self.input_object_name.value
        )

        measurements_for_calls = self.getallbarcodemeasurements(
            listofmeasurements, self.ncycles.value, self.cycle1measure.value
        )

        calledbarcodes = self.callonebarcode(
            measurements_for_calls,
            measurements,
            self.input_object_name.value,
            self.ncycles.value,
        )

        workspace.measurements.add_measurement(
            self.input_object_name.value,
            "_".join([C_CALL_BARCODES, "BarcodeCalled"]),
            calledbarcodes,
        )

        barcodes = self.barcodeset(
            self.metadata_field_barcode.value, self.metadata_field_tag.value
        )

        scorelist = []
        matchedbarcode = []
        matchedbarcodecode = []
        matchedbarcodeid = []
        if self.wants_call_image or self.wants_score_image:
            objects = workspace.object_set.get_objects(self.input_object_name.value)
            labels = objects.segmented
            pixel_data_call = objects.segmented
            pixel_data_score = objects.segmented
        count = 1
        for eachbarcode in calledbarcodes:
            eachscore, eachmatch = self.queryall(barcodes, eachbarcode)
            scorelist.append(eachscore)
            matchedbarcode.append(eachmatch)
            matchedbarcodeid.append(barcodes[eachmatch][0])
            matchedbarcodecode.append(barcodes[eachmatch][1])
            if self.wants_call_image:
                pixel_data_call = numpy.where(
                    labels == count, barcodes[eachmatch][0], pixel_data_call
                )
            if self.wants_score_image:
                pixel_data_score = numpy.where(
                    labels == count, 65535 * eachscore, pixel_data_score
                )
            count += 1
        workspace.measurements.add_measurement(
            self.input_object_name.value,
            "_".join([C_CALL_BARCODES, "MatchedTo_Barcode"]),
            matchedbarcode,
        )
        workspace.measurements.add_measurement(
            self.input_object_name.value,
            "_".join([C_CALL_BARCODES, "MatchedTo_ID"]),
            matchedbarcodeid,
        )
        workspace.measurements.add_measurement(
            self.input_object_name.value,
            "_".join([C_CALL_BARCODES, "MatchedTo_GeneCode"]),
            matchedbarcodecode,
        )
        workspace.measurements.add_measurement(
            self.input_object_name.value,
            "_".join([C_CALL_BARCODES, "MatchedTo_Score"]),
            scorelist,
        )
        if self.wants_call_image:
            workspace.image_set.add(
                self.outimage_calls_name.value,
                cellprofiler_core.image.Image(
                    pixel_data_call.astype("uint16"), convert=False
                ),
            )
        if self.wants_score_image:
            workspace.image_set.add(
                self.outimage_score_name.value,
                cellprofiler_core.image.Image(
                    pixel_data_score.astype("uint16"), convert=False
                ),
            )
        #
        # We record some statistics which we will display later.
        # We format them so that Matplotlib can display them in a table.
        # The first row is a header that tells what the fields are.
        #
        statistics = [["Feature", "Mean", "Median", "SD"]]

        #
        # Put the statistics in the workspace display data so we
        # can get at them when we display
        #
        workspace.display_data.statistics = statistics

    #
    # "display" lets you use matplotlib to display your results.
    #
    def display(self, workspace, figure):
        statistics = workspace.display_data.statistics

        figure.set_subplots((1, 1))

        figure.subplot_table(0, 0, statistics)

    def getallbarcodemeasurements(self, measurements, ncycles, examplemeas):
        stem = re.split("Cycle", examplemeas)[0]
        measurementdict = {}
        for eachmeas in measurements:
            if stem in eachmeas:
                to_parse = re.split("Cycle", eachmeas)[1]
                find_cycle = re.search("[0-9]{1,2}", to_parse)
                parsed_cycle = int(find_cycle.group(0))
                find_base = re.search("[A-Z]", to_parse)
                parsed_base = find_base.group(0)
                if parsed_cycle <= ncycles:
                    if parsed_cycle not in measurementdict.keys():
                        measurementdict[parsed_cycle] = {eachmeas: parsed_base}
                    else:
                        measurementdict[parsed_cycle].update({eachmeas: parsed_base})
        return measurementdict

    def callonebarcode(self, measurementdict, measurements, object_name, ncycles):
        master_cycles = []

        for eachcycle in range(1, ncycles + 1):
            cycles_measures_perobj = []
            cyclecode = []
            cycledict = measurementdict[eachcycle]
            cyclemeasures = cycledict.keys()
            for eachmeasure in cyclemeasures:
                cycles_measures_perobj.append(
                    measurements.get_current_measurement(object_name, eachmeasure)
                )
                cyclecode.append(measurementdict[eachcycle][eachmeasure])
            cycle_measures_perobj = numpy.transpose(numpy.array(cycles_measures_perobj))
            max_per_obj = numpy.argmax(cycle_measures_perobj, 1)
            max_per_obj = list(max_per_obj)
            max_per_obj = [cyclecode[x] for x in max_per_obj]
            master_cycles.append(list(max_per_obj))

        return list(map("".join, zip(*master_cycles)))

    def barcodeset(self, barcodecol, genecol):
        fd = self.open_csv()
        reader = csv.DictReader(fd)
        barcodeset = {}
        count = 1
        for row in reader:
            if len(row[barcodecol]) != 0:
                barcodeset[row[barcodecol]] = (count, row[genecol])
                count += 1
        fd.close()
        return barcodeset

    def queryall(self, barcodeset, query):
        barcodelist = barcodeset.keys()
        scoredict = {
            sum([1 for x in range(len(query)) if query[x] == y[x]])
            / float(len(query)): y
            for y in barcodelist
        }
        scores = list(scoredict.keys())
        scores.sort(reverse=True)
        return scores[0], scoredict[scores[0]]

    #
    # We have to tell CellProfiler about the measurements we produce.
    # There are two parts: one that is for database-type modules and one
    # that is for the UI. The first part gives a comprehensive list
    # of measurement columns produced. The second is more informal and
    # tells CellProfiler how to categorize its measurements.
    #
    # "get_measurement_columns" gets the measurements for use in the database
    # or in a spreadsheet. Some modules need this because they
    # might make measurements of measurements and need those names.
    #
    def get_measurement_columns(self, pipeline):
        #
        # The first thing in the list is the object being measured. If it's
        # the whole image, use cellprofiler.measurement.IMAGE as the name.
        #
        # The second thing is the measurement name.
        #
        # The third thing is the column type. See the COLTYPE constants
        # in measurement.py for what you can use
        #
        input_object_name = self.input_object_name.value

        return [
            (
                input_object_name,
                "_".join([C_CALL_BARCODES, "BarcodeCalled"]),
                cellprofiler.measurement.COLTYPE_VARCHAR,
            ),
            (
                input_object_name,
                "_".join([C_CALL_BARCODES, "MatchedTo_Barcode"]),
                cellprofiler.measurement.COLTYPE_VARCHAR,
            ),
            (
                input_object_name,
                "_".join([C_CALL_BARCODES, "MatchedTo_ID"]),
                cellprofiler.measurement.COLTYPE_INTEGER,
            ),
            (
                input_object_name,
                "_".join([C_CALL_BARCODES, "MatchedTo_GeneCode"]),
                cellprofiler.measurement.COLTYPE_VARCHAR,
            ),
            (
                input_object_name,
                "_".join([C_CALL_BARCODES, "MatchedTo_Score"]),
                cellprofiler.measurement.COLTYPE_FLOAT,
            ),
        ]

    #
    # "get_categories" returns a list of the measurement categories produced
    # by this module. It takes an object name - only return categories
    # if the name matches.
    #
    def get_categories(self, pipeline, object_name):
        if object_name == self.input_object_name:
            return [C_CALL_BARCODES]

        return []

    #
    # Return the feature names if the object_name and category match
    #
    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.input_object_name and category == C_CALL_BARCODES:
            return [
                "BarcodeCalled",
                "MatchedTo_Barcode",
                "MatchedTo_ID",
                "MatchedTo_GeneCode",
                "MatchedTo_Score",
            ]

        return []
