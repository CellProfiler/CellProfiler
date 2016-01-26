'''<b>RunScript</b> runs a Python script.
<hr>
RunScript runs a Python script during the course of pipeline processing. You
can import images, objects and measurements into the script as Python variables
and you can export images, objects and measurements from the script after
it has run. Scripts can be run on the first image set of a group, the last image set of
a group or all image sets. You can learn more about CellProfiler's internals
at the developer's website: https://github.com/CellProfiler/CellProfiler.

<h4>Variables</h4>

<ul><li><b>workspace</b>: The workspace is a predefined variable. Your script
can access the images, objects, measurements and the pipeline itself from
the workspace variable.</li>
<li><b>Images</b>: An image is a <a href="http://www.numpy.org">Numpy</a> array.
Grayscale and binary images are two dimensional arrays. The first axis is Y,
the second is X. Color images are three dimensional arrays with axes of Y, X
and color and the indices for the color axis are red=0, green=1 and blue=2.
Binary images have a dtype of numpy.bool and other images are floating point
with values generally restricted to 0 and 1 by convention.</li>
<li><b>Objects</b>: Objects are stored as a list of 2D integer arrays. Pixels
in each array are labeled as 0 if they are background and with the object number
of the corresponding object if they are foreground (see 
<a href="http://scipy.github.io/devdocs/generated/scipy.ndimage.label.html">
scipy.ndimage.measurements</a>). Typically, the list contains a single array,
but if objects are overlapping, CellProfiler uses two or more arrays in order
to represent pixels with more than one label.</li>
<li><b>Measurements</b>: Image measurements are stored in a single value.
Object measurements are stored in a 1D Numpy array of one value per object
(object #1's value is stored at index 0 and so on).</li>
</ul>
'''
import ast
import exceptions

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import numpy as np

WHEN_FIRST = "First"
WHEN_ALL = "All"
WHEN_LAST = "Last"

IO_IMAGE = "Image measurement"
IO_OBJECTS = "Object measurement"

class RunScript(cpm.CPModule):
    variable_revision_number = 1
    module_name = "RunScript"
    category = "Other"
    
    def create_settings(self):
        self.when = cps.Choice(
            "Run on which image sets?",
            [ WHEN_FIRST, WHEN_ALL, WHEN_LAST ],
            value = WHEN_ALL,
            doc = 
            """This setting restricts RunScript to running the script on the
            <i>%(WHEN_FIRST)s</i> or <i>%(WHEN_LAST)s</i> image set of the
            group or runs the script on <i>%(WHEN_ALL)s</i> image sets.
            """
        )
        self.input_images = []
        self.input_image_count = cps.HiddenCount(
            self.input_images, "Input image count")
        self.add_input_images_button = cps.DoSomething(
            "Add input image", "Add", self.add_input_image,
            doc="Add an image to be imported into the script.")

        self.input_objects = []
        self.input_object_count = cps.HiddenCount(
            self.input_objects, "Input object count")
        self.add_input_objects_button = cps.DoSomething(
            "Add input objects", "Add", self.add_input_objects,
            doc="Add objects to be imported into the script.")

        self.input_measurements = []
        self.input_measurement_count = cps.HiddenCount(
            self.input_measurements, "Input measurement count")
        self.add_input_measurement_button = cps.DoSomething(
            "Add input measurement", "Add", self.add_input_measurement,
            doc="Add a measurement to be imported into the script.")

        self.script = cps.Text(
            "Script", 
            'print "Hello, world."', 
            multiline=True,
            doc = """This setting holds the Python script to be run.""")
        
        self.output_images = []
        self.output_image_count = cps.HiddenCount(
            self.output_images, "Output image count")
        self.add_output_images_button = cps.DoSomething(
            "Add output image", "Add", self.add_output_image,
            doc="Add an image to be exported from the script.")

        self.output_objects = []
        self.output_object_count = cps.HiddenCount(
            self.output_objects, "Output object count")
        self.add_output_objects_button = cps.DoSomething(
            "Add output objects", "Add", self.add_output_objects,
            doc="Add objects to be exported from the script.")

        self.output_measurements = []
        self.output_measurement_count = cps.HiddenCount(
            self.output_measurements, "Output measurement count")
        self.add_output_measurement_button = cps.DoSomething(
            "Add output measurement", "Add", self.add_output_measurement,
            doc="Add a measurement to be exported from the script.")
        
    def add_input_image(self):
        group = cps.SettingsGroup()
        group.append(
            "image_name",
            cps.ImageNameSubscriber(
                "Input image", 
            doc="""Select an image from among those provided by previous
            modules."""))
        group.append("variable_name", cps.Text(
            "Variable name", "image",
            doc="""<b>RunScript</b> will define a variable that you can use
            in your script that has the name you enter here. The variable
            will hold the input image that you chose."""))
        group.append("remover", 
                     cps.RemoveSettingButton(
                         "", "Remove this image", self.input_images, group))
        self.input_images.append(group)
    
    def add_input_objects(self):
        group = cps.SettingsGroup()
        group.append(
            "objects_name",
            cps.ObjectNameSubscriber(
                "Input objects", 
            doc="""Select an objects from among those provided by previous
            modules."""))
        group.append("variable_name", cps.Text(
            "Variable name", "objects",
            doc="""<b>RunScript</b> will define a variable that you can use
            in your script that has the name you enter here. The variable
            will hold the input objects that you chose."""))
        group.append("remover", 
                     cps.RemoveSettingButton(
                         "", "Remove this objects", self.input_objects, group))
        self.input_objects.append(group)
    
    def add_input_measurement(self):
        group = cps.SettingsGroup()
        group.append("measurement_type", cps.Choice(
            "Measurement type", [IO_IMAGE, IO_OBJECTS],
            doc="""This setting determines whether the measurement is an
            image or object measurement."""))
        group.append("objects_name", cps.ObjectNameSubscriber(
        "Measured objects", "Nuclei",
        doc="""<i>(Used only when importing object measurements)</i><br>
        This setting chooses the measurement's objects.
        """))
        def object_fn():
            if group.measurement_type == IO_IMAGE:
                return cpmeas.IMAGE
            return group.objects_name.value
        
        group.append(
            "measurement",
            cps.Measurement(
                "Measurement", object_fn,
                doc="""Select a measurement from among those 
                provided by previous modules."""))
        group.append("variable_name", cps.Text(
            "Variable name", "measurement",
            doc="""<b>RunScript</b> will define a variable that you can use
            in your script that has the name you enter here. The variable
            will hold the measurement that you chose."""))
        group.append("remover", 
                     cps.RemoveSettingButton(
                         "", "Remove this image", 
                         self.input_measurements, group))
        self.input_measurements.append(group)
    
    def add_output_image(self):
        group = cps.SettingsGroup()
        group.append(
            "image_name",
            cps.ImageNameProvider(
                "Output image", 
            doc="""Name the image to be exported."""))
        group.append("variable_name", cps.Text(
            "Variable name", "image",
            doc="""<b>RunScript</b> will turn the array stored in the
            variable in your script with this name into an image that has
            the output image name provided above."""))
        group.append("remover", 
                     cps.RemoveSettingButton(
                         "", "Remove this image", self.output_images, group))
        self.output_images.append(group)
    
    def add_output_objects(self):
        group = cps.SettingsGroup()
        group.append(
            "objects_name",
            cps.ObjectNameProvider(
                "Output objects", 
            doc="""Name the objects to be exported."""))
        group.append("variable_name", cps.Text(
            "Variable name", "objects",
            doc="""<b>RunScript</b> will turn the array stored in the
            variable in your script with this name into objects that have
            the output objects name provided above."""))
        group.append("remover", 
                     cps.RemoveSettingButton(
                         "", "Remove these objects", self.output_objects, group))
        self.output_objects.append(group)
    
    def add_output_measurement(self):
        group = cps.SettingsGroup()
        group.append(
            "measurement_name",
            cps.Text(
                "Measurement name", 
                "Script_Measurement",
                doc="""Name the measurement to be exported."""))
        group.append("variable_name", cps.Text(
            "Variable name", "measurement",
            doc="""<b>RunScript</b> will turn the measurement stored in the
            variable in your script with this name into a measurement."""))
        group.append("measurement_type", cps.Choice(
            "Measurement type", [IO_IMAGE, IO_OBJECTS],
            doc="""This setting determines whether your measurement will
            be stored as an image measurement or an object measurement.
            If it is an image measurement, it will treat your variable
            as a single value. If it is an object measurement, it will
            treat your variable as an array and store it as one measurement
            per object."""))
        group.append("objects_name", cps.ObjectNameSubscriber(
            "Objects name", "Nuclei",
            doc="""<i>(Used only for object measurement types)</i>
            This setting chooses the name of the objects being measured.
            The measurement will be stored with others for these objects."""))
        group.append("remover", 
                     cps.RemoveSettingButton(
                         "", "Remove this measurement", self.output_objects, group))
        self.output_measurements.append(group)
        
    def settings(self):
        result = [
            self.input_image_count, self.input_object_count, 
            self.input_measurement_count,
            self.output_image_count, self.output_object_count, 
            self.output_measurement_count, self.when, self.script]
        for grouplist in self.input_images, self.input_objects, \
            self.input_measurements, self.output_images, self.output_objects,\
            self.output_measurements:
            for group in grouplist:
                result += group.pipeline_settings()
        return result
            
    def visible_settings(self):
        result = [self.when]
        for grouplist, add_button in (
            (self.input_images, self.add_input_images_button),
            (self.input_objects, self.add_input_objects_button)):
            for group in grouplist:
                result += group.visible_settings()
            result.append(add_button)
        for group in self.input_measurements:
            result.append(group.measurement_type)
            if group.measurement_type == IO_OBJECTS:
                result.append(group.objects_name)
            result += [group.measurement, group.variable_name, group.remover]
        result.append(self.add_input_measurement_button)
        result.append(self.script)
        for grouplist, add_button in (
            (self.output_images, self.add_output_images_button),
            (self.output_objects, self.add_output_objects_button)):
            for group in grouplist:
                result += group.visible_settings()
            result.append(add_button)
        for group in self.output_measurements:
            result += [group.measurement_name, group.variable_name,
                       group.measurement_type]
            if group.measurement_type == IO_OBJECTS:
                result.append(group.objects_name)
            result.append(group.remover)
        result.append(self.add_output_measurement_button)
        return result
        
    def prepare_settings(self, setting_values):
        for value, (group, add_fn) in zip(
            setting_values[:6],
            ((self.input_images, self.add_input_image),
             (self.input_objects, self.add_input_objects),
             (self.input_measurements, self.add_input_measurement),
             (self.output_images, self.add_output_image),
             (self.output_objects, self.add_output_objects),
             (self.output_measurements, self.add_output_measurement))):
            count = int(value)
            del group[:]
            for _ in range(count):
                add_fn()
                
    def run(self, workspace):
        if not self.should_run(workspace):
            return
        #
        # Introduce the necessary elements into the workspace
        #
        self.inject_images(workspace, locals())
        self.inject_objects(workspace, locals())
        self.inject_measurements(workspace, locals())
        script = ast.parse(self.script.value, filename="RunScript")
        code = compile(script, "RunScript", "exec")
        exec code
        self.harvest_images(workspace, locals())
        self.harvest_objects(workspace, locals())
        self.harvest_measurements(workspace, locals())
        
    def should_run(self, workspace):
        '''Return True if this image set should be run'''
        if self.when == WHEN_ALL:
            return True
        m = workspace.measurements
        if self.when == WHEN_FIRST:
            return m.is_first_in_group
        return m.is_last_in_group
    
    def inject_images(self, workspace, ld):
        '''Add image variables into the locals dictionary
        
        workspace - workspace for this image set
        ld - locals dictionary
        '''
        image_set = workspace.image_set
        for group in self.input_images:
            ld[group.variable_name.value] = \
                image_set.get_image(group.image_name.value).pixel_data
    
    def inject_objects(self, workspace, ld):
        '''Add object variables into the locals dictionary
        
        workspace - workspace for this image set
        ld - locals dictionary
        '''
        object_set = workspace.object_set
        for group in self.input_objects:
            objects = object_set.get_objects(group.objects_name.value)
            ld[group.variable_name.value] = [
                _[0] for _ in objects.get_labels()]

    def inject_measurements(self, workspace, ld):
        m = workspace.measurements
        for group in self.input_measurements:
            setting = group.measurement
            value = m[setting.get_measurement_object(),
                      setting.value]
            ld[group.variable_name.value] = value
            
    def harvest_images(self, workspace, ld):
        '''Transfer images from variables to the image_set
        
        workspace - workspace for the image set
        ld - dictionary of locals
        '''
        image_set = workspace.image_set
        for group in self.output_images:
            pixel_data = ld[group.variable_name.value]
            image = cpi.Image(pixel_data)
            image_set.add(group.image_name.value, image)
            
    def harvest_objects(self, workspace, ld):
        '''Transfer objects from variables to the object_set
        
        workspace - workspace for the image set
        ld - dictionary of locals
        '''
        object_set = workspace.object_set
        for group in self.output_objects:
            labels = ld[group.variable_name.value]
            objects = cpo.Objects()
            if len(labels) == 1:
                objects.segmented = labels[0]
            else:
                i = []
                j = []
                v = []
                for l in labels:
                    ii, jj = np.where(l > 0)
                    vv = l[ii, jj]
                    i.append(ii)
                    j.append(jj)
                    v.append(vv)
                ijv = np.column_stack((i, j, v))
                objects.set_ijv(ijv, shape = l.shape)
            object_set.add_objects(objects, group.objects_name.value)
        
    def harvest_measurements(self, workspace, ld):
        m = workspace.measurements
        for group in self.output_measurements:
            value = ld[group.variable_name.value]
            measurement_name = group.measurement_name.value
            if group.measurement_type == IO_IMAGE:
                m[cpmeas.IMAGE, measurement_name] = value
            else:
                m[group.objects_name.value, measurement_name] = value
            
    def validate_module(self, pipeline):
        for grouplist in \
            self.input_images, self.input_objects, self.input_measurements, \
            self.output_images, self.output_objects, self.output_measurements:
            for group in grouplist:
                variable = group.variable_name
                try:
                    parsing = ast.parse(variable.value)
                    if len(parsing.body) > 1:
                        raise SyntaxError()
                    if not isinstance(parsing.body[0], ast.Expr):
                        raise SyntaxError()
                    if not isinstance(parsing.body[0].value, ast.Name):
                        raise SyntaxError()
                except exceptions.SyntaxError:
                    raise cps.ValidationError(
                    "Variables names can only be composed of letters, digits"
                    " and the underbar (\"_\") character and must not start with"
                    " a digit.", variable)
            
        