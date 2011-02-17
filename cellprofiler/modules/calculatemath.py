
'''<b>Calculate Math</b> takes measurements produced by previous modules and
performs basic arithmetic operations
<hr>

The arithmetic operations available in this module include addition,
subtraction, multiplication, and division. The result can be
log-transformed or raised to a power and can be used in further
calculations if another <b>CalculateMath</b> module is added to the pipeline. 

<p>The module can make its calculations on a per-image basis (for example, multiplying the area occupied by a stain
in the image by the total intensity in the image) or on an object-by-object basis (for example, dividing the intensity
in the nucleus by the intensity in the cytoplasm for each cell).

<h4>Available measurements</h4>
<ul>
<li><i>Image features:</i> If both input measurements are whole-image
measurements, then the result will also be a whole-image measurement. </li>
<li><i>Object features:</i> Object measurements can be produced in two ways: 
<ul>
<li>If both input measurements are individual object measurements, then the 
result will also be an object measurement. In these cases, the measurement will 
be associated with <i>both</i> objects that were involved in the measurement.</li>
<li>If one measure is object-based and one image-based, then the result will be 
an object measurement.</li>
</ul>
</li>
</ul>
The result of these calculations is a new measurement in the "Math" category. 

See also all <b>Measure</b> modules.
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps

O_MULTIPLY = "Multiply"
O_DIVIDE = "Divide"
O_ADD = "Add"
O_SUBTRACT = "Subtract"
O_NONE = "None"

O_ALL = [O_MULTIPLY, O_DIVIDE, O_ADD, O_SUBTRACT, O_NONE]

MC_IMAGE = cpmeas.IMAGE
MC_OBJECT = "Object"
MC_ALL = [MC_IMAGE, MC_OBJECT]

C_MATH = "Math"

class CalculateMath(cpm.CPModule):

    module_name = "CalculateMath"
    category="Data Tools"
    variable_revision_number = 1
    
    def create_settings(self):
        #XXX needs to use cps.SettingsGroup
        class Operand(object):
            '''Represents the collection of settings needed by each operand'''
            def __init__(self, index, operation):
                self.__index = index
                self.__operation = operation
                self.__operand_choice = cps.Choice(self.operand_choice_text(), MC_ALL,doc="""
                                            Is the operand an image or object measurement?""")
                
                self.__operand_objects = cps.ObjectNameSubscriber(self.operand_objects_text(),"None",doc="""
                                            Which objects do you want to measure for this operation?""")
                
                self.__operand_measurement = cps.Measurement(self.operand_measurement_text(),
                                            self.object_fn,doc="""
                                            Enter the category that was used to create the measurement. You
                                            will be prompted to add additional information depending on 
                                            the type of measurement that is requested.""")
                
                self.__multiplicand = cps.Float("Multiply the above operand by",1,doc="""
                                            By what number would you like to multiply the above operand?""")
                
                self.__exponent = cps.Float("Raise the power of above operand by",1,doc="""
                                            To what power would you like to raise the above operand?""")
            
            @property
            def operand_choice(self):
                '''Either MC_IMAGE for image measurements or MC_OBJECT for object'''
                return self.__operand_choice
            
            @property
            def operand_objects(self):
                '''Get measurements from these objects'''
                return self.__operand_objects
            
            @property
            def operand_measurement(self):
                '''The measurement providing the value of the operand'''
                return self.__operand_measurement
            
            @property 
            def multiplicand(self):
                '''Premultiply the measurement by this value'''
                return self.__multiplicand
            
            @property
            def exponent(self):
                '''Raise the measurement to this power'''
                return self.__exponent
            
            @property
            def object(self):
                '''The name of the object for measurement or "Image"'''
                if self.operand_choice == MC_IMAGE:
                    return cpmeas.IMAGE
                else:
                    return self.operand_objects.value
                
            def object_fn(self):
                if self.__operand_choice == MC_IMAGE:
                    return cpmeas.IMAGE
                elif self.__operand_choice == MC_OBJECT:
                    return self.__operand_objects.value
                else:
                    raise NotImplementedError("Measurement type %s is not supported"%
                                              self.__operand_choice.value)
            def operand_name(self):
                '''A fancy name based on what operation is being performed'''
                if self.__index == 0:
                    return ("first operand" 
                            if self.__operation in (O_ADD, O_MULTIPLY) else
                            "minuend" if self.__operation == O_SUBTRACT else
                            "numerator")
                elif self.__index == 1:
                    return ("second operand" 
                            if self.__operation in (O_ADD, O_MULTIPLY) else
                            "subtrahend" if self.__operation == O_SUBTRACT  else 
                            "denominator")
            
            def operand_choice_text(self):
                return self.operand_text("Select the %s measurement type") 
            
            def operand_objects_text(self):
                return self.operand_text("Select the %s objects")
            
            def operand_text(self, format):
                return format % self.operand_name()
                
            def operand_measurement_text(self): 
                return self.operand_text("Select the %s measurement")
            
            def settings(self):
                '''The operand settings to be saved in the output file'''
                return [self.operand_choice, self.operand_objects, 
                        self.operand_measurement, self.multiplicand, self.exponent]
            
            def visible_settings(self):
                '''The operand settings to be displayed'''
                self.operand_choice.text = self.operand_choice_text()
                self.operand_objects.text = self.operand_objects_text()
                self.operand_measurement.text = self.operand_measurement_text()
                result = [self.operand_choice]
                result += ([self.operand_objects] if self.operand_choice == MC_OBJECT \
                          else [])
                result += [self.operand_measurement, self.multiplicand, self.exponent]
                return (result)
            
        self.output_feature_name = cps.Text("Name the output measurement",
                                            "Measurement",doc="""
                                            What do you want to call the measurement calculated by this module?""")
        
        self.operation = cps.Choice("Operation",
                                    O_ALL,doc="""
                                    What arithmetic operation would you like to perform? <i>None</i> is useful if 
                                    you simply want to select some of the later options in the module, such as multiplying
                                    or exponentiating your image by a constant.</li>""")
        
        self.operands = (Operand(0, self.operation), Operand(1, self.operation))
        
        self.spacer_1 = cps.Divider(line=True)
                
        self.spacer_2 = cps.Divider(line=True)
        
        self.spacer_3 = cps.Divider(line=True)
        
        self.wants_log = cps.Binary("Take log10 of result?", False,doc="""
                                    Do you want the log (base 10) of the result?""")
        
        self.final_multiplicand = cps.Float("Multiply the result by",1,doc="""
                                    <i>(Used only for operations other than None)</i><br>
                                    By what number would you like to multiply the result?""")
        
        self.final_exponent = cps.Float("Raise the power of result by",1,doc="""
                                    <i>(Used only for operations other than None)</i><br>
                                    To what power would you like to raise the result?""")
            
    def settings(self):
        result = [self.output_feature_name, self.operation] 
        result += self.operands[0].settings() + self.operands[1].settings()
        result += [self.wants_log, self.final_multiplicand, self.final_exponent]
        
        return (result)

    def post_pipeline_load(self, pipeline):
        '''Fixup any measurement names that might have been ambiguously loaded
        
        pipeline - for access to other module's measurements
        '''
        for operand in self.operands:
            measurement = operand.operand_measurement.value
            pieces = measurement.split('_')
            if len(pieces) == 4:
                try:
                    measurement = pipeline.synthesize_measurement_name(self,
                                                                       operand.object,
                                                                       pieces[0],
                                                                       pieces[1],
                                                                       pieces[2],
                                                                       pieces[3])
                    operand.operand_measurement.value = measurement
                except:
                    pass
                 
    def visible_settings(self):
        result = [self.output_feature_name, self.operation] + [self.spacer_1] 
        result += self.operands[0].visible_settings() + [self.spacer_2]
        if self.operation != O_NONE:
            result += self.operands[1].visible_settings() + [self.spacer_3]
        result += [self.wants_log]
        if self.operation != O_NONE:
            result += [self.final_multiplicand, self.final_exponent]

        return (result)
        

    def run(self, workspace):
        m = workspace.measurements
        values = []
        input_values = []
        has_image_measurement = any([operand.object == cpmeas.IMAGE
                                     for operand in self.operands])
        all_image_measurements = all([operand.object == cpmeas.IMAGE
                                     for operand in self.operands])
        all_object_names = list(set([operand.operand_objects.value
                                     for operand in self.operands
                                     if operand.object != cpmeas.IMAGE]))
        all_operands = self.operands
        
        if self.operation.value in (O_NONE):
            # Only operate on the first image/object
            all_operands  = all_operands[:1]
            
        for operand in all_operands:
            value = m.get_current_measurement(operand.object,operand.operand_measurement.value)
            # Copy the measurement (if it's right type) or else it gets altered by the operation
            if not np.isscalar(value):
                value = value.copy()
               
            if isinstance(value, str) or isinstance(value, unicode):
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError("Unable to use non-numeric value in measurement, %s"%operand.measurement.value)
            input_values.append(value)
            value *= operand.multiplicand.value
            value **= operand.exponent.value
            values.append(value)
        
        if (not has_image_measurement) and (self.operation.value not in (O_NONE)) and len(values[0]) != len(values[1]):
            raise ValueError("Incompatable objects: %s has %d objects and %s has %d objects"%
                             (self.operands[0].operand_objects.value, len(values[0]),
                              self.operands[1].operand_objects.value, len(values[1])))
        
        if self.operation == O_NONE:
            result = values[0]
        elif self.operation == O_ADD:
            result = values[0]+values[1]
        elif self.operation == O_SUBTRACT:
            result = values[0]-values[1]
        elif self.operation == O_MULTIPLY:
            result = values[0] * values[1]
        elif self.operation == O_DIVIDE:
            if np.isscalar(values[1]):
                if values[1] == 0:
                    if np.isscalar(values[0]):
                        result = np.NaN
                    else:
                        result = np.array([np.NaN] * len(values[0]))
                else:
                    result = values[0] / values[1]
            else:
                result = values[0] / values[1]
                result[values[1] == 0] = np.NaN
        else:
            raise NotImplementedError("Unsupported operation: %s"%self.operation.value)
        #
        # Post-operation rescaling
        #
        if self.wants_log.value:
            result = np.log10(result)
        if self.operation != O_NONE:
            result *= self.final_multiplicand.value
            # Handle NaNs with np.power instead of **
            result = np.power(result, self.final_exponent.value)
        feature = self.measurement_name()
        if all_image_measurements:
            m.add_image_measurement(feature, result)
        else:
            for object_name in all_object_names:
                m.add_measurement(object_name, feature, result)
                
        if workspace.frame is not None:
            workspace.display_data.statistics = [("Measurement name","Measurement type","Result")]
            workspace.display_data.statistics += [(self.output_feature_name.value, 
                                                   "Image" if all_image_measurements else "Object", 
                                                   "%.2f"%np.mean(result))]
            
    def run_as_data_tool(self, workspace):
        workspace.measurements.is_first_image = True
        image_set_count = workspace.measurements.image_set_count
        for i in range(image_set_count):
            self.run(workspace)
            if i < image_set_count - 1:
                workspace.measurements.next_image_set()
        
    def measurement_name(self):
        return "%s_%s" %(C_MATH,self.output_feature_name.value)
            
    def is_interactive(self):
        return False
    
    def display(self, workspace):
        figure = workspace.create_or_find_figure(title="CalculateMath, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,1))
        figure.subplot_table(0, 0, workspace.display_data.statistics,
                             ratio=(.25,.5,.25))
        
    def get_measurement_columns(self, pipeline):
        all_object_names = list(set([operand.operand_objects.value
                                     for operand in self.operands
                                     if operand.object != cpmeas.IMAGE]))
        if len(all_object_names):
            return [(name, self.measurement_name(), cpmeas.COLTYPE_FLOAT)
                    for name in all_object_names]
        else:
            return [(cpmeas.IMAGE, 
                     self.measurement_name(), 
                     cpmeas.COLTYPE_FLOAT)]

    def get_categories(self, pipeline, object_name):
        all_object_names = [operand.operand_objects.value
                            for operand in self.operands
                            if operand.object != cpmeas.IMAGE]
        if len(all_object_names):
            if object_name in all_object_names:
                return [C_MATH]
        elif object_name == cpmeas.IMAGE:
            return [C_MATH]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if category in self.get_categories(pipeline, object_name):
            return [self.output_feature_name.value]
        return []
    
    def validate_module(self, pipeline):
        '''Do further validation on this module's settings
        
        pipeline - this module's pipeline
        
        Check to make sure the output measurements aren't duplicated
        by prior modules.
        '''
        all_object_names = [operand.operand_objects.value
                            for operand in self.operands
                            if operand.object != cpmeas.IMAGE]
        for module in pipeline.modules():
            if module.module_num == self.module_num:
                break
            for name in all_object_names:
                features = module.get_measurements(pipeline, name, C_MATH)
                if self.output_feature_name.value in features:
                    raise cps.ValidationError(
                        'The feature, "%s", was already defined in module # %d'%
                        (self.output_feature_name.value, module.module_num),
                        self.output_feature_name)
        
    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if (from_matlab and variable_revision_number == 6 and 
            module_name == 'CalculateRatios'):
            ratio_name, \
            object_name_1, category_1, feature_1, image_1, scale_1, \
            object_name_2, category_2, feature_2, image_2, scale_2, \
            log_choice = setting_values
            setting_values = [ 
                object_name_1, category_1, feature_1, image_1, scale_1,
                object_name_2, category_2, feature_2, image_2, scale_2,
                log_choice, "1", "1", "1", "1", "Divide", ratio_name]
            variable_revision_number = 6
            module_name = 'CalculateMath'
        if (from_matlab and variable_revision_number == 6 and 
            module_name == 'CalculateMath'):
            new_setting_values = [setting_values[16], # output feature name
                                  setting_values[15]] # operation
            for i,multiply_factor_idx in ((0,11),(5,12)):
                object_name = setting_values[i]
                category = setting_values[i+1]
                feature = setting_values[i+2]
                measurement_image = setting_values[i+3]
                scale = setting_values[i+4]
                measurement = category+'_'+feature
                if len(measurement_image):
                    measurement += '_' + measurement_image
                if len(scale):
                    measurement += '_' + scale
                object_choice = (MC_IMAGE if object_name == cpmeas.IMAGE
                                 else MC_OBJECT) 
                new_setting_values += [object_choice,
                                       object_name,
                                       measurement,
                                       setting_values[multiply_factor_idx],
                                       "1"] # exponent
            new_setting_values += [setting_values[10], # wants log
                                   setting_values[14], # final multiplier
                                   setting_values[13]] # final exponent
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab

