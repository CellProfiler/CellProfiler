'''calculatemath.py - the CalculateMath module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps

O_MULTIPLY = "Multiply"
O_DIVIDE = "Divide"
O_ADD = "Add"
O_SUBTRACT = "Subtract"

O_ALL = [O_MULTIPLY, O_DIVIDE, O_ADD, O_SUBTRACT]

MC_IMAGE = cpmeas.IMAGE
MC_OBJECT = "Object"
MC_ALL = [MC_IMAGE, MC_OBJECT]

class CalculateMath(cpm.CPModule):
    '''SHORT DESCRIPTION:
This module can take measurements produced by previous modules and
performs basic arithmetic operations.
*************************************************************************

The arithmetic operations available in this module include addition,
subtraction, multiplication and division. The operation can be chosen
by adjusting the operations setting. The resulting data can also be
logged or raised to a power. This data can then be used in other
calculations and can be used in Classify Objects.

This module currently works on an object-by-object basis (it calculates
the requested operation for each object) but can also apply the operation
for measurements made for entire images.

Saving:
The math measurements are stored as 'Math_...'. If both measures are 
image-based, then a single calculation (per cycle) will be stored as 'Image' data.  
If one measure is object-based and one image-based, then the calculations will
be stored associated with the object, one calculation per object.  If both are 
objects, then the calculations are stored with both objects.

Note: If you want to use the output of this module in a subsequesnt
calculation, we suggest you specify the output name rather than use
Automatic naming.

See also CalculateRatios, all Measure modules.
'''

    module_name = "CalculateMath"
    category="Measurement"
    variable_revision_number = 1
    
    def create_settings(self):
        class Operand(object):
            '''Represents the collection of settings needed by each operand'''
            def __init__(self, index, operation):
                self.__index = index
                self.__operation = operation
                self.__operand_choice = cps.Choice(self.operand_choice_text(), MC_ALL)
                self.__operand_objects = cps.ObjectNameSubscriber(self.operand_objects_text(),"None")
                self.__operand_measurement = cps.Measurement(self.operand_measurement_text(),
                                                             self.object_fn)
                self.__multiplicand = cps.Float("What number would you like to multiply the above operand by?",1)
                self.__exponent = cps.Float("What power would you like to raise the above operand to?",1)
            
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
                return self.operand_text("Is the %s an image or object measurement?") 
            
            def operand_objects_text(self):
                return self.operand_text("Which objects do you want to measure for the %s?")
            
            def operand_text(self, format):
                return format % self.operand_name()
                
            def operand_measurement_text(self): 
                return self.operand_text("What measurement do you want to use for the %s?")
            
            def settings(self):
                '''The operand settings to be saved in the output file'''
                return [self.operand_choice, self.operand_objects, 
                        self.operand_measurement, self.multiplicand, self.exponent]
            
            def visible_settings(self):
                '''The operand settings to be displayed'''
                self.operand_choice.text = self.operand_choice_text()
                self.operand_objects.text = self.operand_objects_text()
                self.operand_measurement.text = self.operand_measurement_text()
                return ([self.operand_choice] +
                        ([self.operand_objects] if self.operand_choice == MC_OBJECT
                          else []) +
                        [self.operand_measurement,self.multiplicand, self.exponent])
            
        self.output_feature_name = cps.Text("What do you want to call the measurement calculated by this module?",
                                            "Measurement")
        self.operation = cps.Choice("What operation would you like to perform?",
                                    O_ALL)
        self.operands = (Operand(0, self.operation), Operand(1, self.operation))
        self.wants_log = cps.Binary("Do you want the log (base 10) of the ratio?", False)
        self.final_multiplicand = cps.Float("What number would you like to multiply the result by?",1)
        self.final_exponent = cps.Float("What power would you like to raise the result to?",1)
            
    def settings(self):
        return ([self.output_feature_name, self.operation] +
                self.operands[0].settings() + self.operands[1].settings() + 
                [self.wants_log, self.final_multiplicand, self.final_exponent])

    def on_post_load(self, pipeline):
        '''Fixup any measurement names that might have been ambiguously loaded
        
        pipeline - for access to other module's measurements
        '''
        for operand in self.operands:
            measurement = operand.operand_measurement.value
            pieces = measurement.split('_')
            if len(pieces) == 4:
                measurement = pipeline.synthesize_measurement_name(self,
                                                                   operand.object,
                                                                   pieces[0],
                                                                   pieces[1],
                                                                   pieces[2],
                                                                   pieces[3])
                operand.operand_measurement.value = measurement
                 
    def visible_settings(self):
        return ([self.output_feature_name, self.operation] +
                self.operands[0].visible_settings() + 
                self.operands[1].visible_settings() + 
                [self.wants_log, self.final_multiplicand, self.final_exponent])
        

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
        for operand in self.operands:
            value = m.get_current_measurement(operand.object,
                                              operand.operand_measurement.value)
            if isinstance(value, str) or isinstance(value, unicode):
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError("Unable to use non-numeric value in measurement, %s"%operand.measurement.value)
            input_values.append(value)
            value *= operand.multiplicand.value
            value **= operand.exponent.value
            values.append(value)
        
        if (not has_image_measurement) and len(values[0]) != len(values[1]):
            raise ValueError("Incompatable objects: %s has %d objects and %s has %d objects"%
                             (operands[0].operand_objects.value, len(values[0]),
                              operands[1].operand_objects.value, len(values[1])))
        
        if self.operation == O_ADD:
            result = values[0]+values[1]
        elif self.operation == O_SUBTRACT:
            result = values[0]-values[1]
        elif self.operation == O_MULTIPLY:
            result = values[0] * values[1]
        elif self.operation == O_DIVIDE:
            result = values[0] / values[1]
        else:
            raise NotImplementedError("Unsupported operation: %s"%self.operation.value)
        #
        # Post-operation rescaling
        #
        if self.wants_log.value:
            result = np.log10(result)
        result *= self.final_multiplicand.value
        result **= self.final_exponent.value
        feature = self.measurement_name()
        if all_image_measurements:
            m.add_image_measurement(feature, result)
        else:
            for object_name in all_object_names:
                m.add_measurement(object_name, feature, result)

    def measurement_name(self):
        return "Math_"+self.output_feature_name.value
            
    def get_measurement_columns(self, pipeline):
        all_object_names = [operand.operand_objects.value
                            for operand in self.operands
                            if operand.object != cpmeas.IMAGE]
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
                return ["Math"]
        elif object_name == cpmeas.IMAGE:
            return ["Math"]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if category in self.get_categories(pipeline, object_name):
            return [self.output_feature_name.value]
        return []
    
    def backwards_compatibilize(self, setting_values, variable_revision_number,
                                module_name, from_matlab):
        if from_matlab and variable_revision_number == 6:
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

