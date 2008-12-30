"""VariableChoices.py - representation framework for the set of values available to a variable
    
    Many variables have constraints that limit their values to one of
    a set of choices. For some of these variables, the user can choose
    a value "outside of the box" either because the user knows that a
    future edit will align the value with the model or because the
    model doesn't accurately capture all choices.
    
    In any case, the AbstractVariableChoices class and derived classes
    model the choices available to a variable.
"""
__version__="$Revision$"
import cellprofiler.variable
import cellprofiler.pipeline

class AbstractVariableChoices:
    """Represents a list of variable choices that might be tailored to a particular variable
    
    Represents a list of variable choices that might be tailored to a particular variable.
    Maybe you'd like to display the choices available to a variable in a drop-down or menu.
    What do you have to know in order to do this?
    * What choices are available for that variable - these may depend on the instance
      of the variable and its relation to other variables, for instance, what generators
      of objects precede the variable in the pipeline
    * Whether the list is constrained or whether the user can draw outside of the lines
    * When the choices have possibly changed
    """
    def __init__(self):
        pass
    
    def get_choices(self,variable):
        """Return the choices available to a particular variable instance
        
        """
        raise NotImplementedError('GetChoices needs to be implemented by all derived classes')
    
    def can_accept_other(self):
        """Return True if the user can enter a choice that isn't in the list, False if the user is constrained to the list.
        
        """
        raise NotImplementedError('CanAcceptOther needs to be implemented by all derived classes')
    
    def can_change(self):
        """Return true if the choices are not static
        
        """
        raise NotImplementedError('CanChange needs to be implemented by all derived classes')

class StaticVariableChoices(AbstractVariableChoices):
    """Represents a constrained list of choices that does not vary over the variable's lifetime
    
    """
    def __init__(self,choice_list):
        AbstractVariableChoices.__init__(self)
        self.__choice_list = choice_list
    
    def get_choices(self,variable):
        return self.__choice_list
    
    def can_accept_other(self):
        """Don't allow the user to pick a value that's not on the list - that value will never be valid
        
        """
        return False
    
    def can_change(self):
        """The list will never change - a listener is not needed
        
        """
        return False 

class StaticVariableChoicesAllowingOther(StaticVariableChoices):
    """Represents a constrained list, but the user can enter secret values that aren't captured by the model
    
    """
    def __init__(self,choice_list):
        StaticVariableChoices.__init__(self,choice_list)
        
    def can_accept_other(self):
        return True

class AbstractMutableVariableChoices(AbstractVariableChoices):
    """Abstract class representing variable choices that can change if other variables' values change
    
    """
    def __init__(self,pipeline):
        AbstractVariableChoices.__init__(self)
        self.__listeners = []
        self.__pipeline = pipeline
        pipeline.add_listener(self.__on_pipeline_event)
        
    def add_listener(self,listener):
        """Add a listener that will be notified if an event occurs that might change the list of choices
        
        """
        self.__listeners.append(listener)
        
    def remove_listener(self,listener):
        """Remove a previously added listener
        
        """
        self.__listeners.remove(listener)
    
    def notify(self,event):
        """Notify all listeners of some event
        
        """
        for listener in self.__listeners:
            listener(self,event)
            
    def can_change(self):
        """Return true - the list is mutable and can change
        
        """
        return True
    
    def can_accept_other(self):
        """Return true - if the list can change, then a value that will be valid later should be allowed
        
        """
        return True
    
    def __on_pipeline_event(self,pipeline,event):
        """Assume that a pipeline event (such as adding a module) will require an update to the choice list
        
        """
        self.notify(event)
        
class InfoGroupVariableChoices(AbstractMutableVariableChoices):
    """Variable choices based on parsing %infotype groups out of the .m files
    
    """
    def __init__(self,pipeline):
        AbstractMutableVariableChoices.__init__(self,pipeline)
        self.__indep_variables = []
        
    def add_indep_variable(self,variable):
        """Add a variable marked "indep" to the group
        
        Add a variable marked "indep" to the group. Listen for
        changes in that variable - these may indicate changes in available choices.
        """
        self.__indep_variables.append(variable)
        variable.add_listener(self.__on_variable_event)
    
    def __on_variable_event(self,sender,event):
        """Respond to an event on an independent variable
        
        Respond to an event on an independent variable.
        Remove a deleted variable from the variable list and notify listeners.
        Notify listeners after a value change.
        """
        if isinstance(event,cellprofiler.variable.DeleteVariableEvent):
            self.__indep_variables.remove(sender)
            self.Notify(event)
        elif isinstance(event,cellprofiler.variable.AfterChangeVariableEvent):
            self.notify(event)
    
    def get_choices(self,variable):
        """Report the values of the independent variables that appear before this one in the pipeline.
        
        """
        choices = set()
        for indep in self.__indep_variables:
            if (indep.module().module_num < variable.module().module_num and
                indep.value != cellprofiler.variable.DO_NOT_USE):
                choices.add(indep.value)
        choices = list(choices)
        choices.sort()
        return choices
    
class CategoryVariableChoices(AbstractMutableVariableChoices):
    """Variable choices based on the categories that modules produce
    
    """
    def __init__(self,pipeline,object_variable):
        """Initialize with the parent pipeline and the variable
        which holds the name of the object being measured
        """
        self.__pipeline = pipeline
        self.__object_variable = object_variable
        object_variable.add_listener(self.__on_variable_event)
        AbstractMutableVariableChoices.__init__(self,pipeline)

    def __OnVariableEvent(self,sender,event):
        """Respond to an event on the object variable
        
        Notify listeners after a value change.
        """
        if isinstance(event,CellProfiler.Variable.AfterChangeVariableEvent):
            self.notify(event)
    
    def GetChoices(self,variable):
        """Report the values of the independent variables that appear before this one in the pipeline.
        
        """
        choices = [] 
        object_name = self.__object_variable.Value
        for ModuleNum in range(1,variable.module().module_num):
            module = self.__pipeline.module(module_num)
            for choice in module.get_categories(self.__pipeline,object_name):
                if not choice in choices:
                    choices.append(choice)
        return choices

class MeasurementVariableChoices(AbstractMutableVariableChoices):
    """Variable choices based on the measurements that modules produce
    
    """
    def __init__(self,pipeline,object_variable, category_variable):
        """Initialize to supply measurement choices
        
        pipeline - the pipeline containing everything
        object_variable - the variable that supplies the name of the object being measured (or 'Image')
        category_variable - the variable that holds the measurement category
        """
        self.__pipeline = pipeline
        self.__object_variable = object_variable
        object_variable.add_listener(self.__on_variable_event)
        self.__category_variable = category_variable
        self.__category_variable.add_listener(self.__on_variable_event)
        AbstractMutableVariableChoices.__init__(self,pipeline)

    def __on_variable_event(self,sender,event):
        """Respond to an event on the object variable
        
        Notify listeners after a value change.
        """
        if isinstance(event,cellprofiler.variable.AfterChangeVariableEvent):
            self.notify(event)
    
    def get_choices(self,variable):
        """Report the possible measurements for this variable
        
        """
        choices = []
        object_name = self.__object_variable.Value
        category = self.__category_variable.Value
        for module_num in range(1,variable.module().module_num):
            module = self.__pipeline.module(module_num)
            for choice in module.get_measurements(self.__pipeline,object_name,category):
                if not choice in choices:
                    choices.append(choice)
        return choices
        
