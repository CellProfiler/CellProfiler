"""VariableChoices.py - representation framework for the set of values available to a variable
    $Revision$
    
    Many variables have constraints that limit their values to one of
    a set of choices. For some of these variables, the user can choose
    a value "outside of the box" either because the user knows that a
    future edit will align the value with the model or because the
    model doesn't accurately capture all choices.
    
    In any case, the AbstractVariableChoices class and derived classes
    model the choices available to a variable.
"""

import CellProfiler.Variable
import CellProfiler.Pipeline

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
    
    def GetChoices(self,variable):
        """Return the choices available to a particular variable instance
        
        """
        raise NotImplementedError('GetChoices needs to be implemented by all derived classes')
    
    def CanAcceptOther(self):
        """Return True if the user can enter a choice that isn't in the list, False if the user is constrained to the list.
        
        """
        raise NotImplementedError('CanAcceptOther needs to be implemented by all derived classes')
    
    def CanChange(self):
        """Return true if the choices are not static
        
        """
        raise NotImplementedError('CanChange needs to be implemented by all derived classes')

class StaticVariableChoices(AbstractVariableChoices):
    """Represents a constrained list of choices that does not vary over the variable's lifetime
    
    """
    def __init__(self,choice_list):
        AbstractVariableChoices.__init__(self)
        self.__choice_list = choice_list
    
    def GetChoices(self,variable):
        return self.__choice_list
    
    def CanAcceptOther(self):
        """Don't allow the user to pick a value that's not on the list - that value will never be valid
        
        """
        return False
    
    def CanChange(self):
        """The list will never change - a listener is not needed
        
        """
        return False 

class StaticVariableChoicesAllowingOther(StaticVariableChoices):
    """Represents a constrained list, but the user can enter secret values that aren't captured by the model
    
    """
    def __init__(self,choice_list):
        StaticVariableChoices.__init__(self,choice_list)
        
    def CanAcceptOther(self):
        return True

class AbstractMutableVariableChoices(AbstractVariableChoices):
    """Abstract class representing variable choices that can change if other variables' values change
    
    """
    def __init__(self,pipeline):
        AbstractVariableChoices.__init__(self)
        self.__listeners = []
        self.__pipeline = pipeline
        pipeline.AddListener(self.__OnPipelineEvent)
        
    def AddListener(self,listener):
        """Add a listener that will be notified if an event occurs that might change the list of choices
        
        """
        self.__listeners.append(listener)
        
    def RemoveListener(self,listener):
        """Remove a previously added listener
        
        """
        self.__listeners.remove(listener)
    
    def Notify(self,event):
        """Notify all listeners of some event
        
        """
        for listener in self.__listeners:
            listener(self,event)
            
    def CanChange(self):
        """Return true - the list is mutable and can change
        
        """
        return True
    
    def CanAcceptOther(self):
        """Return true - if the list can change, then a value that will be valid later should be allowed
        
        """
        return True
    
    def __OnPipelineEvent(self,pipeline,event):
        """Assume that a pipeline event (such as adding a module) will require an update to the choice list
        
        """
        self.Notify(event)
        
class InfoGroupVariableChoices(AbstractMutableVariableChoices):
    """Variable choices based on parsing %infotype groups out of the .m files
    
    """
    def __init__(self,pipeline):
        AbstractMutableVariableChoices.__init__(self,pipeline)
        self.__indep_variables = []
        
    def AddIndepVariable(self,variable):
        """Add a variable marked "indep" to the group
        
        Add a variable marked "indep" to the group. Listen for
        changes in that variable - these may indicate changes in available choices.
        """
        self.__indep_variables.append(variable)
        variable.AddListener(self.__OnVariableEvent)
    
    def __OnVariableEvent(self,sender,event):
        """Respond to an event on an independent variable
        
        Respond to an event on an independent variable.
        Remove a deleted variable from the variable list and notify listeners.
        Notify listeners after a value change.
        """
        if isinstance(event,CellProfiler.Variable.DeleteVariableEvent):
            self.__indep_variables.remove(sender)
            self.Notify(event)
        elif isinstance(event,CellProfiler.Variable.AfterChangeVariableEvent):
            self.Notify(event)
    
    def GetChoices(self,variable):
        """Report the values of the independent variables that appear before this one in the pipeline.
        
        """
        choices = set()
        for indep in self.__indep_variables:
            if (indep.Module().ModuleNum() < variable.Module().ModuleNum() and
                indep.Value != CellProfiler.Variable.DO_NOT_USE):
                choices.add(indep.Value)
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
        object_variable.AddListener(self.__OnVariableEvent)
        AbstractMutableVariableChoices.__init__(self,pipeline)

    def __OnVariableEvent(self,sender,event):
        """Respond to an event on the object variable
        
        Notify listeners after a value change.
        """
        if isinstance(event,CellProfiler.Variable.AfterChangeVariableEvent):
            self.Notify(event)
    
    def GetChoices(self,variable):
        """Report the values of the independent variables that appear before this one in the pipeline.
        
        """
        choices = [] 
        object_name = self.__object_variable.Value
        for ModuleNum in range(1,variable.Module().ModuleNum()):
            module = self.__pipeline.Module(ModuleNum)
            for choice in module.GetCategories(self.__pipeline,object_name):
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
        object_variable.AddListener(self.__OnVariableEvent)
        self.__category_variable = category_variable
        self.__category_variable.AddListener(self.__OnVariableEvent)
        AbstractMutableVariableChoices.__init__(self,pipeline)

    def __OnVariableEvent(self,sender,event):
        """Respond to an event on the object variable
        
        Notify listeners after a value change.
        """
        if isinstance(event,CellProfiler.Variable.AfterChangeVariableEvent):
            self.Notify(event)
    
    def GetChoices(self,variable):
        """Report the possible measurements for this variable
        
        """
        choices = []
        object_name = self.__object_variable.Value
        category = self.__category_variable.Value
        for ModuleNum in range(1,variable.Module().ModuleNum()):
            module = self.__pipeline.Module(ModuleNum)
            for choice in module.GetMeasurements(self.__pipeline,object_name,category):
                if not choice in choices:
                    choices.append(choice)
        return choices
        
