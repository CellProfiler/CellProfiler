function [CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles)

%%% Reads the current module number and module name, because this is needed
%%% to find the variable values that the user entered and to report the
%%% proper module name in any error messages.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));