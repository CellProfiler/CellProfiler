function [CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles)

CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));