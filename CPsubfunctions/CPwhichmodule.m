function [CurrentModuleNum, ModuleName] = CPwhichmodule(handles)

CurrentModuleNum = str2double(handles.Current.CurrentModuleNumber);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));