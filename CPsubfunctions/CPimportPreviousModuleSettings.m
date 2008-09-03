function [Settings,SavedVarRevNum,IsModuleModified] = CPimportPreviousModuleSettings(Settings,CurrentModuleName,ModuleNum,Skipped,SavedVarRevNum)

% This function attempts to import the settings of older modules into newer
% ones, basically by reordering VariableValues, VariableInfoTypes, and
% updating NumbersOfVariables and SavedVarRevNum

IsModuleModified = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Changes to LoadImages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp('LoadImages',CurrentModuleName) 
    if (SavedVarRevNum == 1)
        ImageOrMovie = Settings.VariableValues{ModuleNum-Skipped,11};
        if strcmp(ImageOrMovie,'Image')
            new_variablevalue = 'individual images';
        else
            if strcmp(Settings.VariableValues{ModuleNum-Skipped,12},'avi')
                new_variablevalue = 'avi movies';
            elseif strcmp(Settings.VariableValues{ModuleNum-Skipped,12},'stk')
                new_variablevalue = 'stk movies';
            end
        end
        Settings.VariableValues{ModuleNum-Skipped,11} = new_variablevalue;
        Settings.VariableValues{ModuleNum-Skipped,12} = Settings.VariableValues{ModuleNum-Skipped,13};
        Settings.VariableValues{ModuleNum-Skipped,13} = Settings.VariableValues{ModuleNum-Skipped,14};   
        SavedVarRevNum = 2;
        IsModuleModified = true;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Changes to RescaleIntensity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'RescaleIntensity')
    if SavedVarRevNum == 2      % RescaleIntensity.m got two new arguments
        Settings.VariableValues{ModuleNum-Skipped,10} = Settings.VariableValues{ModuleNum-Skipped,8};
        Settings.VariableInfoTypes{ModuleNum-Skipped,10} = Settings.VariableInfoTypes{ModuleNum-Skipped,8};
        Settings.VariableValues{ModuleNum-Skipped,9} = Settings.VariableValues{ModuleNum-Skipped,7};
        Settings.VariableInfoTypes{ModuleNum-Skipped,9} = Settings.VariableInfoTypes{ModuleNum-Skipped,7};
        Settings.VariableValues{ModuleNum-Skipped,8} = Settings.VariableValues{ModuleNum-Skipped,6};
        Settings.VariableInfoTypes{ModuleNum-Skipped,8} = Settings.VariableInfoTypes{ModuleNum-Skipped,6};
        Settings.NumbersOfVariables(ModuleNum-Skipped) = Settings.NumbersOfVariables(ModuleNum-Skipped) + 2;
        SavedVarRevNum = 3;
        IsModuleModified = true;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Changes to SaveImages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'SaveImages')
    if SavedVarRevNum == 12     % SaveImages.m got one new argument
        % Re-create subdirectories? Default to No.
        Settings.VariableValues{ModuleNum-Skipped,14} = 'No';
        % Move overwrite warning down one
        Settings.VariableValues{ModuleNum-Skipped,15} = 'n/a';
        Settings.NumbersOfVariables(ModuleNum-Skipped) = Settings.NumbersOfVariables(ModuleNum-Skipped) + 1;
        SavedVarRevNum = 13;
        IsModuleModified = true;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Changes to ExportToDatabase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'ExportToDatabase')
    if SavedVarRevNum == 4      % ExportToDatabase.m got one new argument
        % Create CPA properties file? Default to No.
        Settings.VariableValues{ModuleNum-Skipped,6} = 'No';
        Settings.NumbersOfVariables(ModuleNum-Skipped) = Settings.NumbersOfVariables(ModuleNum-Skipped) + 1;
        SavedVarRevNum = 5;
        IsModuleModified = true;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Standardization of non-used parameter text placeholders
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = ismember(cellstr(lower(char(Settings.VariableValues{ModuleNum-Skipped,:}))),lower({'NO FILE LOADED','Leave this blank','Do not load','Do not save','/'}));
if any(idx),
    [Settings.VariableValues{ModuleNum-Skipped,idx}] = deal('Do not use');
    CPwarndlg('Note: Placeholder text for optional/unused entries have been updated to the standardized value "Do not use." Please see the Developer notes under "Settings" for more details.','LoadPipelines: Some entries updated','modal');
end
