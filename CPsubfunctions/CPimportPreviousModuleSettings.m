function [Settings,SavedVarRevNum,IsModuleModified,NeedsPlaceholderUpdateMsg,CurrentModuleName] = CPimportPreviousModuleSettings(Settings,CurrentModuleName,ModuleNum,Skipped,SavedVarRevNum)

% This function attempts to import the settings of older modules into newer
% ones, basically by reordering VariableValues, VariableInfoTypes, and
% updating NumbersOfVariables and SavedVarRevNum

% $Revision$

IsModuleModified = false;
NeedsPlaceholderUpdateMsg = false;

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
    if (SavedVarRevNum == 2)
        Settings.VariableValues{ModuleNum-Skipped,14} = 'grayscale';
        %%% The last variable is bogus...
        Settings.VariableValues{ModuleNum-Skipped,15} = [];
        Settings.NumbersOfVariables(ModuleNum-Skipped) = 15;
        SavedVarRevNum = 3;
        IsModuleModified = true;
    end
        
    if (SavedVarRevNum == 3)    % File text exclusion added
        for i = Settings.NumbersOfVariables(ModuleNum-Skipped):-1:12,
            Settings.VariableValues{ModuleNum-Skipped,i} = Settings.VariableValues{ModuleNum-Skipped,i-1};
        end
        for i = length(Settings.VariableInfoTypes(ModuleNum-Skipped,:)):-1:12,
            Settings.VariableInfoTypes{ModuleNum-Skipped,i} = Settings.VariableInfoTypes{ModuleNum-Skipped,i-1};
        end

        Settings.VariableValues{ModuleNum-Skipped,11} = 'Do not use';
        Settings.VariableInfoTypes{ModuleNum-Skipped,11} = [];
        
        %%% The last variable is bogus...
        Settings.NumbersOfVariables(ModuleNum-Skipped) = 16;
        Settings.VariableValues{ModuleNum-Skipped,16} = [];
        
        SavedVarRevNum = 4;
        IsModuleModified = true;
    end
    if (SavedVarRevNum == 4) % Added subfolder selection and file QA
        Settings.VariableValues{ModuleNum-Skipped,16} = 'No';
        Settings.VariableValues{ModuleNum-Skipped,17} = 'No';
        Settings.VariableValues{ModuleNum-Skipped,18} = []; %%% Dummy value for last variable
        SavedVarRevNum = 5;
        Settings.NumbersOfVariables(ModuleNum-Skipped)= 18;
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
% Changes to RescaleIntensity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'RescaleIntensity')
    if SavedVarRevNum == 3      % RescaleIntensity got one new argument, but at the end.
        Settings.VariableValues{ModuleNum-Skipped,11} = '';
        Settings.NumbersOfVariables(ModuleNum-Skipped) = Settings.NumbersOfVariables(ModuleNum-Skipped) + 1;
        SavedVarRevNum = 4;
        IsModuleModified = true;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Changes to FilterByObjectMeasurement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'FilterByObjectMeasurement')
    if SavedVarRevNum == 5      % Swapping arguments 1 and 2, so we're able to use the measurment popup fill-in
        [Settings.VariableValues{ModuleNum-Skipped,[1 2]}] = deal(Settings.VariableValues{ModuleNum-Skipped,[2 1]});
        [Settings.VariableInfoTypes{ModuleNum-Skipped,[1 2]}] = deal(Settings.VariableInfoTypes{ModuleNum-Skipped,[2 1]});
        SavedVarRevNum = 6;
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
    if SavedVarRevNum == 13    % Picky revision for specific use of "\"
        
        for fixup=[3,12]
            if strcmp(Settings.VariableValues{ModuleNum-Skipped,fixup},'\')
                Settings.VariableValues{ModuleNum-Skipped,fixup} = 'Do not use';
                NeedsPlaceholderUpdateMsg = true;
                IsModuleModified = true;
            end
        end
        SavedVarRevNum = 14;
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
    if SavedVarRevNum == 5      % Create CPA properties file - Yes - V1.0 format or V2.0 format
        if strncmp(Settings.VariableValues{ModuleNum-Skipped,6},'Y',1)
            Settings.VariableValues{ModuleNum-Skipped,6} = 'Yes - V1.0 format';
        end
        SavedVarRevNum = 6;
        IsModuleModified = true;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete module: Flip
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'Flip')
    CurrentModuleName = 'FlipAndRotate';
    Settings.ModuleNames{ModuleNum-Skipped}=CurrentModuleName;
    Settings.VariableValues{ModuleNum-Skipped,5}='None'; %Don't rotate
    Settings.VariableValues{ModuleNum-Skipped,6}='Yes';  %Crop
    Settings.VariableValues{ModuleNum-Skipped,7}='Individually';
    Settings.VariableValues{ModuleNum-Skipped,8}='horizontally';
    Settings.VariableValues{ModuleNum-Skipped,9}='1,1';
    Settings.VariableValues{ModuleNum-Skipped,10}='100,5';
    Settings.VariableValues{ModuleNum-Skipped,11}='5';
    Settings.NumbersOfVariables(ModuleNum-Skipped)=11;
    SavedVarRevNum = 1;
    IsModuleModified = true;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete module: Rotate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'Rotate')
    CurrentModuleName = 'FlipAndRotate';
    Settings.ModuleNames{ModuleNum-Skipped}=CurrentModuleName;
    for i=9:-1:3
        Settings.VariableValues{ModuleNum-Skipped,i+2} =...
            Settings.VariableValues{ModuleNum-Skipped,i};
    end
    Settings.VariableValues{ModuleNum-Skipped,3}='No';
    Settings.VariableValues{ModuleNum-Skipped,4}='No';
    Settings.NumbersOfVariables(ModuleNum-Skipped)=11;
    SavedVarRevNum = 1;
    IsModuleModified = true;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Changes to SmoothOrEnhance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName,'SmoothOrEnhance')
    if SavedVarRevNum == 4
        Settings.VariableValues{ModuleNum-Skipped,7}='16.0';
        Settings.VariableValues{ModuleNum-Skipped,8}='0.1';
        Settings.NumbersOfVariables(ModuleNum-Skipped)=8;
        SavedVarRevNum = 5;
        IsModuleModified = true;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete: SmoothKeepingEdges
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName,'SmoothKeepingEdges')
    CurrentModuleName='SmoothOrEnhance';
    Settings.ModuleNames{ModuleNum-Skipped}=CurrentModuleName;
    Settings.VariableValues{ModuleNum-Skipped,7} = Settings.VariableValues{ModuleNum-Skipped,4};
    Settings.VariableValues{ModuleNum-Skipped,8} = Settings.VariableValues{ModuleNum-Skipped,5};
    Settings.VariableValues{ModuleNum-Skipped,3} = 'Smooth Keeping Edges';
    Settings.VariableValues{ModuleNum-Skipped,4} = 'Automatic';
    Settings.VariableValues{ModuleNum-Skipped,5} = 'Do not use';
    Settings.VariableValues{ModuleNum-Skipped,6} = 'No';
    Settings.NumbersOfVariables(ModuleNum-Skipped)=8;
    SavedVarRevNum = 5;
    IsModuleModified = true;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Standardization of non-used parameter text placeholders
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = ismember(cellstr(lower(char(Settings.VariableValues{ModuleNum-Skipped,:}))),lower({'NO FILE LOADED','Leave this blank','Leave this black','None','Do not load','Do not save','/'}));
if any(idx),
    [Settings.VariableValues{ModuleNum-Skipped,idx}] = deal('Do not use');
    NeedsPlaceholderUpdateMsg = true;
end
% if NeedsPlaceholderUpdateMsg
%     CPwarndlg('Note: Placeholder text for optional/unused entries have been updated to the standardized value "Do not use." Please see the Developer notes under "Settings" for more details.','LoadPipelines: Some entries updated');
% end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete module: Subtract
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'Subtract')
    CurrentModuleName = 'ImageMath';
    Settings.ModuleNames{ModuleNum-Skipped}=CurrentModuleName;
    Settings.VariableValues{ModuleNum-Skipped,1} = Settings.VariableValues{ModuleNum-Skipped,1};
    Settings.VariableValues{ModuleNum-Skipped,2} = Settings.VariableValues{ModuleNum-Skipped,2};
    Settings.VariableValues{ModuleNum-Skipped,6} = Settings.VariableValues{ModuleNum-Skipped,5};
    Settings.VariableValues{ModuleNum-Skipped,5} = Settings.VariableValues{ModuleNum-Skipped,4};
    Settings.VariableValues{ModuleNum-Skipped,6} = Settings.VariableValues{ModuleNum-Skipped,5};
    Settings.VariableValues{ModuleNum-Skipped,10} = Settings.VariableValues{ModuleNum-Skipped,6};
    Settings.VariableValues{ModuleNum-Skipped,4} = 'Subtract';
    Settings.VariableValues{ModuleNum-Skipped,12} = Settings.VariableValues{ModuleNum-Skipped,3};
    Settings.NumbersOfVariables(ModuleNum-Skipped)=12;
    IsModuleModified = true;
    SavedVarRevNum = 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete module: Average
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'Average')
    if strcmp(Settings.VariableValues{ModuleNum-Skipped,3},'Pipeline')
        CPwarndlg('Note: The Pipeline option previously available in the Average module is now available using CorrectIllumination_Calculate module and selecting the option "(For All mode only) What do you want to call the averaged image (prior to dilation or smoothing)?  (This is an image produced during the calculations - it is typically not needed for downstream modules)"  This will be an average over all images.');
    end
    CurrentModuleName = 'ImageMath';
    Settings.ModuleNames{ModuleNum-Skipped}=CurrentModuleName;
    Settings.VariableValues{ModuleNum-Skipped,1} = Settings.VariableValues{ModuleNum-Skipped,1};
    Settings.VariableValues{ModuleNum-Skipped,2} = Settings.VariableValues{ModuleNum-Skipped,2};
    Settings.VariableValues{ModuleNum-Skipped,3} = Settings.VariableValues{ModuleNum-Skipped,3};
    SavedVarRevNum = 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete module: Combine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'Combine')
    CurrentModuleName = 'ImageMath';
    Settings.ModuleNames{ModuleNum-Skipped}=CurrentModuleName;
    Settings.VariableValues{ModuleNum-Skipped,1} = Settings.VariableValues{ModuleNum-Skipped,1};
    Settings.VariableValues{ModuleNum-Skipped,2} = Settings.VariableValues{ModuleNum-Skipped,2};
    Settings.VariableValues{ModuleNum-Skipped,3} = Settings.VariableValues{ModuleNum-Skipped,3};
    Settings.VariableValues{ModuleNum-Skipped,7} = Settings.VariableValues{ModuleNum-Skipped,7};
    Settings.VariableValues{ModuleNum-Skipped,6} = Settings.VariableValues{ModuleNum-Skipped,6};
    Settings.VariableValues{ModuleNum-Skipped,5} = Settings.VariableValues{ModuleNum-Skipped,5};
    Settings.VariableValues{ModuleNum-Skipped,12} = Settings.VariableValues{ModuleNum-Skipped,4};
    Settings.VariableValues{ModuleNum-Skipped,4} = 'Combine';
    Settings.NumbersOfVariables(ModuleNum-Skipped)=12;
    IsModuleModified = true;
    SavedVarRevNum = 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete module: InvertIntensity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'InvertIntensity')
    CurrentModuleName = 'ImageMath';
    Settings.ModuleNames{ModuleNum-Skipped}=CurrentModuleName;
    Settings.VariableValues{ModuleNum-Skipped,1} = Settings.VariableValues{ModuleNum-Skipped,1};
    Settings.VariableValues{ModuleNum-Skipped,12} = Settings.VariableValues{ModuleNum-Skipped,2};
    Settings.VariableValues{ModuleNum-Skipped,4} = 'Invert';
    Settings.NumbersOfVariables(ModuleNum-Skipped)=12;
    IsModuleModified = true;
    SavedVarRevNum = 2;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete module: Multiply
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'Multiply')
    CurrentModuleName = 'ImageMath';
    Settings.ModuleNames{ModuleNum-Skipped}=CurrentModuleName;
    Settings.VariableValues{ModuleNum-Skipped,1} = Settings.VariableValues{ModuleNum-Skipped,1};
    Settings.VariableValues{ModuleNum-Skipped,2} = Settings.VariableValues{ModuleNum-Skipped,2};
    Settings.VariableValues{ModuleNum-Skipped,12} = Settings.VariableValues{ModuleNum-Skipped,3};
    Settings.VariableValues{ModuleNum-Skipped,4} = 'Multiply';
    Settings.NumbersOfVariables(ModuleNum-Skipped)=12;
    IsModuleModified = true;
    SavedVarRevNum = 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete module: CalculateRatios
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'CalculateRatios')
    CurrentModuleName = 'CalculateMath';
    Settings.ModuleNames{ModuleNum-Skipped}=CurrentModuleName;
    Settings.VariableValues{ModuleNum-Skipped,17} = Settings.VariableValues{ModuleNum-Skipped,1};
    Settings.VariableValues{ModuleNum-Skipped,1} = Settings.VariableValues{ModuleNum-Skipped,2};
    Settings.VariableValues{ModuleNum-Skipped,2} = Settings.VariableValues{ModuleNum-Skipped,3};
    Settings.VariableValues{ModuleNum-Skipped,3} = Settings.VariableValues{ModuleNum-Skipped,4};
    Settings.VariableValues{ModuleNum-Skipped,4} = Settings.VariableValues{ModuleNum-Skipped,5};
    Settings.VariableValues{ModuleNum-Skipped,5} = Settings.VariableValues{ModuleNum-Skipped,6};
    Settings.VariableValues{ModuleNum-Skipped,6} = Settings.VariableValues{ModuleNum-Skipped,7};
    Settings.VariableValues{ModuleNum-Skipped,7} = Settings.VariableValues{ModuleNum-Skipped,8};
    Settings.VariableValues{ModuleNum-Skipped,8} = Settings.VariableValues{ModuleNum-Skipped,9};
    Settings.VariableValues{ModuleNum-Skipped,9} = Settings.VariableValues{ModuleNum-Skipped,10};
    Settings.VariableValues{ModuleNum-Skipped,10} = Settings.VariableValues{ModuleNum-Skipped,11};
    Settings.VariableValues{ModuleNum-Skipped,11} = Settings.VariableValues{ModuleNum-Skipped,12};
    Settings.VariableValues{ModuleNum-Skipped,16} = 'Divide';
    Settings.NumbersOfVariables(ModuleNum-Skipped)=17;
    IsModuleModified = true;
    SavedVarRevNum = 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete module: UnifyObjects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName,'UnifyObjects')
    CurrentModuleName = 'RelabelObjects';
    Settings.ModuleNames{ModuleNum-Skipped}=CurrentModuleName;
    Settings.VariableValues{ModuleNum-Skipped,1} = Settings.VariableValues{ModuleNum-Skipped,1};
    Settings.VariableValues{ModuleNum-Skipped,2} = Settings.VariableValues{ModuleNum-Skipped,2};
    Settings.VariableValues{ModuleNum-Skipped,5} = Settings.VariableValues{ModuleNum-Skipped,4};
    Settings.VariableValues{ModuleNum-Skipped,4} = Settings.VariableValues{ModuleNum-Skipped,3};
    Settings.VariableValues{ModuleNum-Skipped,3} = 'Unify';
    Settings.NumbersOfVariables(ModuleNum-Skipped)= 5;
    SavedVarRevNum = 1;
    IsModuleModified = true;
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obsolete module: SplitIntoContiguousObjects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName,'SplitIntoContiguousObjects')
    CurrentModuleName = 'RelabelObjects';
    Settings.ModuleNames{ModuleNum-Skipped} = CurrentModuleName;
    Settings.VariableValues{ModuleNum-Skipped,3} = 'Split';
    Settings.VariableValues{ModuleNum-Skipped,4} = '';
    Settings.VariableValues{ModuleNum-Skipped,5} = '';
    Settings.NumbersOfVariables(ModuleNum-Skipped)= 5;
    SavedVarRevNum = 1;
    IsModuleModified = true;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Changes to ExportToExcel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'ExportToExcel')
    if SavedVarRevNum == 1      % ExportToExcel.m got one new argument
        % Re-create subdirectories? Default to No.
        Settings.VariableValues{ModuleNum-Skipped,9} = 'No';
        Settings.NumbersOfVariables(ModuleNum-Skipped) = Settings.NumbersOfVariables(ModuleNum-Skipped) + 1;
        SavedVarRevNum = 2;
        IsModuleModified = true;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Changes to FileNameMetadata
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'FileNameMetadata')
    if SavedVarRevNum == 1      % FileNameMetadata got one new argument, but at the end.
        Settings.VariableValues{ModuleNum-Skipped,3} = 'No';
        Settings.NumbersOfVariables(ModuleNum-Skipped) = Settings.NumbersOfVariables(ModuleNum-Skipped) + 1;
        SavedVarRevNum = 2;
        IsModuleModified = true;
    end
    if SavedVarRevNum == 2      % Same number of arguments, but replaced third argument.
        if strcmpi(Settings.VariableValues{ModuleNum-Skipped,3},'no')
            % User only wants to look at the filename. Keep where it is,
            % and ignore the new path setting
            Settings.VariableValues{ModuleNum-Skipped,3} = 'Do not use';
        else
            % User wants to look at the path. We can't assume that the
            % fields specified the previous version are only for the path, but
            % we'll put it there and let the user decide
            Settings.VariableValues{ModuleNum-Skipped,3} = Settings.VariableValues{ModuleNum-Skipped,2};
            Settings.VariableValues{ModuleNum-Skipped,2} = 'Do not use';
        end
        SavedVarRevNum = 3;
        IsModuleModified = true;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Changes to GrayToColor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(CurrentModuleName, 'GrayToColor')
    if SavedVarRevNum == 2      % GrayToColor got two new arguments, but at the end.
        Settings.VariableValues{ModuleNum-Skipped,9} = '1';
        Settings.VariableValues(ModuleNum-Skipped,5:8) = Settings.VariableValues(ModuleNum-Skipped,4:7); 
        Settings.VariableValues{ModuleNum-Skipped,4} = 'Do not use'; 
        Settings.VariableInfoTypes{ModuleNum-Skipped,4} = 'imagegroup';
        Settings.NumbersOfVariables(ModuleNum-Skipped) = Settings.NumbersOfVariables(ModuleNum-Skipped) + 2;
        SavedVarRevNum = 3;
        IsModuleModified = true;
    end
end
