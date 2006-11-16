function handles = Morph(handles)

% Help for the Morph module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Beta version: provides access to built in Matlab morphological functions.
% *************************************************************************
%
% Beta version: provides access to built in Matlab morphological functions.
%
% Settings:
%
% Beta

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision: 4170 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What image do you want to morph?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the resulting image?
%defaultVAR02 = MorphBlue
%infotypeVAR02 = imagegroup indep
MorphImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What function do you want to perform?
%choiceVAR03 = Do not use
%choiceVAR03 = bothat
%choiceVAR03 = bridge
%choiceVAR03 = clean
%choiceVAR03 = close
%choiceVAR03 = diag
%choiceVAR03 = dilate
%choiceVAR03 = erode
%choiceVAR03 = fill
%choiceVAR03 = hbreak
%choiceVAR03 = majority
%choiceVAR03 = open
%choiceVAR03 = remove
%choiceVAR03 = shrink
%choiceVAR03 = skel
%choiceVAR03 = spur
%choiceVAR03 = thicken
%choiceVAR03 = thin
%choiceVAR03 = tophat
%inputtypeVAR03 = popupmenu
Functions{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = How many times do you want to repeat the function?
%defaultVAR04 = 1
FunctionVariables{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What function do you want to perform?
%choiceVAR05 = Do not use
%choiceVAR05 = bothat
%choiceVAR05 = bridge
%choiceVAR05 = clean
%choiceVAR05 = close
%choiceVAR05 = diag
%choiceVAR05 = dilate
%choiceVAR05 = erode
%choiceVAR05 = fill
%choiceVAR05 = hbreak
%choiceVAR05 = majority
%choiceVAR05 = open
%choiceVAR05 = remove
%choiceVAR05 = shrink
%choiceVAR05 = skel
%choiceVAR05 = spur
%choiceVAR05 = thicken
%choiceVAR05 = thin
%choiceVAR05 = tophat
%inputtypeVAR05 = popupmenu
Functions{2} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = How many times do you want to repeat the function?
%defaultVAR06 = 1
FunctionVariables{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What function do you want to perform?
%choiceVAR07 = Do not use
%choiceVAR07 = bothat
%choiceVAR07 = bridge
%choiceVAR07 = clean
%choiceVAR07 = close
%choiceVAR07 = diag
%choiceVAR07 = dilate
%choiceVAR07 = erode
%choiceVAR07 = fill
%choiceVAR07 = hbreak
%choiceVAR07 = majority
%choiceVAR07 = open
%choiceVAR07 = remove
%choiceVAR07 = shrink
%choiceVAR07 = skel
%choiceVAR07 = spur
%choiceVAR07 = thicken
%choiceVAR07 = thin
%choiceVAR07 = tophat
%inputtypeVAR07 = popupmenu
Functions{3} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = How many times do you want to repeat the function?
%defaultVAR08 = 1
FunctionVariables{3} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What function do you want to perform?
%choiceVAR09 = Do not use
%choiceVAR09 = bothat
%choiceVAR09 = bridge
%choiceVAR09 = clean
%choiceVAR09 = close
%choiceVAR09 = diag
%choiceVAR09 = dilate
%choiceVAR09 = erode
%choiceVAR09 = fill
%choiceVAR09 = hbreak
%choiceVAR09 = majority
%choiceVAR09 = open
%choiceVAR09 = remove
%choiceVAR09 = shrink
%choiceVAR09 = skel
%choiceVAR09 = spur
%choiceVAR09 = thicken
%choiceVAR09 = thin
%choiceVAR09 = tophat
%inputtypeVAR09 = popupmenu
Functions{4} = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = How many times do you want to repeat the function?
%defaultVAR10 = 1
FunctionVariables{4} = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = What function do you want to perform?
%choiceVAR11 = Do not use
%choiceVAR11 = bothat
%choiceVAR11 = bridge
%choiceVAR11 = clean
%choiceVAR11 = close
%choiceVAR11 = diag
%choiceVAR11 = dilate
%choiceVAR11 = erode
%choiceVAR11 = fill
%choiceVAR11 = hbreak
%choiceVAR11 = majority
%choiceVAR11 = open
%choiceVAR11 = remove
%choiceVAR11 = shrink
%choiceVAR11 = skel
%choiceVAR11 = spur
%choiceVAR11 = thicken
%choiceVAR11 = thin
%choiceVAR11 = tophat
%inputtypeVAR11 = popupmenu
Functions{5} = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = How many times do you want to repeat the function?
%defaultVAR12 = 1
FunctionVariables{5} = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = What function do you want to perform?
%choiceVAR13 = Do not use
%choiceVAR13 = bothat
%choiceVAR13 = bridge
%choiceVAR13 = clean
%choiceVAR13 = close
%choiceVAR13 = diag
%choiceVAR13 = dilate
%choiceVAR13 = erode
%choiceVAR13 = fill
%choiceVAR13 = hbreak
%choiceVAR13 = majority
%choiceVAR13 = open
%choiceVAR13 = remove
%choiceVAR13 = shrink
%choiceVAR13 = skel
%choiceVAR13 = spur
%choiceVAR13 = thicken
%choiceVAR13 = thin
%choiceVAR13 = tophat
%inputtypeVAR13 = popupmenu
Functions{6} = char(handles.Settings.VariableValues{CurrentModuleNum,13});

%textVAR14 = How many times do you want to repeat the function?
%defaultVAR14 = 1
FunctionVariables{6} = char(handles.Settings.VariableValues{CurrentModuleNum,14});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the images.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');

tmp1 = {};
tmp2 = {};
for n = 1:6
    if ~strcmp(Functions{n}, 'Do not use')
        tmp1{end+1} = Functions{n}; %#ok Ignore MLint
        tmp2{end+1} = FunctionVariables{n}; %#ok Ignore MLint
    end
end
Functions = tmp1;
FunctionVariables = tmp2;
%%% Check FunctionVariables to make sure they are all valid, if not reset
%%% to the default value of 1.
for i=1:length(FunctionVariables)
    if (str2double(FunctionVariables(i))<1) || (isnan(str2double(FunctionVariables(i))))
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Times to repeat the function invalid']))
            CPwarndlg(['The number of times to repeat the function you have entered in the ' ModuleName ' module is invalid or less than one. It is being set to the default value of 1.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Times to repeat the function invalid']);
        end
        FunctionVariables{i} = '1';
    end
end
%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

FinalImage = OrigImage;
for i = 1:length(Functions)
    FinalImage = bwmorph(FinalImage,Functions{i},str2double(FunctionVariables{i}));
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1);
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    CPimagesc(FinalImage,handles);
    title('Morphed Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(MorphImageName) = FinalImage;
