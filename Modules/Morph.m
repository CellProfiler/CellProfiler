function handles = Morph(handles)

% Help for the Morph module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Beta version: provides access to built in Matlab morphological functions.
% *************************************************************************
%
% Beta version: provides access to built in Matlab morphological functions.
% If you have defined more than one function to be applied, each individual 
% function is repeated the number of times specified before progressing to 
% the next function in the list.  
%
% Note that these will only operate on binary images
%
% Settings:
% The number of times repeated can be 'Inf', which ceases operation when
% the image no longer changes.
%
% Beta

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

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

%% Remove unused settings
idx = strcmp(Functions,'Do not use');
Functions = Functions(~idx);
FunctionVariables = FunctionVariables(~idx);

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

%%% Reads the images.
Images = cell(1,length(Functions)+1);
Images{1} = CPretrieveimage(handles,ImageName,ModuleName,'DontCheckColor','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

for i = 1:length(Functions)
    Images{i+1} = bwmorph(Images{i},Functions{i},str2double(FunctionVariables{i}));
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

    %%%% Display FinalImage as default
    CPimagesc(Images{end}, handles,ThisModuleFigureNumber);

    %%% Construct structure which holds images and figure titles
    ud = cell2struct(Images,'img',1);
    
    if isempty(findobj(ThisModuleFigureNumber,'tag','PopupImage')),
        
%         ud.img = Images;
        ud(1).title = 'Input Image';
        for i = 1:length(Functions)
            ud(i+1).title = ['Image after ' Functions{i} ' operation, cycle # ',num2str(handles.Current.SetBeingAnalyzed)];
        end
        ud(end).title = ['Final ' ud(end).title];
        title(ud(end).title)
        
        %%% uicontrol for displaying other images
        uicontrol(ThisModuleFigureNumber, ...
            'Style', 'popup',...
            'String', {ud.title},...
            'UserData',ud,...
            'units','normalized',...
            'position',[.01 .95 .25 .04],...
            'backgroundcolor',[.7 .7 .9],...
            'tag','PopupImage',...
            'Value',length(Functions)+1,...
            'Callback', @CP_ImagePopupmenu_Callback);

    else
        title(['Final Image = ', MorphImageName])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(MorphImageName) = Images{end};