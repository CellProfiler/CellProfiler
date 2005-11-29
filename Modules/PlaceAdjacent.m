function handles = PlaceAdjacent(handles)

% Help for the Place Adjacent module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Place up to six images next to each other to produce a single image.
% *************************************************************************
%
% This module places two images next to each other, either
% horizontally or vertically.
%
% See also <nothing relevant>.

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
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%
%%%%%%%%%%%%%%%%
drawnow


[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the first image to be placed?
%infotypeVAR01 = imagegroup
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the second image to be placed?
%infotypeVAR02 = imagegroup
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What did you call the first image to be placed?
%choiceVAR03 = Do not use
%infotypeVAR03 = imagegroup
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What did you call the second image to be placed?
%choiceVAR04 = Do not use
%infotypeVAR04 = imagegroup
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = What did you call the first image to be placed?
%choiceVAR05 = Do not use
%infotypeVAR05 = imagegroup
ImageName{5} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = What did you call the second image to be placed?
%choiceVAR06 = Do not use
%infotypeVAR06 = imagegroup
ImageName{6} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = What do you want to call the resulting image?
%defaultVAR07 = AdjacentImage
%infotypeVAR07 = imagegroup indep
AdjacentImageName = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Placement Type.
%choiceVAR08 = Horizontal
%choiceVAR08 = Vertical
HorizontalOrVertical = char(handles.Settings.VariableValues{CurrentModuleNum,8});
HorizontalOrVertical = HorizontalOrVertical(1);
%inputtypeVAR08 = popupmenu

%textVAR09 = Can the incoming images be deleted from the pipeline after they are placed adjacent (this saves memory, but prevents you from using the incoming images later in the pipeline)?
%choiceVAR09 = No
%choiceVAR09 = Yes
DeletePipeline = char(handles.Settings.VariableValues{CurrentModuleNum,9});
DeletePipeline = DeletePipeline(1);
%inputtypeVAR09 = popupmenu

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

tmp1 = {};
for n = 1:6
    if ~strcmp(ImageName{n}, 'Do not use')
        tmp1{end+1} = ImageName{n};
    end
end
ImageName = tmp1;

OrigImage = {};
for i=1:length(ImageName)
    %%% Reads (opens) the image you want to analyze and assigns it to a
    %%% variable.
    fieldname = ['', ImageName{i}];
    %%% Checks whether the image to be analyzed exists in the handles structure.
    if ~isfield(handles.Pipeline, fieldname)
        %%% If the image is not there, an error message is produced.  The error
        %%% is not displayed: The error function halts the current function and
        %%% returns control to the calling function (the analyze all images
        %%% button callback.)  That callback recognizes that an error was
        %%% produced because of its try/catch loop and breaks out of the image
        %%% analysis loop without attempting further modules.
        error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName1, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    %%% Reads the image.
    OrigImage{i} = handles.Pipeline.(fieldname);

    %%% Removes the image from the pipeline to save memory if requested.
    if strncmpi(DeletePipeline,'Y',1) == 1
        handles.Pipeline = rmfield(handles.Pipeline,fieldname);
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Check that the images are the same height or width and place them
%%% adjacent to each other.

if strcmpi(HorizontalOrVertical,'H')
    for i=1:(length(OrigImage)-1)
        if size(OrigImage{i},1) ~= size(OrigImage{i+1},1)
            error(['Image processingwas canceled in the ', ModuleName, ' module because the two input images must have the same height if they are to be placed horizontally adjacent to each other.'])
        end
        %%% If one of the images is multidimensional (color), the other
        %%% one is replicated to match its dimensions.
        if size(OrigImage{i},3) ~= size(OrigImage{i+1},3)
            DesiredLayers = max(size(OrigImage{i},3),size(OrigImage{i+1},3));
            if size(OrigImage{i},3) > size(OrigImage{i+1},3)
                for i = 1:DesiredLayers, OrigImage{i+1}(:,:,i) = OrigImage{i+1}(:,:,1); end
            else
                for i = 1:DesiredLayers, OrigImage{i}(:,:,i) = OrigImage{i}(:,:,1); end
            end
        end
        if i == 1
            TempAdjacentImage = cat(2,OrigImage{i},OrigImage{i+1});
        else
            TempAdjacentImage = cat(2,TempAdjacentImage,OrigImage{i+1});
        end
    end
    AdjacentImage = TempAdjacentImage;
elseif strcmpi(HorizontalOrVertical,'V')
    for i=1:(length(OrigImage)-1)
        if size(OrigImage{i},2) ~= size(OrigImage{i+1},2)
            error(['Image processing was canceled in the ', ModuleName, ' module because the two input images must have the same height if they are to be placed horizontally adjacent to each other.'])
        end
        %%% If one of the images is multidimensional (color), the other
        %%% one is replicated to match its dimensions.
        if size(OrigImage{i},3) ~= size(OrigImage{i+1},3)
            DesiredLayers = max(size(OrigImage{i},3),size(OrigImage{i+1},3));
            if size(OrigImage{i},3) > size(OrigImage{i+1},3)
                for i = 1:DesiredLayers, OrigImage{i+1}(:,:,i) = OrigImage{i+1}(:,:,1); end
            else
                for i = 1:DesiredLayers, OrigImage{i}(:,:,i) = OrigImage{i}(:,:,1); end
            end
        end
        if i == 1
            TempAdjacentImage = cat(1,OrigImage{i},OrigImage{i+1});
        else
            TempAdjacentImage = cat(1,TempAdjacentImage,OrigImage{i+1});
        end
    end
    AdjacentImage = TempAdjacentImage;
else
    error(['Image processing was canceled in the ', ModuleName, ' module because you must enter H or V to specify whether to place the images adjacent to each other horizontally or vertically.'])
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    CPimagesc(AdjacentImage);
    title(['Adjacent Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the procesed imge to the
%%% handles structure so it can be used by subsequent modules.
handles.Pipeline.(AdjacentImageName) = AdjacentImage;