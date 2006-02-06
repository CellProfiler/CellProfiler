function handles = PlaceAdjacent(handles)

% Help for the Place Adjacent module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Places up to six images next to each other, either horizontally or
% vertically, to produce a single image.
% *************************************************************************
%
% To place together many images, you can use several of this module in one
% pipeline.
%
% See also Tile.

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

%textVAR01 = Select images to be placed, in order
%infotypeVAR01 = imagegroup
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = 
%infotypeVAR02 = imagegroup
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = 
%choiceVAR03 = Do not use
%infotypeVAR03 = imagegroup
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = 
%choiceVAR04 = Do not use
%infotypeVAR04 = imagegroup
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = 
%choiceVAR05 = Do not use
%infotypeVAR05 = imagegroup
ImageName{5} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = 
%choiceVAR06 = Do not use
%infotypeVAR06 = imagegroup
ImageName{6} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = What do you want to call the resulting image?
%defaultVAR07 = AdjacentImage
%infotypeVAR07 = imagegroup indep
AdjacentImageName = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = How do you want to place the images, Horizontal (left to right) or Vertical (top to bottom)?
%choiceVAR08 = Horizontal
%choiceVAR08 = Vertical
HorizontalOrVertical = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = Should the incoming images be deleted from the pipeline after they are placed? (This saves memory, but prevents you from using the incoming images later in the pipeline)
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
    OrigImage{i} = CPretrieveimage(handles,ImageName{i},ModuleName);

    %%% Removes the image from the pipeline to save memory if requested.
    if strncmpi(DeletePipeline,'Y',1) == 1
        handles.Pipeline = rmfield(handles.Pipeline,ImageName{i});
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determine which dimension to check for the same height or width.
if strcmpi(HorizontalOrVertical,'Vertical')
    DimensionToCheck = 2;
    PotentialErrorMsg = 'width if they are to be placed vertically';
    DimensionToPlace = 1;
elseif strcmpi(HorizontalOrVertical,'Horizontal')
    DimensionToCheck = 1;
    PotentialErrorMsg = 'height if they are to be placed horizontally';
    DimensionToPlace = 2;
end

for i=1:(length(OrigImage)-1)
    if size(OrigImage{i},DimensionToCheck) ~= size(OrigImage{i+1},DimensionToCheck)
        error(['Image processing was canceled in the ', ModuleName, ' module because the two input images must have the same ',PotentialErrorMsg,' adjacent to each other.'])
    end
    OrigImage = MakeLayersMatch(OrigImage,i);
    
    if i == 1
        AdjacentImage = cat(DimensionToPlace,OrigImage{i},OrigImage{i+1});
    else
        AdjacentImage = cat(DimensionToPlace,AdjacentImage,OrigImage{i+1});
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(AdjacentImage,'OneByOne',ThisModuleFigureNumber)
    end
    CPimagesc(AdjacentImage,handles);
    title(['Adjacent Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the processed image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(AdjacentImageName) = AdjacentImage;

%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION %%%
%%%%%%%%%%%%%%%%%%%

function OrigImage = MakeLayersMatch(OrigImage,i)

%%% If one of the images is multidimensional (color), the other one is
%%% replicated to match its dimensions.
if size(OrigImage{i},3) ~= size(OrigImage{i+1},3)
    DesiredLayers = max(size(OrigImage{i},3),size(OrigImage{i+1},3));
    if size(OrigImage{i},3) > size(OrigImage{i+1},3)
        for j = 1:DesiredLayers, OrigImage{i+1}(:,:,j) = OrigImage{i+1}(:,:,1); end
    else
        for j = 1:DesiredLayers, OrigImage{i}(:,:,j) = OrigImage{i}(:,:,1); end
    end
end