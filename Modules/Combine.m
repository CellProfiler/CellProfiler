function handles = Combine(handles)

% Help for the Combine module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Takes 1 to 3 color or grayscale images and combines them into 1. Each image's
% intensity can be adjusted independently.
% *************************************************************************
%
% This module takes color or grayscale images as inputs, and produces a new one
% which results from the weighted average of the pixel intensities of the
% input images. All the images to be combined must be all either grayscale
% or color.  The average is found by taking the product of each weight
% and the intensity of its corresponding image, adding the products, and
% then dividing the result by the sum of the weights. By taking the
% weighted average of the pixel intensities, the overall intensity of the
% resulting image will remain the same as that of the inputs. If you want
% to change the overall intensity, you should use the Rescale module.
%
% Settings:
%
% Choose the input images: You must select at least two images which you
% would like to combine. They must all be the same size, since the average
% will be taken pixel by pixel.
%
% Weights: The weights will determine how each input image will affect the
% combined one. The higher the weight of an image, the more it will be
% reflected in the combined image. Because of the way the average is taken,
% it only matters how these weights relate to each other (e.g. entering
% weights 0.27, 0.3, and 0.3 is the same as entering weights 0.9, 1, and
% 1). Make sure all the weights have positive values.
%
% See also GrayToColor, ColorToGray, and Rescale modules.

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
% $Revision: 3524 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the first image to be combined?
%choiceVAR01 = Leave this blank
%infotypeVAR01 = imagegroup
ImageNames{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the second image to be combined?
%choiceVAR02 = Leave this blank
%infotypeVAR02 = imagegroup
ImageNames{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What did you call the third image to be combined?
%choiceVAR03 = Leave this blank
%infotypeVAR03 = imagegroup
ImageNames{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What do you want to call the combined image?
%defaultVAR04 = CombinedImage
%infotypeVAR04 = imagegroup indep
CombinedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter the weight you want to give the first image
%defaultVAR05 = 1
ImageWeights{1} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Enter the weight you want to give the second image
%defaultVAR06 = 1
ImageWeights{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Enter the weight you want to give the third image
%defaultVAR07 = 1
ImageWeights{3} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Make the ImageWeights be numbers
Weights = str2double(ImageWeights);

%%% If selected, load images and create existance flag matrix for them
Images = {};
for i = 1:3
    if ~strcmpi(ImageNames{i},'Leave this blank')
        try
            Images{end+1} = CPretrieveimage(handles,ImageNames{i},ModuleName,'DontCheckColor','CheckScale'); %#ok Ignore MLint
        catch
            error(['Image processing was canceled in the ' ModuleName 'module because an error occurred while trying to load the image ' ImageNames{i} '. Please make sure it is a valid input image. Perhaps you chose an image that will be created later in the pipeline, in which case you should relocate the Combine module or the other one.']);
        end
    else
        Weights(i) = NaN;
    end
end
drawnow

%%% Do error check for number of images, size, and weights
if length(Images)<2
    error(['Image processing was canceled in the ' ModuleName ' module because you have not selected enough images to be combined.'])
end
[Rows Columns] = cellfun(@size,Images);
if any(Rows~=Rows(1)) || any(Columns~=Columns(1))
    error(['Image processing was canceled in the ' ModuleName ' module because the images selected are not the same size. The pixel dimensions must be identical. Most likely one of the images is not in the same format as the others - for example, one of the images might already be in color (RGB) format.'])
end
Weights = Weights(~isnan(Weights));
if sum(Weights)<=0 || any(Weights<0)
    if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Non-positive values entered']))
        CPwarndlg(['The values entered in ' ModuleName ' are required to all be positive values. Non-positive values are being changed to default values of 1.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Non-positive values entered'],'replace');
    end
    Weights((Weights < 0)) = 1;
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Make the weights a cell array so we can apply cellfun
ImageWeights = num2cell(Weights);

%%% If any of the images are binary/logical format, they must be converted
%%% to a double first before immultiply
Images = cellfun(@double,Images,'UniformOutput',0);
WeightedImages = cellfun(@immultiply,Images,ImageWeights,'UniformOutput',0);

AddedImages = zeros(size(WeightedImages{1}));
for i = 1:length(WeightedImages)
    AddedImages = WeightedImages{i}+AddedImages;
end
CombinedImage = AddedImages./sum(Weights);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(CombinedImage,'TwoByTwo',ThisModuleFigureNumber);
    end
    %%% A subplot of the figure window is set to display the Combined Image
    %%% image.  Using CPimagesc or image instead of imshow doesn't work when
    %%% some of the pixels are saturated.
    subplot(2,2,1);
    CPimagesc(CombinedImage,handles);
    title(['Combined Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display Image 1.
    for i = 1:length(Images)
        subplot(2,2,i+1)
        CPimagesc(Images{i},handles);
        title(['Image ' num2str(i)]);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(CombinedImageName) = CombinedImage;