function handles = GrayToColor(handles)
  
% Help for the Gray To Color module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Takes 1 to 4 images and assigns them to colors in a final red, green,
% blue (RGB) image. Each color's brightness can be adjusted independently.
% *************************************************************************
%
% This module takes up to four grayscale images as inputs, and produces
% either a new color (RGB) image which results from assigning each of the 
% input images the colors red, green, and blue (RGB, for 3 color) or cyan, 
% yellow, magenta, and gray (CMYK, for 4 color) respectively.
% In addition, each color's intensity can be adjusted independently by
% using adjustment factors (see below).
%
% Settings:
%
% Choose the input images: You must select at least one image which you
% would like to use to create the color image. Also, all images must be the
% same size, since they will combined pixel by pixel.
%
% Adjustment factors: Leaving the adjustment factors set to 1 will balance
% all colors equally in the final image, and they will use the same range
% of intensities as each individual incoming image. Using factors less than
% 1 will decrease the intensity of that color in the final image, and 
% values greater than 1 will increase it. Setting the adjustment factor
% to zero will cause that color to be entirely blank.
%
% See also ColorToGray.

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

%%%%%%%
% VARIABLES %
%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be colored red (RGB) or cyan (CMYK)?
%choiceVAR01 = Do not use
%infotypeVAR01 = imagegroup
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu custom

%textVAR02 = What did you call the image to be colored green (RGB) or magneta (CMYK)?
%choiceVAR02 = Do not use
%infotypeVAR02 = imagegroup
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu custom

%textVAR03 = What did you call the image to be colored blue (RGB) or yellow (CMYK)?
%choiceVAR03 = Do not use
%infotypeVAR03 = imagegroup
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = What did you call the image to be colored gray (CMYK only)?
%choiceVAR04 = Do not use
%infotypeVAR04 = imagegroup
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = What do you want to call the resulting image?
%defaultVAR05 = ColorImage
%infotypeVAR05 = imagegroup indep
RGBImageName = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Enter the adjustment factor for the red/cyan image
%defaultVAR06 = 1
AdjustmentFactor{1} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Enter the adjustment factor for the green/magneta image
%defaultVAR07 = 1
AdjustmentFactor{2} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Enter the adjustment factor for the blue/yellow image
%defaultVAR08 = 1
AdjustmentFactor{3} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = Enter the adjustment factor for the gray image (optional; CMYK only)
%defaultVAR09 = 1
AdjustmentFactor{4} = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%
% PRELIMINARY CALCULATIONS & FILE HANDLING %
%%%%%%%%%%%%%%%%
drawnow

% Determines which images the user has specified 
[Image,ImageExists] = deal(cell(1,4));
for i = 1:4,
    if ~strcmp(ImageName{i}, 'Do not use')
        % Read (open) the images and assign them to variables.
        Image{i} = CPretrieveimage(handles,ImageName{i},ModuleName,'MustBeGray','CheckScale');
        ImageExists{i} = 1;
    end
end

% Check if all images are specified
ImagesSpecified = ~cellfun(@isempty,ImageExists);
if all(~ImagesSpecified)
    error(['Image processing was canceled in the ', ModuleName, ' module because you have not selected any images to be merged.'])
end

% Determine whether 3- or 4-color combination
[isRGB,isCMYK] = deal(0);
if ~ImagesSpecified(4)
    isRGB = 1;
else
    isCMYK = 1;
end

% Checks whether the images have the same color depth.
sz = cellfun(@size,Image(ImagesSpecified),'uniformoutput',false);
if length(unique(cell2mat(cellfun(@ndims,sz,'uniformoutput',false)))) > 1
    error(['Image processing was canceled in the ', ModuleName, ' module because the images selected are not the same color depth.  Most likely one of the images is not in the same format as the others - for example, one of the images might already be in color (RGB) format.'])
end

% Checks whether the three images are the same size
UniqueDims = unique(cat(1,sz{:}),'rows');
if size(UniqueDims,1) > 1
    error(['Image processing was canceled in the ', ModuleName, ' module because the images selected are not the same size.  The pixel dimensions must be identical.'])
end

% If any of the colors are to be left black, creates the appropriate image.
[Image{cellfun(@isempty,ImageExists)}] = deal(zeros(UniqueDims));

% Check to see if all adjustment factors are in the correct range of 0 to 1
adj = str2double(AdjustmentFactor);
outOfRange = adj < 0 | isnan(adj);
if any(outOfRange)
    str = num2str(find(outOfRange));
    if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Adjustment factor invalid']))
        CPwarndlg(['The adjustment factor entered for image ',str,' in the ' ModuleName ' module is invalid or less than 0. It is being set to the default value of 1.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Adjustment factor invalid']);
    end
end
AdjustmentFactor(outOfRange) = {'1'};

%%%%%%%%
% IMAGE ANALYSIS %
%%%%%%%%
drawnow

% If any of the images are binary/logical format, they must be
% converted to a double first before immultiply

Image = cellfun(@double,Image,'uniformoutput',false);
adj = str2double(AdjustmentFactor);
if isRGB
    RGBImage = [];
    for i = 1:3,
        RGBImage = cat(3,RGBImage,Image{i}.*adj(i));
    end
elseif isCMYK
    c = {[0 1 1 1],[1 1 0 1],[1 0 1 1]};
    RGBImage = [];
    for i = 1:3,
        RGBImage = cat(3, RGBImage, c{i}(1)*Image{1}.*adj(1) + ...
                                    c{i}(2)*Image{2}.*adj(2) + ...
                                    c{i}(3)*Image{3}.*adj(3) + ...
                                    c{i}(4)*Image{4}.*adj(4));
    end
end
% Normalize by adjustment factors. Not sure if this is needed?
%RGBImage = RGBImage/sum(adj);

%%%%%%%%%
% DISPLAY RESULTS %
%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    % Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        if isRGB
            CPresizefigure(RGBImage,'TwoByTwo',ThisModuleFigureNumber);
            
            % A subplot of the figure window is set to display the merged image.
            hAx = subplot(2,2,1,'parent',ThisModuleFigureNumber); 
            CPimagesc(RGBImage,handles,hAx);
            title(hAx,['Merged Color Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
            % A subplot of the figure window is set to display the blue image.
            hAx = subplot(2,2,2,'parent',ThisModuleFigureNumber); 
            CPimagesc(Image{3},handles,hAx); 
            title(hAx,'Blue Image');
            % A subplot of the figure window is set to display the green image.
            hAx = subplot(2,2,3,'parent',ThisModuleFigureNumber); 
            CPimagesc(Image{2},handles,hAx); 
            title(hAx,'Green Image');
            % A subplot of the figure window is set to display the red image.
            hAx = subplot(2,2,4,'parent',ThisModuleFigureNumber); 
            CPimagesc(Image{1},handles,hAx); 
            title(hAx,'Red Image');
        elseif isCMYK
            CPresizefigure(Image{1},'TwobyThree',ThisModuleFigureNumber);
            
            % A subplot of the figure window is set to display the cyan image.
            hAx = subplot(2,3,1,'parent',ThisModuleFigureNumber); 
            CPimagesc(Image{1},handles,hAx);
            title(hAx,'Cyan Image');
            % A subplot of the figure window is set to display the magenta image.
            hAx = subplot(2,3,2,'parent',ThisModuleFigureNumber); 
            CPimagesc(Image{2},handles,hAx); 
            title(hAx,'Yellow Image');
            % A subplot of the figure window is set to display the yellow image.
            hAx = subplot(2,3,3,'parent',ThisModuleFigureNumber); 
            CPimagesc(Image{3},handles,hAx); 
            title(hAx,'Magenta Image');
            % A subplot of the figure window is set to display the gray image.
            hAx = subplot(2,3,4,'parent',ThisModuleFigureNumber); 
            CPimagesc(Image{4},handles,hAx); 
            title(hAx,'Gray Image');
            % A subplot of the figure window is set to display the merged image.
            hAx = subplot(2,3,5,'parent',ThisModuleFigureNumber); 
            CPimagesc(RGBImage,handles,hAx); 
            title(hAx,['Merged Color Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        end
    end

    
end

%%%%%%%%%%%%%%
% SAVE DATA TO HANDLES STRUCTURE %
%%%%%%%%%%%%%%
drawnow

% Saves the adjusted image to the handles structure so it can be used by
% subsequent modules.
handles.Pipeline.(RGBImageName) = RGBImage;