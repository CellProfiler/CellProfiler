function handles = Combine(handles)

% Help for the Combine module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Takes 1 to 3 images and combines the images. Each color's brightness can be adjusted independently.
% *************************************************************************
%
% Settings:
%
% Adjustment factors: Leaving the adjustment factors set to 1 will
% balance all three colors equally in the final image, and they will
% use the same range of intensities as each individual incoming image.
% Using factors less than 1 will decrease the intensity of that
% color in the final image, and values greater than 1 will increase
% it.  Setting the adjustment factor to zero will cause that color to
% be entirely blank.
%
% See also GrayToColor.

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
%
% Website: http://www.cellprofiler.org
%
% $Revision: 3524 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be called Image 1?
%choiceVAR01 = Leave this black
%infotypeVAR01 = imagegroup
Image1Name = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the image to be called Image 2?
%choiceVAR02 = Leave this black
%infotypeVAR02 = imagegroup
Image2Name = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What did you call the image to be colored Image 3?
%choiceVAR03 = Leave this black
%infotypeVAR03 = imagegroup
Image3Name = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What do you want to call the resulting image?
%defaultVAR04 = CombinedImage
%infotypeVAR04 = imagegroup indep
CombinedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter the adjustment factor for Image 1
%defaultVAR05 = 1
Image1AdjustmentFactor = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Enter the adjustment factor for Image 2
%defaultVAR06 = 1
Image2AdjustmentFactor = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Enter the adjustment factor for Image 3
%defaultVAR07 = 1
Image3AdjustmentFactor = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines whether the user has specified an Image 3 to be loaded 
if ~strcmp(Image3Name, 'Leave this black')
    %%% Read (open) the images and assign them to variables.
    Image3 = CPretrieveimage(handles,Image3Name,ModuleName,'DontCheckColor','CheckScale');
    Image3Exists = 1;
else
    Image3Exists = 0;
end

%%% Repeat for 1 and 2.
if ~strcmp(Image2Name, 'Leave this black')
    Image2 = CPretrieveimage(handles,Image2Name,ModuleName,'DontCheckColor','CheckScale');
    Image2Exists = 1;
else Image2Exists = 0;
end

if ~strcmp(Image1Name, 'Leave this black')
    Image1 = CPretrieveimage(handles,Image1Name,ModuleName,'DontCheckColor','CheckScale');
    Image1Exists = 1;
else Image1Exists = 0;
end
drawnow

%%% If any of the colors are to be left black, creates the appropriate
%%% image.
if ~Image3Exists && ~Image1Exists && ~Image2Exists
    error(['Image processing was canceled in the ', ModuleName, ' module because you have not selected any images to be merged.'])
end
if ((~Image3Exists && ~Image1Exists && Image2Exists)||(~Image3Exists && Image1Exists && ~Image2Exists)||(Image3Exists && ~Image1Exists && ~Image2Exists))
    error(['Image processing was canceled in the ', ModuleName, ' module because you have not selected enough images to be merged.'])
end
if Image3Exists && Image1Exists && ~Image2Exists
    Image2 = zeros(size(Image3));
end
if ~Image3Exists && Image1Exists && Image2Exists
    Image3 = zeros(size(Image2));
end
if Image3Exists && ~Image1Exists && Image2Exists
    Image1 = zeros(size(Image3));
end

%%% Checks whether the three images are the same size.
try
    if size(Image3) ~= size(Image2)
        error(['Image processing was canceled in the ', ModuleName, ' module because the three images selected are not the same size.  The pixel dimensions must be identical.'])
    end
    if size(Image1) ~= size(Image2)
        error(['Image processing was canceled in the ', ModuleName, ' module because the three images selected are not the same size.  The pixel dimensions must be identical.'])
    end
catch error(['Image processing was canceled in the ', ModuleName, ' module because there was a problem with one of three images selected. Most likely one of the images is not in the same format as the others - for example, one of the images might already be in color (RGB) format.'])
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% If any of the images are binary/logical format, they must be
%%% converted to a double first before immultiply.
AdjustedImage1 = immultiply(double(Image1),str2double(Image1AdjustmentFactor));
AdjustedImage2 = immultiply(double(Image2),str2double(Image2AdjustmentFactor));
AdjustedImage3 = immultiply(double(Image3),str2double(Image3AdjustmentFactor));


if ~Image3Exists && ~Image1Exists && ~Image2Exists
    error(['Image processing was canceled in the ', ModuleName, ' module because you have not selected any images to be merged.'])
end
if ~Image3Exists && ~Image1Exists && Image2Exists
    CombinedImage = AdjustedImage2;
end
if ~Image3Exists && Image1Exists && ~Image2Exists
    CombinedImage = AdjustedImage1;
end
if Image3Exists && ~Image1Exists && ~Image2Exists
    CombinedImage = AdjustedImage3;
end
if Image3Exists && Image1Exists && ~Image2Exists
    CombinedImage = imadd(AdjustedImage1, AdjustedImage3)/2;
end
if ~Image3Exists && Image1Exists && Image2Exists
    CombinedImage = imadd(AdjustedImage1, AdjustedImage2)/2;
end
if Image3Exists && ~Image1Exists && Image2Exists
    CombinedImage = imadd(AdjustedImage2, AdjustedImage3)/2;
end
if Image3Exists && Image1Exists && Image2Exists
    CombinedImage = imadd(AdjustedImage1, AdjustedImage2);
    CombinedImage = imadd(CombinedImage, AdjustedImage3)/3;
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
        CPresizefigure(CombinedImage,'TwoByTwo',ThisModuleFigureNumber);
    end
    %%% A subplot of the figure window is set to display the Combined Image
    %%% image.  Using CPimagesc or image instead of imshow doesn't work when
    %%% some of the pixels are saturated.
    subplot(2,2,1); 
    CPimagesc(CombinedImage,handles);
    title(['Merged Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display Image 1.
    counter = 2;
    if Image1Exists
        subplot(2,2,counter); 
        CPimagesc(Image1,handles); 
        title('Image 1');
        counter=counter+1;
    end
    %%% A subplot of the figure window is set to display the Image 2.
    if Image2Exists
        subplot(2,2,counter); 
        CPimagesc(Image2,handles); 
        title('Image 2');
        counter=counter+1;
    end
    %%% A subplot of the figure window is set to display the Image 3.
    if Image3Exists
        subplot(2,2,counter);
        CPimagesc(Image3,handles); 
        title('Image 3');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(CombinedImageName) = CombinedImage;