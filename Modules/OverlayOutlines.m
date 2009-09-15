function handles = OverlayOutlines(handles)

% Help for the Overlay Outlines module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Places outlines produced by an identify module over a desired image.
% *************************************************************************
%
% Outlines (in a special format produced by an identify module) can be
% placed on any desired image (grayscale, color, or blank) and then this 
% resulting image can be saved using the SaveImages module.
%
% Settings:
% Would you like to set the intensity (brightness) of the outlines to be
% the same as the brightest point in the image, or the maximum possible
% value for this image format?
%
% If your image is quite dim, then putting bright white lines onto it may
% not be useful. It may be preferable to make the outlines equal to the
% maximal brightness already occurring in the image.
%
% If you choose to display outlines on a Blank image, the maximum intensity
% will default to 'Max possible'.
%
% See also identify modules.

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

% Variable Settings for PyCP
% I can't find anything to change as far as variables are concerned.  The
% only thing that comes to me with this module is that users often don't
% realize they can just name outlines in the Identify modules and save them
% directly from there- there's actually no need to overlay on a blank
% image, unless you would like to see it during processing.  For that
% reason I've sometimes wondered if we should have a different category of
% modules, 'Display Modules' (right now most of them live in 'Other'),
% since OverlayOutlines doesn't actually do any image processing; I think
% it's a little misleading.
%
% Anne 4-9-09: I agree that this module could go into a new Display
% category and that the other modules in the Other category that begin with
% "Display" could be moved. I suppose we still want to keep the "Display"
% prefix in their names even though they will be in a new Display category.
% This module could be DisplayOutlines rather than OverlayOutlines. See the
% PyCP settings wiki for more discussion of this.
%
% I'm not sure how to handle the situation where users are confused,
% thinking they must overlay outlines with this module in order to save the
% outlines. I guess there isn't really much harm in them doing so...
% perhaps we could focus on just making it more obvious in Identify modules
% how to 'save' the outlines (to the handles structure) for later use.
%
% Re: color: we should make sure that the module handles things properly;
% for example, right now it says "for color images" choose the outline
% color. We should make sure that color outlines work even if the user has
% chosen a grayscale image to put the outlines onto.
%
%
% MBray 2009_04_16: Comments on variables for pyCP upgrade
% A button should sbe added that lets the user add/subtract outlines to 
% overlay. Right now, we have to string multiple modules together to create
% pictures with multiple outlines in different colors.
%
% See also the ConvertToImage module for some relevant discussion.

drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = On which image would you like to display the outlines?
%choiceVAR01 = Blank
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the outlines that you would like to display?
%infotypeVAR02 = outlinegroup
OutlineName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = Would you like the intensity (brightness) of the outlines to be the same as the brightest point in the image, or the maximum possible value for this image format? Note: if you chose to display on a Blank image, this will default to Max possible.
%choiceVAR03 = Max of image
%choiceVAR03 = Max possible
MaxType = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What do you want to call the image with the outlines displayed?
%defaultVAR04 = Do not use
%infotypeVAR04 = imagegroup indep
SavedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For color images, what do you want the color of the outlines to be?
%choiceVAR05 = White
%choiceVAR05 = Black
%choiceVAR05 = Red
%choiceVAR05 = Green
%choiceVAR05 = Blue
%choiceVAR05 = Yellow
%inputtypeVAR05 = popupmenu
OutlineColor = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(ImageName,'Blank')
    OutlineImage = CPretrieveimage(handles,OutlineName,ModuleName,'MustBeGray','DontCheckScale');
    OrigImage = zeros(size(OutlineImage));
else
    OrigImage = CPretrieveimage(handles,ImageName,ModuleName);
    OutlineImage = CPretrieveimage(handles,OutlineName,ModuleName,'MustBeGray','DontCheckScale',size(OrigImage));
end


if size(OrigImage,3) ~= 3
    if strcmp(MaxType,'Max of image') && ~strcmp(ImageName,'Blank')
        ValueToUseForOutlines = max(max(OrigImage));
    elseif strcmp(MaxType,'Max possible') || strcmp(ImageName,'Blank')
        if isfloat(OrigImage(1,1)) || islogical(OrigImage(1,1))
            ValueToUseForOutlines = 1;
        else
            ValueToUseForOutlines = intmax(class(OrigImage(1,1)));
        end
    else
        error(['Image processing was canceled in the ', ModuleName, ' module because the value of MaxType was not recognized.']);
    end

    NewImage = OrigImage;
    NewImage(OutlineImage ~= 0) = ValueToUseForOutlines;
else
    Color1 = 1;
    Color2 = 1;
    Color3 = 1;
    if strcmpi(OutlineColor,'Black')
        Color1 = 0;
        Color2 = 0;
        Color3 = 0;
    elseif strcmpi(OutlineColor,'Red')
        Color2 = 0;
        Color3 = 0;
    elseif strcmpi(OutlineColor,'Green')
        Color1 = 0;
        Color3 = 0;
    elseif strcmpi(OutlineColor,'Blue')
        Color1 = 0;
        Color2 = 0;
    elseif strcmpi(OutlineColor,'Yellow')
        Color3 = 0;
    end
    NewImage1 = OrigImage(:,:,1);
    NewImage2 = OrigImage(:,:,2);
    NewImage3 = OrigImage(:,:,3);
    NewImage1(OutlineImage ~= 0) = Color1;
    NewImage2(OutlineImage ~= 0) = Color2;
    NewImage3(OutlineImage ~= 0) = Color3;
    NewImage(:,:,1) = NewImage1;
    NewImage(:,:,2) = NewImage2;
    NewImage(:,:,3) = NewImage3;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    FigHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);
    hAx = axes('parent',FigHandle);
    CPimagesc(NewImage,handles,hAx);
    title(hAx,['Original Image with Outline Overlay, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    uicontrol(FigHandle,'units','normalized','position',[.01 .5 .06 .04],'string','off',...
        'UserData',{OrigImage NewImage},'backgroundcolor',[.7 .7 .9],...
        'Callback',@CP_OrigNewImage_Callback);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if ~strcmp(SavedImageName,'Do not use')
    handles = CPaddimages(handles,SavedImageName,NewImage);
end
