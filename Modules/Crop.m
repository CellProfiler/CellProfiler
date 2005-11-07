function handles = Crop(handles)

% Help for the Crop module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Images can be cropped into a rectangle, ellipse, an arbitrary shape
% provided by the user, or a shape identified by an identify module.
% *************************************************************************
%
% Allows the images to be cropped in any shape:
%
% Shape: You can crop by rectangular coordinates, ellipse, or by an object
% identified in a previous module. To crop by an object, simply type in the
% name of that identified object instead of Rectangle or Ellipse. Please
% see PlateFix for information on identifying crop plates.
%
% Rectangle: enter the pixel coordinates for the left to right and
% top to bottom pixels, and every image will be cropped at these
% locations.  For the right and bottom pixels, you can type "end" instead
% of a numerical pixel position if you want the right-most or
% bottom-most pixel position to be calculated automatically.
%
% Ellipse: You will be asked to click five or more points to define an
% ellipse around the part of the image you want to analyze.  Keep in
% mind that the more points you click, the longer it will take to
% calculate the ellipse shape.
%
% Individually or Once: This option will allow you to apply the same
% cropping to all images or define each image separately.
%
% File: You can crop images to any arbitrary shape by loading that
% shape into CellProfiler. Use the LoadSingleImage module to load the
% black and white image (that you have already prepared) from a file.
% If you have created this image in an image program such as
% Photoshop, this binary image should actually contain only the values
% 0 and 255, with zeros (black) for the parts you want to remove and
% 255 (white) for the parts you want to retain.  Or, you may have
% previously generated a binary image using this module (e.g. using
% the ellipse option) and saved it using the SaveImages module (see
% SAVING IMAGES below).  In any case, the image must be the exact same
% starting size as your image and should contain a contiguous block of
% white pixels, because keep in mind that the cropping module will
% remove rows and columns that are completely blank.
%
% PlateFix: When cropping identified plates, sometimes pixels within the
% object are set to zero. This can cause problems with later modules
% (especially IlluminationCorrection). PlateFix takes the identified object
% and finds where the object covers at least 50% of the image in both the
% horizontal and vertical directions. It then crops a rectangle based on
% this information. This sets pixels that otherwise would be zero to the
% background pixel value of your image thus avoiding the problems with
% other modules. PlateFix also uses the coordinates for rectangle cropping
% to tighten the edges around your identified plate. This is done because
% in the majority of plate identifications you do not want to include the
% sides of the plate. If you would like the entire plate to be shown, you
% can enter 1:end for both coordinates.
%
% Warning: Keep in mind that cropping changes the size of your images,
% which may have unexpected consequences.  For example, identifying
% objects in a cropped image and then trying to measure their
% intensity in the original image will not work because the two images
% are not the same size. As another example, identify primary modules
% ignore objects that touch the outside edge of the image because they
% would be partial objects and therefore not measured properly.
% However, if you crop a round shape, the edge is still officially the
% square edge of the image, and not the round contour, so partial
% objects will be included.
%
% SAVING IMAGES: The cropped images produced by this module can be
% easily saved using the Save Images module, using the name you
% assign. If you want to save other intermediate images, alter the
% code for this module to save those images to the handles structure
% (see the SaveImages module help) and then use the Save Images
% module. You can save the cropping shape that you have used (e.g. an
% ellipse you drew), so that in future analyses you can use the File
% option.  To do this, you need to add the prefix "Cropping" to the
% name you called the cropped image (e.g. CroppingCropBlue) and this
% is the name of the image you will want to save using the SaveImages
% module.  I think you will want to save it in mat format. You can
% also save the cropping shape, trimmed for any unused rows and
% columns at the edges.  This image has the prefix "CropMask" plus the
% name you called the cropped image (e.g. CropMaskCropBlue).  This
% image is used for downstream modules that use the CPgraythresh
% function.  The Cropping and CropMask images are similar (both are
% binary and contain the cropping shape you used), but the Cropping
% image is the same size as the original images to be processed
% whereas the CropMask image is the same size as the final, cropped
% image.
%
% See also <nothing relevant>.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What did you call the image to be cropped?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the cropped image?
%defaultVAR02 = CropBlue
%infotypeVAR02 = imagegroup indep
CroppedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Into which shape would you like to crop? To crop to a shape based on another image you have loaded, type its name here (see help for details).
%choiceVAR03 = Rectangle
%choiceVAR03 = Ellipse
Shape = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Would you like to crop by coordinates or mouse?
%choiceVAR04 = Coordinates
%choiceVAR04 = Mouse
CropMethod = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Would you like to crop each image individually?
%choiceVAR05 = Just Once
%choiceVAR05 = Individually
IndividualOrOnce = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Specify the (Left, Right) pixel positions. (only if you are using rectangle, coordinates, and Just Once)(end can be substituted for right pixel if you do not want to crop the right edge)
%defaultVAR06 = 1,100
Pixel1 = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Specify the (Top, Bottom) pixel positions. (only if you are using rectangle, coordinates, and Just Once)(end can be substituted for bottom pixel if you do not want to crop the bottom edge)
%defaultVAR07 = 1,100
Pixel2 = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What is the center of the ellipse in form X,Y? (only if you are using ellipse, coordinates, and Just Once)
%defaultVAR08 = 500,500
Center = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What is the radius of the X axis? (only if you are using ellipse, coordinates, and Just Once)
%defaultVAR09 = 400
X_axis = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = What is the radius of the Y axis? (only if you are using ellipse, coordinates, and Just Once)
%defaultVAR10 = 200
Y_axis = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Do you want to use Plate Fix? (see Help)
%choiceVAR11 = No
%choiceVAR11 = Yes
%inputtypeVAR11 = popupmenu
PlateFix = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image to be analyzed and assigns it to a variable,
%%% "OrigImage".
fieldname = ['', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if ~isfield(handles.Pipeline, fieldname)
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(fieldname);
if max(OrigImage(:)) > 1 || min(OrigImage(:)) < 0
    CPwarndlg('The images you have loaded are outside the 0-1 range, and you may be losing data.','Outside 0-1 Range','replace');
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

CropFromObjectFlag = 0;

if handles.Current.SetBeingAnalyzed == 1 || strcmp(IndividualOrOnce, 'Individually') || strcmp(PlateFix,'Yes')
    %%% Not sure when this code was used, but IndividualOrOnce can never be
    %%% 'Only Once' so I have commented it out to see if this creates
    %%% errors anywhere.
    %     if strcmp(IndividualOrOnce, 'Only Once')
    %         %%% The user can choose an image from the pipeline or from the hard drive to use for
    %         %%% cropping.
    %         Answer = CPquestdlg('Choose an image to be used for cropping...','Select image','Image file from hard drive','Image from this image set','Image from this image set');
    %         if strcmp(Answer,'Cancel')
    %             error(['Image processing was canceled by the user in the ', ModuleName, ' module.']);
    %         elseif strcmp(Answer,'Image from this image set')
    %             try
    %                 ImageToBeCropped = OrigImage;
    %             catch
    %                 error(['Image processing was canceled in the ', ModuleName, ' module because you did not select a valid image to use for cropping in the Crop module.'])
    %             end
    %         elseif strcmp(Answer,'Image file from hard drive')
    %             %%% Asks the user to open an image file upon which to draw the
    %             %%% ellipse.
    %             %%% Opens a user interface window which retrieves a file name and path
    %             %%% name for the image to be used as a test image.
    %             [CroppingFileName,CroppingPathname] = uigetfile(fullfile(handles.Current.DefaultImageDirectory,'.','*'),'Select the image to use for cropping');
    %             %%% If the user presses "Cancel", the FileName will = 0 and an error
    %             %%% message results.
    %             if CroppingFileName == 0
    %                 error(['Image processing was canceled in the ', ModuleName, ' module because you did not select an image to use for cropping in the Crop module.'])
    %             else
    %                 [ImageToBeCropped, handles] = CPimread(fullfile(CroppingPathname,CroppingFileName), handles);
    %             end
    %         end
    %     else
    %     end
    ImageToBeCropped = OrigImage;

    if ~strcmp(Shape,'Ellipse') && ~strcmp(Shape,'Rectangle')
        if strcmp(PlateFix,'Yes')
            try BinaryCropImage = handles.Pipeline.(Shape);
            catch
                fieldname = ['Segmented',Shape];
                try BinaryCropImage = handles.Pipeline.(fieldname);
                catch
                    fieldname = ['Cropping',Shape];
                    try BinaryCropImage = handles.Pipeline.(fieldname);
                    catch error('Image cannot be found!');
                    end
                end
            end
            [m,n] = size(BinaryCropImage);
            flag = 0;
            for i = 1:m
                if sum(BinaryCropImage(i,:))/n >= .5
                    LastY = i;
                    if flag == 0
                        FirstY = i;
                        flag = 1;
                    end
                end
            end
            flag = 0;
            for i = 1:n
                if sum(BinaryCropImage(:,i))/m >= .5
                    LastX = i;
                    if flag == 0
                        FirstX = i;
                        flag = 1;
                    end
                end
            end

            %%% Using PlateFix, we get the pixel values specified by the
            %%% user and combine them with those calculated from the
            %%% identified object
            index = strfind(Pixel1,',');
            if isempty(index)
                error('The format of the Left, Right pixel positions is invalid. Please include a comma.');
            end
            x1 = str2num(Pixel1(1:index-1));
            x2 = Pixel1(index+1:end);
            if strcmp(x2,'end')
                x2 = LastX-FirstX;
            else
                x2 = min(LastX-FirstX,str2num(x2));
            end
            
            
            
            index = strfind(Pixel2,',');
            if isempty(index)
                error('The format of the Top, Bottom pixel positions is invalid. Please include a comma.');
            end
            y1 = str2num(Pixel2(1:index-1));
            y2 = Pixel2(index+1:end);
            if strcmp(y2,'end')
                y2 = LastY-FirstY;
            else
                y2 = min(LastY-FirstY,str2num(y2));
            end

            %%% Combining identified object boundaries with user-specified
            %%% values
            try
                Pixel1 = [num2str(FirstX+x1),',',num2str(FirstX+x2)];
                Pixel2 = [num2str(FirstY+y1),',',num2str(FirstY+y2)];
            catch
                error('There was a problem finding the X and Y pixels from the object. PlateFix is currently hard-coded to crop based on objects which occupy at least 50% of the field of vue. If your object is smaller than this, it will fail.');
            end
            Shape = 'Rectangle';
            CropFromObjectFlag = 1;
        else
            CropFromObjectFlag = 1;
            try [handles, CroppedImage, BinaryCropImage] = CropImageBasedOnMaskInHandles(handles,OrigImage,Shape,ModuleName);
            catch
                try [handles, CroppedImage, BinaryCropImage] = CropImageBasedOnMaskInHandles(handles,OrigImage,['Segmented',Shape], ModuleName);
                catch
                    try [handles, CroppedImage, BinaryCropImage] = CropImageBasedOnMaskInHandles(handles,OrigImage,['Cropping',Shape], ModuleName);
                    catch error('Image cannot be found!');
                    end
                end
            end
            
        handles.Pipeline.(['Cropping' CroppedImageName]) = BinaryCropImage;

        end
    end

    if strcmp(Shape, 'Ellipse')
        if strcmp(CropMethod,'Mouse')
            %%% Displays the image and asks the user to choose points for the
            %%% ellipse.
            CroppingFigureHandle = CPfigure(handles);
            CroppingImageHandle = imagesc(ImageToBeCropped);
            pixval
            title({'Click on 5 or more points to be used to create a cropping ellipse & then press Enter.'; 'Press delete to erase the most recently clicked point.'})
            try imcontrast(CroppingImageHandle); end %#ok
            try imcontrast(CroppingImageHandle); end %#ok
            [Pre_x,Pre_y] = getpts(CroppingFigureHandle);
            close(CroppingFigureHandle)
            x = Pre_y;
            y = Pre_x;
            drawnow
            %%% Removes bias of the ellipse - to make matrix inversion more
            %%% accurate. (will be added later on) (Not really sure what this
            %%% is doing).
            mean_x = mean(x);
            mean_y = mean(y);
            New_x = x-mean_x;
            New_y = y-mean_y;
            %%% the estimation for the conic equation of the ellipse
            X = [New_x.^2, New_x.*New_y, New_y.^2, New_x, New_y ];
            params = sum(X)/(X'*X);
            masksize = size(ImageToBeCropped);
            [X,Y] = meshgrid(1:masksize(2), 1:masksize(1));
            X = X - mean_x;
            Y = Y - mean_y;
            drawnow
            %%% Produces the BinaryCropImage.
            BinaryCropImage = ((params(1) * (X .* X) + ...
                params(2) * (X .* Y) + ...
                params(3) * (Y .* Y) + ...
                params(4) * X + ...
                params(5) * Y) < 1);
            %%% Need to flip X and Y. Why? It doesnt work.
            % BinaryCropImage = BinaryCropImage';
        elseif strcmp(CropMethod,'Coordinates')

            if strcmp(IndividualOrOnce,'Individually')
                Answers = inputdlg({'What is the center in the form X,Y?' 'What is the length of the radius along the X-axis?' 'What is the length of the radius along the Y-axis?'});
                Center = Answers{1};
                X_axis = Answers{2};
                Y_axis = Answers{3};
            end

            index = strfind(Center,',');
            if isempty(index)
                error('The format of the center is invalid. Please include a comma.');
            end
            X_center = Center(1:index-1);
            Y_center = Center(index+1:end);

            masksize = size(ImageToBeCropped); %#ok
            [X,Y] = meshgrid(1:masksize(2), 1:masksize(1)); %#ok
            if eval([X_axis '>' Y_axis])
                eval(['foci_1_x = ' X_center '+ sqrt(' X_axis '^2-' Y_axis '^2);']);
                eval(['foci_2_x = ' X_center '- sqrt(' X_axis '^2-' Y_axis '^2);']);
                eval(['foci_1_y = ' Y_center ';']);
                eval(['foci_2_y = ' Y_center ';']);
                eval(['BinaryCropImage = sqrt((X-foci_1_x).^2+(Y-foci_1_y).^2)+sqrt((X-foci_2_x).^2+(Y-foci_2_y).^2) < 2*' X_axis ';']);
            else
                eval(['foci_1_x = ' X_center ';']);
                eval(['foci_2_x = ' X_center ';']);
                eval(['foci_1_y = ' Y_center ' + sqrt(' Y_axis '^2-' X_axis '^2);']);
                eval(['foci_2_y = ' Y_center ' - sqrt(' Y_axis '^2-' X_axis '^2);']);
                eval(['BinaryCropImage = sqrt((X-foci_1_x).^2+(Y-foci_1_y).^2)+sqrt((X-foci_2_x).^2+(Y-foci_2_y).^2) < 2*' Y_axis ';']);
            end
        else
            error('The value of CropMethod is not recognized');
        end
        handles.Pipeline.(['Cropping' CroppedImageName]) = BinaryCropImage;
        [handles, CroppedImage, BinaryCropImage] = CropImageBasedOnMaskInHandles(handles, OrigImage, ['Cropping',CroppedImageName], ModuleName);
    elseif strcmp(Shape,'Rectangle')
        if strcmp(CropMethod,'Coordinates')

            if strcmp(IndividualOrOnce,'Individually') && (CropFromObjectFlag == 0)
                %%% Displays the image so that you can see which
                %%% pixel positions you want to use to crop. But wait,
                %%% you would probably just use the Mouse option
                %%% instead, so I've commented it out.
                %                 TempFigHandle = figure, imagesc(ImageToBeCropped), title('Close this window when you have identified the pixel positions to be used for cropping.'), pixval
                %                 waitfor(TempFigHandle)
                Answers = inputdlg({'Specify the (Left, Right) pixel positions:' 'Specify the (Top, Bottom) pixel positions:'});
                Pixel1=Answers{1};
                Pixel2=Answers{2};
            end

            index = strfind(Pixel1,',');
            if isempty(index)
                error('The format of the Left, Right pixel positions is invalid. Please include a comma.');
            end
            x1 = Pixel1(1:index-1);
            x2 = Pixel1(index+1:end);

            index = strfind(Pixel2,',');
            if isempty(index)
                error('The format of the Top, Bottom pixel positions is invalid. Please include a comma.');
            end
            y1 = Pixel2(1:index-1);
            y2 = Pixel2(index+1:end);

            [a b c] = size(ImageToBeCropped); %#ok
            BinaryCropImage = zeros(a,b);
            eval(['BinaryCropImage(min(' y1 ',' y2 '):max(' y1 ',' y2 '),min(' x1 ',' x2 '):max(' x1 ',' x2 ')) = 1;']);

        elseif strcmp(CropMethod,'Mouse')
            %%% Displays the image and asks the user to choose points.
            CroppingFigureHandle = figure;
            CroppingImageHandle = imagesc(ImageToBeCropped);
            colormap('gray'); pixval
            title({'Click on at least two points that are inside the region to be retained (e.g. top left and bottom right point) & then press Enter.'; 'Press delete to erase the most recently clicked point.'})
            try imcontrast(CroppingImageHandle); end %#ok
            try imcontrast(CroppingImageHandle); end %#ok
            [x,y] = getpts(CroppingFigureHandle);
            close(CroppingFigureHandle);

            [a b c] = size(ImageToBeCropped);
            BinaryCropImage = zeros(a,b);
            BinaryCropImage(round(min(y)):round(max(y)),round(min(x)):round(max(x))) = 1;
        else
            error('The value of CropMethod is not recognized');
        end
        handles.Pipeline.(['Cropping' CroppedImageName]) = BinaryCropImage;
        [handles, CroppedImage, BinaryCropImage] = CropImageBasedOnMaskInHandles(handles, OrigImage, ['Cropping',CroppedImageName], ModuleName);
    end
    %%% See subfunction below.
else
    [handles, CroppedImage, BinaryCropImage] = CropImageBasedOnMaskInHandles(handles, OrigImage, ['Cropping',CroppedImageName], ModuleName);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        %%% Sets the window to be half as wide as usual.
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 250;
        set(ThisModuleFigureNumber, 'position', newsize);
    end

    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1);
    ImageHandle = imagesc(OrigImage);
    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)');
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    ImageHandle = imagesc(CroppedImage);
    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)');
    title('Cropped Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(CroppedImageName) = CroppedImage;

function [handles, CroppedImage, BinaryCropImage] = CropImageBasedOnMaskInHandles(handles, OrigImage, CroppedImageName, ModuleName)
%%% Retrieves the Cropping image from the handles structure.
try
    BinaryCropImage = handles.Pipeline.(CroppedImageName);
catch
    error(['You must choose rectangular cropping, ellipse or the name of something from a previous module.']);
end

if size(OrigImage(:,:,1)) ~= size(BinaryCropImage(:,:,1))
    error(['Image processing was canceled in the ', ModuleName, ' module because an image you wanted to analyze is not the same size as the image used for cropping.  The pixel dimensions must be identical.'])
end
%%% Sets pixels in the original image to zero if those pixels are zero in
%%% the binary image file.
PrelimCroppedImage = OrigImage;
ImagePixels = size(PrelimCroppedImage,1)*size(PrelimCroppedImage,2);
for Channel = 1:size(PrelimCroppedImage,3),
    PrelimCroppedImage((Channel-1)*ImagePixels + find(BinaryCropImage == 0)) = 0;
end
drawnow
%%% Removes Rows and Columns that are completely blank.
ColumnTotals = sum(BinaryCropImage,1);
RowTotals = sum(BinaryCropImage,2)';
warning off all
ColumnsToDelete = ~logical(ColumnTotals);
RowsToDelete = ~logical(RowTotals);
warning on all
drawnow
CroppedImage = PrelimCroppedImage;
CroppedImage(:,ColumnsToDelete,:) = [];
CroppedImage(RowsToDelete,:,:) = [];
%%% The Binary Crop Mask image is saved to the handles
%%% structure so it can be used in subsequent image sets to
%%% show which parts of the image were cropped (this will be used
%%% by CPgraythresh).
BinaryCropMaskImage = BinaryCropImage;
BinaryCropMaskImage(:,ColumnsToDelete,:) = [];
BinaryCropMaskImage(RowsToDelete,:,:) = [];
fieldname = ['CropMask', CroppedImageName];
handles.Pipeline.(fieldname) = BinaryCropMaskImage;