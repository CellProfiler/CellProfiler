function handles = Crop(handles)

% Help for the Crop module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Images can be cropped into a rectangle, ellipse, an arbitrary shape
% provided by the user, a shape identified by an identify module, or a
% shape used at a previous step in the pipeline on another image.
% *************************************************************************
%
% Warnings about this module: The cropping module will remove rows and
% columns that are completely blank. Also, keep in mind that cropping
% changes the size of your images, which may have unexpected consequences.
% For example, identifying objects in a cropped image and then trying to
% measure their intensity in the *original* image will not work because the
% two images are not the same size.
% 
% Settings:
%
% Shape:
% Rectangle - self-explanatory.
% Ellipse - self-explanatory.
% Other...
% To crop based on an object identified in a previous module, type in the
% name of that identified object instead of Rectangle or Ellipse. Please
% see PlateFix for information on cropping based on previously identified
% plates.
% To crop into an arbitrary shape you define, use the LoadSingleImage
% module to load a black and white image (that you have already prepared)
% from a file. If you have created this image in an image program such as
% Photoshop, this binary image should contain only the values 0
% and 255, with zeros (black) for the parts you want to remove and 255
% (white) for the parts you want to retain.  Or, you may have previously
% generated a binary image using this module (e.g. using the ellipse
% option) and saved it using the SaveImages module (see Special note on
% saving images below).  In any case, the image must be the exact same
% starting size as your image and should contain a contiguous block of
% white pixels, because keep in mind that the cropping module will remove
% rows and columns that are completely blank.
% To crop into the same shape as was used previously in the pipeline to
% crop another image, type in CroppingPreviousCroppedImageName, where
% PreviousCroppedImageName is the image you produced with the previous Crop
% module.
%
% Coordinate or mouse: For ellipse, you will be asked to click five or more
% points to define an ellipse around the part of the image you want to
% analyze.  Keep in mind that the more points you click, the longer it will
% take to calculate the ellipse shape. For rectangle, you can click as many
% points as you like that are in the interior of the region you wish to
% retain.
%
% PlateFix: To be used only when cropping based on previously identified
% objects. When attempting to crop based on a previously identified object
% (such as a yeast plate), sometimes the identified plate does not have
% precisely straight edges - there might be a tiny, almost unnoticeable
% 'appendage' sticking out of the plate.  Without plate fix, the crop
% module would not crop the image tightly enough - it would include enough
% of the image to retain even the tiny appendage, so there would be a lot
% of blank space around the plate. This can cause problems with later
% modules (especially IlluminationCorrection). PlateFix takes the
% identified object and crops to exclude any minor appendages (technically,
% any horizontal or vertical line where the object covers less than 50% of
% the image). It also sets pixels around the edge of the object (for
% regions > 50% but less than 100%) that otherwise would be zero to the
% background pixel value of your image thus avoiding the problems with
% other modules. Important note >> PlateFix uses the coordinates
% entered in the boxes normally used for rectangle cropping (Top, Left) and
% (Bottom, Right) to tighten the edges around your identified plate. This
% is done because in the majority of plate identifications you do not want
% to include the sides of the plate. If you would like the entire plate to
% be shown, you should enter 1:end for both coordinates. If you would like
% to crop 80 pixels from each edge of the plate, you could enter 80:end-80
% for (Top, Left) and (Bottom, Right).
%
% Special note on saving images: See the help for SaveImages. Also, you can
% save the cropping shape that you have used (e.g. an ellipse you drew), so
% that in future analyses you can use the File option.  To do this, you
% need to add the prefix "Cropping" to the name you called the cropped
% image (e.g. CroppingCropBlue) and this is the name of the image you will
% want to save using the SaveImages module.  I think you will want to save
% it in mat format. You can also save the cropping shape, trimmed for any
% unused rows and columns at the edges.  This image has the prefix
% "CropMask" plus the name you called the cropped image (e.g.
% CropMaskCropBlue).  This image is used for downstream modules that use
% the CPgraythresh function.  The Cropping and CropMask images are similar
% (both are binary and contain the cropping shape you used), but the
% Cropping image is the same size as the original images to be processed
% whereas the CropMask image is the same size as the final, cropped image.
%
% See also <nothing relevant>.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
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

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be cropped?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the cropped image?
%defaultVAR02 = CropBlue
%infotypeVAR02 = imagegroup indep
CroppedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Into which shape would you like to crop? See the help for several other options.
%choiceVAR03 = Rectangle
%choiceVAR03 = Ellipse
Shape = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Would you like to crop by typing in pixel coordinates or clicking with the mouse?
%choiceVAR04 = Coordinates
%choiceVAR04 = Mouse
CropMethod = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Would you like to perform the step above (typing or clicking) for each image individually or just once then apply the settings to subsequent image cycles?
%choiceVAR05 = Just Once
%choiceVAR05 = Individually
IndividualOrOnce = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = For COORDINATES + JUST ONCE + RECTANGLE, specify the (Left, Right) pixel positions (the word "end" can be substituted for right pixel if you do not want to crop the right edge)
%defaultVAR06 = 1,100
Pixel1 = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = For COORDINATES + JUST ONCE + RECTANGLE, specify the (Top, Bottom) pixel positions (the word "end" can be substituted for bottom pixel if you do not want to crop the bottom edge)
%defaultVAR07 = 1,100
Pixel2 = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = For COORDINATES + JUST ONCE + ELLIPSE, what is the center pixel position of the ellipse in form X,Y?
%defaultVAR08 = 500,500
Center = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = For COORDINATES + JUST ONCE + ELLIPSE, what is the radius of the ellipse in the X direction?
%defaultVAR09 = 400
X_axis = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = For COORDINATES + JUST ONCE + ELLIPSE, what is the radius of the ellipse in the Y direction?
%defaultVAR10 = 200
Y_axis = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Do you want to use Plate Fix? (see Help, only used when cropping based on previously identified objects)
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
%%% Checks whether the image to be analyzed exists in the handles structure.
if ~isfield(handles.Pipeline, ImageName)
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(ImageName);
if max(OrigImage(:)) > 1 || min(OrigImage(:)) < 0
    CPwarndlg(['The images you have loaded in the ', ModuleName, ' module  are outside the 0-1 range, and you may be losing data.'],'Outside 0-1 Range','replace');
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

CropFromObjectFlag = 0;

if handles.Current.SetBeingAnalyzed == 1 || strcmp(IndividualOrOnce, 'Individually') || strcmp(PlateFix,'Yes')

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
                    catch error(['Image processing was canceled in the ', ModuleName, ' module because the image to be used for cropping cannot be found.']);
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
                error(['Image processing was canceled in the ', ModuleName, ' module because the format of the Left, Right pixel positions is invalid. Please include a comma.']);
            end
            x1 = str2double(Pixel1(1:index-1));
            x2 = Pixel1(index+1:end);
            if strncmp(x2,'end',3)
                if length(x2) == 3
                    x2 = LastX-FirstX;
                else
                    if length(x2) >= 5
                        x2 = max(1,LastX-FirstX-str2double(x2(5:end)));
                    else
                        x2 = LastX-FirstX;
                    end
                end
            else
                x2 = min(LastX-FirstX,str2double(x2));
            end

            index = strfind(Pixel2,',');
            if isempty(index)
                error(['Image processing was canceled in the ', ModuleName, ' module because the format of the Top, Bottom pixel positions is invalid. Please include a comma.']);
            end
            y1 = str2double(Pixel2(1:index-1));
            y2 = Pixel2(index+1:end);
            if strncmp(y2,'end',3)
                if length(y2) == 3
                    y2 = LastY-FirstY;
                else
                    if length(y2) >= 5
                        y2 = max(1,LastY-FirstY-str2double(y2(5:end)));
                    else
                        x2 = LastY-FirstY;
                    end
                end
            else
                y2 = min(LastY-FirstY,str2double(y2));
            end

            %%% Combining identified object boundaries with user-specified
            %%% values
            try
                Pixel1 = [num2str(FirstX+x1),',',num2str(FirstX+x2)];
                Pixel2 = [num2str(FirstY+y1),',',num2str(FirstY+y2)];
            catch
                error(['Image processing was canceled in the ', ModuleName, ' module because there was a problem finding the cropping boundaries from the previously identified object. PlateFix is currently hard-coded to crop based on objects which occupy at least 50% of the field of view. If your object is smaller than this, it will fail.']);
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
                    catch error(['Image processing was canceled in the ', ModuleName, ' module because the image to be used for cropping cannot be found.']);
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
            [Pre_x,Pre_y] = getpts(CroppingFigureHandle);
            [a b c] = size(ImageToBeCropped);
            if any(Pre_x < 1) || any(Pre_y < 1) || any(Pre_x > b) || any(Pre_y > a)
                Pre_x(Pre_x<1) = 1;
                Pre_x(Pre_x>b) = b-1;
                Pre_y(Pre_y<1) = 1;
                Pre_y(Pre_y>a) = a-1;
                CPmsgbox('You have chosen points outside of the range of the image. These points have been rounded to the closest compatible number.');
            end
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
            [X,Y] = meshgrid(1:masksize(1), 1:masksize(2));
            X = X - mean_x;
            Y = Y - mean_y;
            drawnow
            %%% Produces the BinaryCropImage.
            BinaryCropImage = ((params(1) * (X .* X) + params(2) * (X .* Y) + params(3) * (Y .* Y) + params(4) * X + params(5) * Y) < 1);
            %%% Need to flip X and Y. Why? It doesnt work.
            BinaryCropImage = BinaryCropImage';
        elseif strcmp(CropMethod,'Coordinates')

            if strcmp(IndividualOrOnce,'Individually')
                Answers = inputdlg({'What is the center in the form X,Y?' 'What is the length of the radius along the X-axis?' 'What is the length of the radius along the Y-axis?'});
                Center = Answers{1};
                X_axis = Answers{2};
                Y_axis = Answers{3};
            end

            index = strfind(Center,',');
            if isempty(index)
                error(['Image processing was canceled in the ', ModuleName, ' module because the format of the center of the ellipse is invalid. Please include a comma.']);
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
            error(['Image processing was canceled in the ', ModuleName, ' module because your entry for the cropping method is not recognized']);
        end
        handles.Pipeline.(['Cropping' CroppedImageName]) = BinaryCropImage;
        [handles, CroppedImage, BinaryCropImage] = CropImageBasedOnMaskInHandles(handles,OrigImage,CroppedImageName,ModuleName);
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
                error(['Image processing was canceled in the ', ModuleName, ' module because the format of the Left, Right pixel positions is invalid. Please include a comma.']);
            end
            x1 = Pixel1(1:index-1);
            x2 = Pixel1(index+1:end);

            index = strfind(Pixel2,',');
            if isempty(index)
                error(['Image processing was canceled in the ', ModuleName, ' module because the format of the Top, Bottom pixel positions is invalid. Please include a comma.']);
            end
            y1 = Pixel2(1:index-1);
            y2 = Pixel2(index+1:end);

            [a b c] = size(ImageToBeCropped); %#ok
            BinaryCropImage = zeros(a,b);
            eval(['BinaryCropImage(min(' y1 ',' y2 '):max(' y1 ',' y2 '),min(' x1 ',' x2 '):max(' x1 ',' x2 ')) = 1;']);

        elseif strcmp(CropMethod,'Mouse')
            %%% Displays the image and asks the user to choose points.
            CroppingFigureHandle = CPfigure(handles,'name','Manual Rectangle Cropping');
            CroppingImageHandle = imagesc(ImageToBeCropped);
            colormap('gray');
            title({'Click on at least two points that are inside the region to be retained'; '(e.g. top left and bottom right point) & then press Enter.'; 'Press delete to erase the most recently clicked point.'})
            [x,y] = getpts(CroppingFigureHandle);
            close(CroppingFigureHandle);

            [a b c] = size(ImageToBeCropped);
            if any(x < 1) || any(y < 1) || any(x > b) || any(y > a)
                x(x<1) = 1;
                x(x>b) = b-1;
                y(y<1) = 1;
                y(y>a) = a-1;
                CPmsgbox('You have chosen points outside of the range of the image. These points have been rounded to the closest compatible number.');
            end
            BinaryCropImage = zeros(a,b);
            BinaryCropImage(round(min(y)):round(max(y)),round(min(x)):round(max(x))) = 1;
        else
            error(['Image processing was canceled in the ', ModuleName, ' module because your entry for the cropping method is not recognized']);
        end
        handles.Pipeline.(['Cropping' CroppedImageName]) = BinaryCropImage;
        [handles, CroppedImage, BinaryCropImage] = CropImageBasedOnMaskInHandles(handles, OrigImage,CroppedImageName, ModuleName);
    end
    %%% See subfunction below.
else
    [handles, CroppedImage, BinaryCropImage] = CropImageBasedOnMaskInHandles(handles,OrigImage,CroppedImageName,ModuleName);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
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
    CPimagesc(OrigImage);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    CPimagesc(CroppedImage);
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
    BinaryCropImage = handles.Pipeline.(['Cropping',CroppedImageName]);
catch
    try
        BinaryCropImage = handles.Pipeline.(CroppedImageName);
    catch
        error(['Image processing was canceled in the ', ModuleName, ' module because you must choose rectangle, ellipse or the name of something from a previous module.']);
    end
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
%%% structure so it can be used in subsequent cycles to
%%% show which parts of the image were cropped (this will be used
%%% by CPthreshold).
BinaryCropMaskImage = BinaryCropImage;
BinaryCropMaskImage(:,ColumnsToDelete,:) = [];
BinaryCropMaskImage(RowsToDelete,:,:) = [];
fieldname = ['CropMask',CroppedImageName];
handles.Pipeline.(fieldname) = BinaryCropMaskImage;