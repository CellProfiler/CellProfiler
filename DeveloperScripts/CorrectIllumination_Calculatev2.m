function handles = CorrectIllumination_Calculatev2(handles)

% Help for the Correct Illumination Calculate module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Calculates an illumination function, used to correct uneven
% illumination/lighting/shading or to reduce uneven background in images.
% *************************************************************************
%
% This module calculates an illumination function which can be saved
% later use (you should save in .mat format using the Save Images
% module), or it can be immediately applied to images later in the
% pipeline (using the CorrectIllumination_Apply module). This will
% correct for uneven illumination of each image.
%
% Illumination correction is challenging and we are writing a paper on it
% which should help clarify it (TR Jones, AE Carpenter, P Golland, in
% preparation). In the meantime, please be patient in trying to understand
% this module.
%
% Settings:
%
% XXXX - needs to be written
%
% See also Average, CorrectIllumination_Apply, and Smooth modules.

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
%   Kyungnam Kim
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Are you correcting Each image independently, or All images together?  See the help for details.
%choiceVAR01 = Each
%choiceVAR01 = All
%defaultVAR01 = All
EachOrAll = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = How smooth should the correction image be (filter diameter in pixels)?
%defaultVAR02 = 50
SmoothingDiameter = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,2}));

%textVAR03 = 

%textVAR04 = What is the first image (aka channel) to be corrected?
%infotypeVAR04 = imagegroup
ImageName1 = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = What type is the first image?
%choiceVAR05 = Fluorescent
%choiceVAR05 = Brightfield
%choiceVAR05 = Phase Contrast
%choiceVAR05 = DIC
%choiceVAR05 = Linear
%choiceVAR05 = Multiplicative
Modality1 = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = What do you want to call the first illumination correction image?
%defaultVAR06 = IllumCorrection
%infotypeVAR06 = imagegroup indep
IlluminationImageName1 = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = 

%textVAR08 = What is the second image (aka channel) to be corrected?
%choiceVAR08 = None
%infotypeVAR08 = imagegroup
ImageName2 = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu custom

%textVAR09 = What type is the second image?
%choiceVAR09 = Fluorescent
%choiceVAR09 = Brightfield
%choiceVAR09 = Phase Contrast
%choiceVAR09 = DIC
%choiceVAR09 = Linear
%choiceVAR09 = Multiplicative
Modality2 = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = What do you want to call the second illumination correction image?
%defaultVAR10 = Do not save
%infotypeVAR10 = imagegroup indep
IlluminationImageName2 = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = 

%textVAR12 = What is the third image (aka channel) to be corrected?
%infotypeVAR12 = imagegroup
%choiceVAR12 = None
ImageName3 = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu custom

%textVAR13 = What type is the third image?
%choiceVAR13 = Fluorescent
%choiceVAR13 = Brightfield
%choiceVAR13 = Phase Contrast
%choiceVAR13 = DIC
%choiceVAR13 = Linear
%choiceVAR13 = Multiplicative
Modality3 = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu

%textVAR14 = What do you want to call the third illumination correction image?
%defaultVAR14 = Do not save
%infotypeVAR14 = imagegroup indep
IlluminationImageName3 = char(handles.Settings.VariableValues{CurrentModuleNum,14});

%textVAR15 = 

%textVAR16 = What is the fourth image (aka channel) to be corrected?
%infotypeVAR16 = imagegroup
%choiceVAR16 = None
ImageName4 = char(handles.Settings.VariableValues{CurrentModuleNum,16});
%inputtypeVAR16 = popupmenu custom

%textVAR17 = What type is the fourth image?
%choiceVAR17 = Fluorescent
%choiceVAR17 = Brightfield
%choiceVAR17 = Phase Contrast
%choiceVAR17 = DIC
%choiceVAR17 = Linear
%choiceVAR17 = Multiplicative
Modality4 = char(handles.Settings.VariableValues{CurrentModuleNum,17});
%inputtypeVAR17 = popupmenu

%textVAR18 = What do you want to call the fourth illumination correction image?
%defaultVAR18 = Do not save
%infotypeVAR18 = imagegroup indep
IlluminationImageName4 = char(handles.Settings.VariableValues{CurrentModuleNum,18});


%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% fetch the images
Image(:,:,1) = PreTreatImage(CPretrieveimage(handles,ImageName1,ModuleName,'MustBeGray','CheckScale'), Modality1);


if ~ strcmp(ImageName2, 'None'),
    Image(:,:,end+1) = PreTreatImage(CPretrieveimage(handles,ImageName2,ModuleName,'MustBeGray','CheckScale'), Modality2);
end

if ~ strcmp(ImageName3, 'None'),
    Image(:,:,end+1) = PreTreatImage(CPretrieveimage(handles,ImageName3,ModuleName,'MustBeGray','CheckScale'), Modality3);
end

if ~ strcmp(ImageName4, 'None'),
    Image(:,:,end+1) = PreTreatImage(CPretrieveimage(handles,ImageName4,ModuleName,'MustBeGray','CheckScale'), Modality4);
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% Get the pixel locations
[i, j] = meshgrid(1:size(Image, 2), 1:size(Image, 1));
Locations = [j(:) i(:)];

%%% linearize the image data in columns
Samples = reshape(Image, size(Image, 1) * size(Image, 2), size(Image, 3));

%%% Remove any samples that are NaN or inf
GoodData = all(isfinite(Samples), 2);
Samples = Samples(GoodData, :);
Locations = Locations(GoodData, :);

%%% Get the image dimensions.
ImageSize = [size(Image, 1) size(Image, 2)];

%%% We just use 10 components.
NumberOfComponents = 10;

%%% Now there are two possiblities, depending on Each or All.  
if strcmp(EachOrAll, 'Each'),
    %%% If Each, we don't sample, we just pass the whole image into
    %%% the correction calculation.
    IlluminationField = CPcalc_illum_corrxn(Samples, Locations, SmoothingDiameter / 2.0, NumberOfComponents, ImageSize, false);
else
    %%% Otherwise, we sample randomly.  We want to keep a few
    %%% image-worths of pixels in the sample buffer.  The correction
    %%% algorithm can easily handle a few million pixels, so we'll use
    %%% 5x the size of the image.
    if (handles.Current.SetBeingAnalyzed == 1),
        handles.Pipeline.(IlluminationImageName1).Samples = Samples;
        handles.Pipeline.(IlluminationImageName1).Locations = Locations;
    elseif (handles.Current.SetBeingAnalyzed <= 5),
        handles.Pipeline.(IlluminationImageName1).Samples = [handles.Pipeline.(IlluminationImageName1).Samples; Samples];
        handles.Pipeline.(IlluminationImageName1).Locations = [handles.Pipeline.(IlluminationImageName1).Locations ; Locations];
    else
        %%% We need to randomly sample the right fraction of pixels
        %%% from this image and replace the corresponding number
        %%% randomly within the sample buffer.
        
        %%% Get the old samples
        SampleBuffer = handles.Pipeline.(IlluminationImageName1).Samples;
        LocationBuffer = handles.Pipeline.(IlluminationImageName1).Locations;

        %%% Seed the random number generator so this code is repeatable.
        RandState = rand('state');
        rand('state', handles.Current.SetBeingAnalyzed);
        
        %%% get a random ordering of the data
        [ignore, Order] = sort(rand(size(Samples, 1), 1));
        
        %%% Drop all but the fraction of the ordering we care about
        FractionToKeep = 5 / handles.Current.NumberOfImageSets;
        NewData = Order(1:ceil(FractionToKeep * length(Order)));

        %%% Sample the new data
        NewSamples = Samples(NewData, :);
        NewLocations = Locations(NewData, :);

        %%% Now choose random locations to replace in the sample buffer.
        [ignore, BufferOrder] = sort(rand(size(SampleBuffer, 1), 1));
        
        %%% Keep only how many we want to replace.
        OldData = BufferOrder(1:size(NewSamples, 1));
        
        %%% Replace the old data with new values.
        SampleBuffer(OldData, :) = Samples(NewData, :);
        LocationBuffer(OldData, :) = Locations(NewData, :);
        
        %%% Put the new buffers back into place
        handles.Pipeline.(IlluminationImageName1).Samples = SampleBuffer;
        handles.Pipeline.(IlluminationImageName1).Locations = LocationBuffer;

        %%% restore the random state
        rand('state', RandState);
    end
    
    %%% Is this the last image set?
    if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets,
        %%% Get the samples
        SampleBuffer = handles.Pipeline.(IlluminationImageName1).Samples;
        LocationBuffer = handles.Pipeline.(IlluminationImageName1).Locations;

        %%% This could happen if the images are all bad.
        assert(size(SampleBuffer, 1) > 0);

        ShowComputation = false;
        ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
        if any(findobj == ThisModuleFigureNumber)
            CPfigure(handles,'Correction',ThisModuleFigureNumber);
            ShowComputation = true;
        end


        %%% Calculate the illumination correction
        IlluminationField = CPcalc_illum_corrxn(SampleBuffer, LocationBuffer, SmoothingDiameter / 2.0, NumberOfComponents, ImageSize, ShowComputation);

    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if strcmp(EachOrAll, 'Each') | (handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets),
    handles.Pipeline.(IlluminationImageName1) = PostTreatImage(IlluminationField(:,:,1), Modality1);

    %%% We need this count variable because of the code above that fetches
    %%% images (see PRELIMINARY CALCULATIONS above)
    count = 2;
    
    if ~ strcmp(ImageName2, 'None')
        if (~ strcmp(IlluminationImageName2, 'Do not save')) ,
            handles.Pipeline.(IlluminationImageName2) = PostTreatImage(IlluminationField(:,:,count), Modality2);
        end
        count = count + 1;
    end
    
    
    if ~ strcmp(ImageName3, 'None')
        if (~ strcmp(IlluminationImageName3, 'Do not save')) ,
            handles.Pipeline.(IlluminationImageName3) = PostTreatImage(IlluminationField(:,:,count), Modality3);
        end
        count = count + 1;
    end
    
    if ~ strcmp(ImageName4, 'None')
        if (~ strcmp(IlluminationImageName4, 'Do not save')) ,
            handles.Pipeline.(IlluminationImageName4) = PostTreatImage(IlluminationField(:,:,count), Modality4);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(EachOrAll, 'Each') | (handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets),
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        %%% Activates the appropriate figure window.
        CPfigure(handles,'Correction',ThisModuleFigureNumber);
        imagesc(handles.Pipeline.(IlluminationImageName1));
        colorbar;
        drawnow;
    end
end
   

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function TreatedImage = PreTreatImage(Image, Modality)
switch Modality,
    case {'Fluorescent', 'Brightfield', 'Multiplicative'},
        TreatedImage = log(Image);
    case {'Phase Contrast', 'DIC', 'Linear'}
        TreatedImage = Image;
end

function TreatedImage = PostTreatImage(Image, Modality)
switch Modality,
    case {'Fluorescent', 'Brightfield', 'Multiplicative'},
        TreatedImage = exp(Image);
        TreatedImage = TreatedImage / min(TreatedImage(:));
    case {'Phase Contrast', 'DIC', 'Linear'}
        TreatedImage = Image;
end
