function handles = IdentifyPrimLog(handles)

% Help for the Identify Primary LoG module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
%
% Identifies the centers of blob-like primary objects.  The result
% consists of only a single pixel per object, located near the center
% of the object.
%
% *************************************************************************
%
% This module identifies the centers of blob-like primary objects
% (e.g. nuclei) in grayscale images that show bright objects on a dark
% background.  When the objects of interest are fairly round and of
% even size, this module may be more sensitive than the methods in
% IdentifyPrimAutomatic and therefore detect objects that would
% otherwise be lost.
% 
% The result consists of only a single pixel per object, located near
% the center of the object; the IdentifySecondary module can be used
% to fill out the object based on this center point.
%
% The radius parameter should be set to the approximate radius of the
% objects of interest.  The algorithm is not very sensitive to this
% parameter.
%
% The threshold parameter tells the algorithm how inclusive to be when
% looking for objects.  Internally, each potential object is assigned
% a score that depends on both how bright the object is and how
% blob-like its shape is.  Only objects that score above the threshold
% are returned.  The threshold must be determined experimentally.  If
% it is too high, objects will be lost; if it is too low, spurious
% objects will be found.
%
% The module works by convolving the image with the Laplacian of
% Gaussian (LoG) kernel.  This is equivalent to convolving with the
% Gaussian kernel and then with the Laplace operator.  The regional
% maxima in the filter response that exceed the specificed threshold
% are identified as objects.  The radius parameter specifies the width
% of the kernel.
%
% Ultimately, this module will become an option in
% IdentifyPrimAutomatic, so that its options for maxima suppression
% and finding edges between clumps can be used.
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Nuclei
%infotypeVAR02 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Typical diameter of objects, in pixel units:
%defaultVAR03 = 10
Radius = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Score threshold for match
%defaultVAR04 = 1e-3
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY ERROR CHECKING & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

OrigImage = double(CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale'));
Radius = str2double(Radius);
Threshold = str2double(Threshold);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

im = double(OrigImage) - double(min(OrigImage(:)));
im = im / max(im(:));

%% Set regions outside of CropMasks equal to 0
fieldname = ['CropMask', ImageName];
if isfield(handles.Pipeline,fieldname)
    %%% Retrieves previously selected cropping mask from handles
    %%% structure.
    try 
        im(~(handles.Pipeline.(fieldname))) = 0;
    catch
        error('The image in which you want to identify objects has been cropped, but there was a problem recognizing the cropping pattern.');
    end
end

ac = lapofgau(1 - im, Radius);
ac(ac < Threshold) = Threshold;
ac = ac - Threshold;
indices = find(imregionalmax(ac));
maxima = sortrows([indices ac(indices)], -2);

bw = false(size(im));
bw(maxima(:,1)) = true;
FinalLabelMatrixImage = bwlabel(bw);

% The dilated mask is used only for visualization.
dilated = imdilate(bw, strel('disk', 2));
vislabel = bwlabel(dilated);
r = im;
g = im;
b = im;
r(dilated) = 1;
g(dilated) = 0;
b(dilated) = 0;
visRGB = cat(3, r, g, b);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
  h_fig = CPfigure(handles,'Image',ThisModuleFigureNumber);
  CPimagesc(visRGB, handles);
  title([ObjectName, ' cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
  
  uicontrol(h_fig,'units','normalized','position',[.01 .5 .06 .04],'string','off',...
      'UserData',{OrigImage visRGB},'backgroundcolor',[.7 .7 .9],...
      'Callback',@CP_OrigNewImage_Callback);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

prefixes = {'Segmented', 'SmallRemovedSegmented'};
for i=1:length(prefixes)
  prefix = prefixes{i};
  fieldname = [prefix, ObjectName];
  handles.Pipeline.(fieldname) = FinalLabelMatrixImage;
% 
%   handles = CPsaveObjectCount(handles, fieldname, FinalLabelMatrixImage);
%   handles = CPsaveObjectLocations(handles, fieldname, FinalLabelMatrixImage);
end
handles = CPsaveObjectCount(handles, ObjectName, FinalLabelMatrixImage);
handles = CPsaveObjectLocations(handles, ObjectName, FinalLabelMatrixImage);

function f = lapofgau(im, s)
% im: image matrix (2 dimensional)
% s: filter width
% f: filter output.
% Author: Baris Sumengen - sumengen@ece.ucsb.edu

sigma = (s-1)/3;
op = fspecial('log',s,sigma); 
op = op - sum(op(:))/numel(op); % make the op to sum to zero
f = filter2(op,im);
