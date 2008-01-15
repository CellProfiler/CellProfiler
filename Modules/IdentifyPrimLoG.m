function handles = IdentifyPrimLog(handles)

% Help for the Identify Primary LoG module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Identifies objects given only an image as input.
% *************************************************************************
%
% This module identifies primary objects (e.g. nuclei) in grayscale images
% that show bright objects on a dark background.
%
% $Revision: 5009 $

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

ac = lapofgau(1 - im, Radius);
ac(find(ac < Threshold)) = Threshold;
ac = ac - Threshold;
ac2 = ac;
indices = find(imregionalmax(ac2));
maxima = sortrows([indices ac2(indices)], -2);

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
  CPfigure(handles,'Image',ThisModuleFigureNumber);
  CPimagesc(visRGB, handles);
  title([ObjectName, ' cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
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

  handles = CPsaveObjectCount(handles, fieldname, FinalLabelMatrixImage);
  handles = CPsaveObjectLocations(handles, fieldname, FinalLabelMatrixImage);
end

function f = lapofgau(im, s);
% im: image matrix (2 dimensional)
% s: filter width
% f: filter output.
% Author: Baris Sumengen - sumengen@ece.ucsb.edu

sigma = (s-1)/3;
op = fspecial('log',s,sigma); 
op = op - sum(op(:))/prod(size(op)); % make the op to sum to zero
f = filter2(op,im);
