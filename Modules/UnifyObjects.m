function handles = UnifyObjects(handles)

% Help for UnifyObjects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
%
% Relabels objects so that objects within a specified distance of each
% other get the same label and thereby become the same object.
% Optionally, only unify two objects if the straight line connecting
% their centroids has a relatively uniform intensity in a specified
% image.
%
% *************************************************************************
%
% This module unifies objects that are within a certain distance of
% each other.  If the distance threshold is zero (the default), only
% objects that are touching will be unified.
%
% "Unifying" two objects means to change the labels of the pixels in
% one object to equal the label of the other object.  Thus, they
% become one object.  Note that the module does not connect or bridge
% the two objects by adding any new pixels, so the new, unified object
% may consist of two or more unconnected components.
% 
% As an experimental feature, it is possible to specify a grayscale
% image to help guide the decision of which objects to unify.  When
% the module considers merging two objects, it looks at the pixels
% along the line connecting their centroids in this image.  If the
% intensity of any of these pixels is below 90 percent of either
% centroid, the objects are not unified.
%
% In order to ensure that objects are labeled consecutively (which
% other modules depend on), UnifyObjects may change the label (i.e.,
% the object number) of any object.  A new "measurement" will be added
% for each input object.  This "measurement" is a number that
% indicates the relabeled object number.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003--2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision: 5039 $

%%% VARIABLES
drawnow
[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects you want to filter?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the relabeled objects?
%defaultVAR02 = UnifiedObjects
%infotypeVAR02 = objectgroup indep
RelabeledObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%textVAR03 = Distance within which objects should be unified
%defaultVAR03 = 0
DistanceThreshold = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,3})); %#ok Ignore MLint

%textVAR04 = Grayscale image the intensities of which are used to determine whether to merge (optional, see help)
%infotypeVAR04 = imagegroup
%defaultVAR04 = None
GrayscaleImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%%%VariableRevisionNumber = 1

%%%
%%% CALL THE FUNCTION THAT DOES THE ACTUAL WORK FOR EACH SET OF OBJECTS
%%%

handles = doItForObjectName(handles, 'Segmented', ObjectName, ...
			    RelabeledObjectName, DistanceThreshold, ...
			    GrayscaleImageName);
if isfield(handles.Pipeline, ['UneditedSegmented', ObjectName])
  handles = doItForObjectName(handles, 'UneditedSegmented', ObjectName, ...
			      RelabeledObjectName, DistanceThreshold, ...
			      GrayscaleImageName);
end
if isfield(handles.Pipeline, ['SmallRemovedSegmented', ObjectName])
  handles = doItForObjectName(handles, 'SmallRemovedSegmented', ObjectName, ...
			      RelabeledObjectName, DistanceThreshold, ...
			      GrayscaleImageName);
end

% Save measurements for the 'Segmented' objects only, so as to agree
% with IdentifyPrimaryAuto.
fieldName = ['Segmented', RelabeledObjectName];
labels = handles.Pipeline.(fieldName);
handles = CPsaveObjectCount(handles, RelabeledObjectName, labels);
handles = CPsaveObjectLocations(handles, RelabeledObjectName, labels);


%%%
%%% DISPLAY RESULTS
%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
  Relabeled = CPretrieveimage(handles, ['Segmented', RelabeledObjectName], ...
			      ModuleName);
  vislabel = Relabeled;
  props = regionprops(Relabeled, {'ConvexImage', 'BoundingBox'});
  for k=1:length(props)
    ci = props(k).ConvexImage;
    bb = props(k).BoundingBox;
    mask = false(size(Relabeled));
    mask(bb(2)+0.5:bb(2)+bb(4)-0.5, bb(1)+0.5:bb(1)+bb(3)-0.5) = ci;
    mask(imerode(mask, strel('disk', 1))) = 0;
    vislabel(mask) = k;
  end
  RelabeledRGB = CPlabel2rgb(handles, vislabel);
  
  CPfigure(handles,'Image',ThisModuleFigureNumber);
  CPimagesc(RelabeledRGB,handles);
  title(RelabeledObjectName);
end


%%%
%%% SUBFUNCTION THAT DOES THE ACTUAL WORK
%%%

function handles = doItForObjectName(handles, prefix, ObjectName, ...
				     RelabeledObjectName, ...
				     DistanceThreshold, GrayscaleImageName)
drawnow
[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

Orig = CPretrieveimage(handles, [prefix, ObjectName], ModuleName);

%%% IMAGE ANALYSIS
drawnow

if strcmp(GrayscaleImageName, 'None')
  dilated = imdilate(Orig > 0, strel('disk', DistanceThreshold));
  merged = bwlabel(dilated);
  merged(Orig == 0) = 0;
  Relabeled = merged;
else
  % Within each non-contiguous object of Relabeled, consider all pairs
  % of components.  Compute a profile of intensities (from
  % GrayscaleImage) along the line connecting the components' centroids.
  % If this profile looks like the components belong to separate
  % objects, break the object back up. (XXX)
  GrayscaleImage = double(CPretrieveimage(handles, GrayscaleImageName, ...
					  ModuleName, 'MustBeGray', ...
					  'CheckScale'));
  Relabeled = Orig;
  props = regionprops(Orig, {'Centroid'});
  n = length(props);
  for a=1:n
    xa = props(a).Centroid(1);
    ya = props(a).Centroid(2);
    for b=1:n
      xb = props(b).Centroid(1);
      yb = props(b).Centroid(2);
      d = sqrt((xa - xb)^2 + (ya - yb)^2);
      if d <= DistanceThreshold
	coords = brlinexya(xa, ya, xb, yb);
	xp = coords(:,1);
	yp = coords(:,2);
	profile = zeros(size(coords,1),1);
	for p = 1:length(profile)
	  profile(p) = GrayscaleImage(round(yp(p)), round(xp(p)));
	end
	if min(profile) > min(profile(1), profile(end))*0.9
	  Relabeled(Relabeled == b) = a;
	end
      end
    end
  end
  % Make sure objects are consecutively numbered, otherwise downstream
  % modules will choke.
  Relabeled = CPrelabelDensely(Relabeled);
end

%%% SAVE DATA TO HANDLES STRUCTURE
drawnow

%%% Saves the label matrix image to the handles structure.
fieldName = [prefix, RelabeledObjectName];
handles.Pipeline.(fieldName) = Relabeled;

%%%
%%% Subfunction
%%%

function [Coords]=brlinexya(Sx,Sy,Ex,Ey)
% function [Coords]=brlinexya(Sx,Sy,Ex,Ey)
% Bresenham line algorithm.
% Sx, Sy, Ex, Ey - desired endpoints
% Coords - nx2 ordered list of x,y coords.
% Author: Andrew Diamond;
%
%	if(length(M) == 0)
%		M = zeros(max([Sx,Sy]),max([Ex,Ey]));
%	end
	Dx = Ex - Sx;
	Dy = Ey - Sy;
%	Coords = [];
	CoordsX = zeros(2 .* ceil(abs(Dx)+abs(Dy)),1);
	CoordsY = zeros(2 .* ceil(abs(Dx)+abs(Dy)),1);
    iCoords=0;
	if(abs(Dy) <= abs(Dx))
		if(Ey >= Sy)
			if(Ex >= Sx)
				D = 2*Dy - Dx;
				IncH = 2*Dy;
				IncD = 2*(Dy - Dx);
				X = Sx;
				Y = Sy;
%				M(Y,X) = Value;
				% Coords = [Sx,Sy];
                iCoords = iCoords + 1;
                CoordsX(iCoords) = Sx;
                CoordsY(iCoords) = Sy;
				while(X < Ex)
					if(D <= 0)
						D = D + IncH;
						X = X + 1;
					else
						D = D + IncD;
						X = X + 1;
						Y = Y + 1;
					end
%					M(Y,X) = Value;
                    iCoords = iCoords + 1;
                    CoordsX(iCoords) = X;
                    CoordsY(iCoords) = Y;
					% Coords = [Coords; [X,Y]];
				end
			else % Ex < Sx
				D = -2*Dy - Dx;
				IncH = -2*Dy;
				IncD = 2*(-Dy - Dx);
				X = Sx;
				Y = Sy;
%				M(Y,X) = Value;
				% Coords = [Sx,Sy];
                iCoords = iCoords + 1;
                CoordsX(iCoords) = Sx;
                CoordsY(iCoords) = Sy;
				while(X > Ex)
					if(D >= 0)
						D = D + IncH;
						X = X - 1;
					else
						D = D + IncD;
						X = X - 1;
						Y = Y + 1;
					end
%					M(Y,X) = Value;
                    iCoords = iCoords + 1;
                    CoordsX(iCoords) = X;
                    CoordsY(iCoords) = Y;
%					Coords = [Coords; [X,Y]];
				end
			end
		else % Ey < Sy
			if(Ex >= Sx)
				D = 2*Dy + Dx;
				IncH = 2*Dy;
				IncD = 2*(Dy + Dx);
				X = Sx;
				Y = Sy;
%				M(Y,X) = Value;
				% Coords = [Sx,Sy];
                iCoords = iCoords + 1;
                CoordsX(iCoords) = Sx;
                CoordsY(iCoords) = Sy;
				while(X < Ex)
					if(D >= 0)
						D = D + IncH;
						X = X + 1;
					else
						D = D + IncD;
						X = X + 1;
						Y = Y - 1;
					end
%					M(Y,X) = Value;
                    iCoords = iCoords + 1;
                    CoordsX(iCoords) = X;
                    CoordsY(iCoords) = Y;
					% Coords = [Coords; [X,Y]];
				end
			else % Ex < Sx
				D = -2*Dy + Dx;
				IncH = -2*Dy;
				IncD = 2*(-Dy + Dx);
				X = Sx;
				Y = Sy;
%				M(Y,X) = Value;
				% Coords = [Sx,Sy];
                iCoords = iCoords + 1;
                CoordsX(iCoords) = Sx;
                CoordsY(iCoords) = Sy;
				while(X > Ex)
					if(D <= 0)
						D = D + IncH;
						X = X - 1;
					else
						D = D + IncD;
						X = X - 1;
						Y = Y - 1;
					end
%					M(Y,X) = Value;
                    iCoords = iCoords + 1;
                    CoordsX(iCoords) = X;
                    CoordsY(iCoords) = Y;
%					Coords = [Coords; [X,Y]];
				end
			end
		end
	else % abs(Dy) > abs(Dx) 
		Tmp = Ex;
		Ex = Ey;
		Ey = Tmp;
		Tmp = Sx;
		Sx = Sy;
		Sy = Tmp;
		Dx = Ex - Sx;
		Dy = Ey - Sy;
		if(Ey >= Sy)
			if(Ex >= Sx)
				D = 2*Dy - Dx;
				IncH = 2*Dy;
				IncD = 2*(Dy - Dx);
				X = Sx;
				Y = Sy;
%				M(X,Y) = Value;
				% Coords = [Sx,Sy];
                iCoords = iCoords + 1;
                CoordsX(iCoords) = Sy;
                CoordsY(iCoords) = Sx;
				while(X < Ex)
					if(D <= 0)
						D = D + IncH;
						X = X + 1;
					else
						D = D + IncD;
						X = X + 1;
						Y = Y + 1;
					end
%					M(X,Y) = Value;
                    iCoords = iCoords + 1;
                    CoordsX(iCoords) = Y;
                    CoordsY(iCoords) = X;
%					Coords = [Coords; [Y,X]];
				end
			else % Ex < Sx
				D = -2*Dy - Dx;
				IncH = -2*Dy;
				IncD = 2*(-Dy - Dx);
				X = Sx;
				Y = Sy;
%				M(X,Y) = Value;
				% Coords = [Sx,Sy];
                iCoords = iCoords + 1;
                CoordsX(iCoords) = Sy;
                CoordsY(iCoords) = Sx;
				while(X > Ex)
					if(D >= 0)
						D = D + IncH;
						X = X - 1;
					else
						D = D + IncD;
						X = X - 1;
						Y = Y + 1;
					end
%					M(X,Y) = Value;
                    iCoords = iCoords + 1;
                    CoordsX(iCoords) = Y;
                    CoordsY(iCoords) = X;
%					Coords = [Coords; [Y,X]];
				end
			end
		else % Ey < Sy
			if(Ex >= Sx)
				D = 2*Dy + Dx;
				IncH = 2*Dy;
				IncD = 2*(Dy + Dx);
				X = Sx;
				Y = Sy;
%				M(X,Y) = Value;
				% Coords = [Sx,Sy];
                iCoords = iCoords + 1;
                CoordsX(iCoords) = Sy;
                CoordsY(iCoords) = Sx;
				while(X < Ex)
					if(D >= 0)
						D = D + IncH;
						X = X + 1;
					else
						D = D + IncD;
						X = X + 1;
						Y = Y - 1;
					end
%					M(X,Y) = Value;
                    iCoords = iCoords + 1;
                    CoordsX(iCoords) = Y;
                    CoordsY(iCoords) = X;
%					Coords = [Coords; [Y,X]];
				end
			else % Ex < Sx
				D = -2*Dy + Dx;
				IncH = -2*Dy;
				IncD = 2*(-Dy + Dx);
				X = Sx;
				Y = Sy;
%				M(X,Y) = Value;
				% Coords = [Sx,Sy];
                iCoords = iCoords + 1;
                CoordsX(iCoords) = Sy;
                CoordsY(iCoords) = Sx;
				while(X > Ex)
					if(D <= 0)
						D = D + IncH;
						X = X - 1;
					else
						D = D + IncD;
						X = X - 1;
						Y = Y - 1;
					end
%					M(X,Y) = Value;
                    iCoords = iCoords + 1;
                    CoordsX(iCoords) = Y;
                    CoordsY(iCoords) = X;
%					Coords = [Coords; [Y,X]];
				end
			end
		end
	end
Coords = [CoordsX(1:iCoords),CoordsY(1:iCoords)];
