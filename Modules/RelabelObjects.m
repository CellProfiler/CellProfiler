function handles = RelabelObjects(handles)

% Help for RelabelObjects module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
%
% Relabels objects so that objects within a specified distance of each
% other, or objects with a straight line connecting
% their centroids that has a relatively uniform intensity, 
% get the same label and thereby become the same object.
% Optionally, if an object consists of two or more unconnected components, this
% module can relabel them so that the components become separate objects.
%
% *************************************************************************
% Relabeling objects changes the labels of the pixels in an object such
% that it either becomes equal to the label of another (unify) or changes
% the labels to distinguish two different components of an object such that
% they are two different objects (Split).
%
% If the distance threshold is zero (the default), only
% objects that are touching will be unified. Note that selecting "unify" will not connect or bridge
% the two objects by adding any new pixels. The new, unified object
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
% other modules depend on), RelabelObjects may change the label (i.e.,
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
% $Revision$

%%% VARIABLES

% New Settings for PyCP:
% This module combined 'Split' and 'Unify' since both relabeled objects. We
% could think about including other schemes for relabeling objects in this
% module; also, given that in PyCP objects are allowed to touch I think
% this module might need to be looked at more carefully. Anne 3-26-09: Also
% think about whether the Relate module is relevant here. It relabels
% objects based on their parentage although the primary goal is to relabel
% and combine the measurements rather than produce a relabeled image.
%
% A major point that could use some discussion is that most users have no
% idea what a label even is. The object label concept is not something that
% is heavily emphasized. So, we should really reconsider how to make this
% understandable; e.g., should this module be something like "Redefine
% Relationships" and include the Unify/Split/Relate functionalities?
%
% Vars 4&5 should only appear when the user has selected Unify, of course.
% Var 4 can be reworded: "What is the distance within which objects should
% be unified?"
% Var 5 should then be: "If you would also like to use intensities to
% determine which objects to unify, select the image here:" (and 'Do not use' is
% the default option).
% 
% We should be careful about using the words "unify" and "merge". I think
% we should add a new Merge option that behaves like Unify except that it
% physically fills in the space between the two objects so it becomes a
% continguous object.

drawnow
[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects you want to relabel?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the relabeled objects?
%defaultVAR02 = RelabeledObjects
%infotypeVAR02 = objectgroup indep
RelabeledObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Would you like to Unify or Split these objects?
%choiceVAR03 = Unify
%choiceVAR03 = Split
RelabelOption = char(handles.Settings.VariableValues{CurrentModuleNum,3}); %#ok Ignore MLint
%inputtypeVAR03 = popupmenu

%textVAR04 = For 'Unify' option, what is the distance within which objects should be unified?
%defaultVAR04 = 0
DistanceThreshold = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,4})); %#ok Ignore MLint

%textVAR05 = For 'Unify' option, Grayscale image the intensities of which are used to determine whether to merge (optional, see help). Select 'Do not use' if you do not want to use this option.
%infotypeVAR05 = imagegroup
%defaultVAR05 = Do not use
GrayscaleImageName = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%%%VariableRevisionNumber = 1

%%%
%%% CALL THE FUNCTION THAT DOES THE ACTUAL WORK FOR EACH SET OF OBJECTS
%%%

handles = doItForObjectName(handles, 'Segmented', ObjectName, ...
			    RelabeledObjectName, RelabelOption, DistanceThreshold, ...
			    GrayscaleImageName);
if CPisimageinpipeline(handles, ['SmallRemovedSegmented', ObjectName])
         handles = doItForObjectName(handles, 'SmallRemovedSegmented', ObjectName, ...
			      RelabeledObjectName, RelabelOption, DistanceThreshold, ...
    			  GrayscaleImageName);
end

if CPisimageinpipeline(handles, ['UneditedSegmented', ObjectName])
        handles = doItForObjectName(handles, 'UneditedSegmented', ObjectName, ...
			      RelabeledObjectName, RelabelOption, DistanceThreshold, ...
			      GrayscaleImageName);
end


% Save measurements for the 'Segmented' objects only, so as to agree
% with IdentifyPrimaryAuto.
fieldName = ['Segmented', RelabeledObjectName];
labels = CPretrieveimage(handles,fieldName,ModuleName);
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
  
    %% This commented out code effectively adds a convex hull to each object,
    %% but is unnecessary and adds confusing new information to the displayed output
    %% since CPlabel2rgb color codes each object (Dlogan 2009-05-29).
%   props = regionprops(Relabeled, {'ConvexImage', 'BoundingBox'});
%   for k=1:length(props)
%     ci = props(k).ConvexImage;
%     bb = props(k).BoundingBox;
%     mask = false(size(Relabeled));
%     mask(bb(2)+0.5:bb(2)+bb(4)-0.5, bb(1)+0.5:bb(1)+bb(3)-0.5) = ci;
%     mask(imerode(mask, strel('disk', 1))) = 0;
%     vislabel(mask) = k;
%   end

  RelabeledRGB = CPlabel2rgb(handles, vislabel);
  
  CPfigure(handles,'Image',ThisModuleFigureNumber);
  [hImage,hAx]=CPimagesc(RelabeledRGB,handles,ThisModuleFigureNumber);
  title(hAx,RelabeledObjectName);
end


%%%
%%% SUBFUNCTION THAT DOES THE ACTUAL WORK
%%%

function handles = doItForObjectName(handles, prefix, ObjectName, ...
				     RelabeledObjectName, RelabelOption,...
				     DistanceThreshold, GrayscaleImageName)
drawnow
[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

Orig = CPretrieveimage(handles, [prefix, ObjectName], ModuleName);

if strcmp(RelabelOption, 'Split')

    Relabeled = bwlabel(Orig > 0);

    % Compute the mapping from the new labels to the old labels.  This
    % mapping is stored as a measurement later so we know which objects
    % have been split.
    Mapping = zeros(max(Relabeled(:)), 1);
    props = regionprops(Relabeled, {'PixelIdxList'});
    for i=1:max(Relabeled(:))
    Mapping(i,1) = Orig(props(i).PixelIdxList(1));
    end
end

if strcmp(RelabelOption, 'Unify')

    if strcmp(GrayscaleImageName, 'Do not use')
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
    end

    
  % Make sure objects are consecutively numbered, otherwise downstream
  % modules will choke.
  Relabeled = CPrelabelDensely(Relabeled);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE DATA TO HANDLES STRUCTURE %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the label matrix image to the handles structure.
fieldName = [prefix, RelabeledObjectName];
handles = CPaddimages(handles,fieldName,Relabeled);

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
