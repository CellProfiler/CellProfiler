function outstats = regionprops(varargin)
%REGIONPROPS Measure properties of image regions.
%   STATS = REGIONPROPS(L,PROPERTIES) measures a set of properties for each
%   labeled region in the label matrix L. Positive integer elements of L
%   correspond to different regions. For example, the set of elements of L
%   equal to 1 corresponds to region 1; the set of elements of L equal to 2
%   corresponds to region 2; and so on. STATS is a structure array of length
%   max(L(:)). The fields of the structure array denote different properties
%   for each region, as specified by PROPERTIES.
%
%   PROPERTIES can be a comma-separated list of strings, a cell array
%   containing strings, the string 'all', or the string 'basic'. The set of
%   valid measurement strings includes:
%
%     'Area'              'ConvexHull'    'EulerNumber'
%     'Centroid'          'ConvexImage'   'Extrema'       
%     'BoundingBox'       'ConvexArea'    'EquivDiameter' 
%     'SubarrayIdx'       'Image'         'Solidity'      
%     'MajorAxisLength'   'PixelList'     'Extent'        
%     'MinorAxisLength'   'PixelIdxList'  'FilledImage'  
%     'Orientation'                       'FilledArea'                   
%     'Eccentricity'                      'Perimeter'  
%                                                         
%   Property strings are case insensitive and can be abbreviated.
%
%   If PROPERTIES is the string 'all', then all of the above measurements
%   are computed. If PROPERTIES is not specified or if it is the string
%   'basic', then these measurements are computed: 'Area', 'Centroid', and
%   'BoundingBox'.
%
%   Perimeter should be used on a label matrix with contiguous regions, such
%   as L = bwlabel(BW). Otherwise, 'perimeter' gives unexpected results on
%   discontiguous regions.
%  
%   Note - REGIONPROPS and binary images
%   ------------------------------------
%   REGIONPROPS does not accept a binary image as its first input.  There
%   are two common ways to convert a binary image to a label matrix:
%
%       1.  L = bwlabel(BW);
%
%       2.  L = double(BW);
%
%   Suppose that BW were a logical matrix containing these values:
%
%       1 1 0 0 0 0
%       1 1 0 0 0 0
%       0 0 0 0 0 0
%       0 0 0 0 1 1
%       0 0 0 0 1 1
%
%   The first method of forming a label matrix, L = bwlabel(BW), results
%   in a label matrix containing two contiguous regions labeled by the
%   integer values 1 and 2.  The second method of forming a label matrix,
%   L = double(BW), results in a label matrix containing one
%   discontiguous region labeled by the integer value 1.  Since each
%   result is legitimately desirable in certain situations, REGIONPROPS
%   does not accept binary images and convert them using either method.
%   You should convert a binary image to a label matrix using one of
%   these methods (or another method if appropriate) before calling
%   REGIONPROPS.
%
%   Example
%   -------
%   Label the connected pixel components in the text.png image, compute
%   their centroids, and superimpose the centroid locations on the
%   image.
%
%       bw = imread('text.png');
%       L = bwlabel(bw);
%       s  = regionprops(L, 'centroid');
%       centroids = cat(1, s.Centroid);
%       imshow(bw)
%       hold on
%       plot(centroids(:,1), centroids(:,2), 'b*')
%       hold off
%
%   Class Support
%   -------------
%   The input label matrix L can have any numeric class.
%
%   See also BWLABEL, BWLABELN, ISMEMBER, WATERSHED.

%   Copyright 1993-2004 The MathWorks, Inc.
%   $Revision.4.2.3 $  $Date: 2004/08/10 01:46:30 $

officialStats = {'Area'
                 'Centroid'
                 'BoundingBox'
                 'SubarrayIdx'
                 'MajorAxisLength'
                 'MinorAxisLength'
                 'Eccentricity'
                 'Orientation'
                 'ConvexHull'
                 'ConvexImage'
                 'ConvexArea'
                 'Image'
                 'FilledImage'
                 'FilledArea'
                 'EulerNumber'
                 'Extrema'
                 'EquivDiameter'
                 'Solidity'
                 'Extent'
                 'PixelIdxList'
                 'PixelList'
                 'Perimeter'};

tempStats = {'PerimeterCornerPixelList'};

allStats = [officialStats; tempStats];

[L, requestedStats] = ParseInputs(officialStats, varargin{:});

if ndims(L) > 2
  % Remove stats that aren't supported for N-D input and issue
  % warning messages as appropriate.
  requestedStats = PreprocessRequestedStats(requestedStats);
end

if isempty(requestedStats)
  eid = sprintf('Images:%s:noPropertiesWereSelected',mfilename);
  msg = 'No input properties';
  error(eid,'%s',msg);
end

if (isempty(L))
  numObjs = 0;
else
  numObjs = round(double(max(L(:))));
end

% Initialize the stats structure array.
numStats = length(allStats);
empties = cell(numStats, numObjs);
stats = cell2struct(empties, allStats, 1);

% Initialize the computedStats structure array.
zz = cell(numStats, 1);
for k = 1:numStats
  zz{k} = 0;
end
computedStats = cell2struct(zz, allStats, 1);

% Calculate PixelIdxList
[stats, computedStats] = ComputePixelIdxList(L, stats, computedStats, ...
                                             numObjs);

% Compute statistics.
for k = 1:length(requestedStats)
    switch requestedStats{k}
        
    case 'Area'
        [stats, computedStats] = ComputeArea(L, stats, computedStats);
        
    case 'FilledImage'
        [stats, computedStats] = ComputeFilledImage(L,stats,computedStats);
        
    case 'FilledArea'
        [stats, computedStats] = ComputeFilledArea(L,stats,computedStats);
        
    case 'ConvexArea'
        [stats, computedStats] = ComputeConvexArea(L, stats, computedStats);
        
    case 'Centroid'
        [stats, computedStats] = ComputeCentroid(L, stats, computedStats);
        
    case 'EulerNumber'
        [stats, computedStats] = ComputeEulerNumber(L,stats,computedStats);
        
    case 'EquivDiameter'
        [stats, computedStats] = ComputeEquivDiameter(L, stats, computedStats);
        
    case 'Extrema'
        [stats, computedStats] = ComputeExtrema(L, stats, computedStats);
        
    case 'BoundingBox'
        [stats, computedStats] = ComputeBoundingBox(L, stats, computedStats);
        
    case 'SubarrayIdx'
        [stats, computedStats] = ComputeSubarrayIdx(L, stats, computedStats);
        
    case {'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Eccentricity'}
        [stats, computedStats] = ComputeEllipseParams(L, stats, ...
                                                      computedStats);
        
    case 'Solidity'
        [stats, computedStats] = ComputeSolidity(L, stats, computedStats);
        
    case 'Extent'
        [stats, computedStats] = ComputeExtent(L, stats, computedStats);
        
    case 'ConvexImage'
        [stats, computedStats] = ComputeConvexImage(L, stats, computedStats);
        
    case 'ConvexHull'
        [stats, computedStats] = ComputeConvexHull(L, stats, computedStats);
        
    case 'Image'
        [stats, computedStats] = ComputeImage(L, stats, computedStats);
        
    case 'PixelList'
        [stats, computedStats] = ComputePixelList(L, stats, computedStats);
    
    case 'Perimeter'
        [stats, computedStats] = ComputePerimeter(L, stats, computedStats);
    end
end

% Initialize the output stats structure array.
numStats = length(requestedStats);
empties = cell(numStats, numObjs);
outstats = cell2struct(empties, requestedStats, 1);

fnames = fieldnames(stats);
deleteStats = fnames(~ismember(fnames,requestedStats));
outstats = rmfield(stats,deleteStats);

%%%
%%% ComputePixelIdxList
%%%
function [stats, computedStats] = ComputePixelIdxList(L, stats,computedStats,numobj)
%   A P-by-1 matrix, where P is the number of pixels belonging to
%   the region.  Each element contains the linear index of the
%   corresponding pixel.

computedStats.PixelIdxList = 1;
  
if ~isempty(L) && numobj ~= 0
  idxList = regionpropsmex(L, numobj);
  [stats.PixelIdxList] = deal(idxList{:});
end

%%%
%%% ComputeArea
%%%
function [stats, computedStats] = ComputeArea(L, stats, computedStats)
%   The area is defined to be the number of pixels belonging to
%   the region.

  if ~computedStats.Area
    computedStats.Area = 1;

    for k = 1:length(stats)
      stats(k).Area = size(stats(k).PixelIdxList, 1);
    end
  end

%%%
%%% ComputeEquivDiameter
%%%
function [stats, computedStats] = ComputeEquivDiameter(L, stats, computedStats)
%   Computes the diameter of the circle that has the same area as
%   the region.
%   Ref: Russ, The Image Processing Handbook, 2nd ed, 1994, page
%   511.

  if ~computedStats.EquivDiameter
    computedStats.EquivDiameter = 1;
    
    if ndims(L) > 2
      NoNDSupport('EquivDiameter');
      return
    end
    
    [stats, computedStats] = ComputeArea(L, stats, computedStats);

    factor = 2/sqrt(pi);
    for k = 1:length(stats)
      stats(k).EquivDiameter = factor * sqrt(stats(k).Area);
    end
  end

%%%
%%% ComputeFilledImage
%%%
function [stats, computedStats] = ComputeFilledImage(L,stats,computedStats)
%   Uses imfill to fill holes in the region.

  if ~computedStats.FilledImage
    computedStats.FilledImage = 1;
    
    [stats, computedStats] = ComputeImage(L, stats, computedStats);
    
    conn = conndef(ndims(L),'minimal');
    
    for k = 1:length(stats)
      stats(k).FilledImage = imfill(stats(k).Image,conn,'holes');
    end
  end

%%%
%%% ComputeConvexArea
%%%
function [stats, computedStats] = ComputeConvexArea(L, stats, computedStats)
%   Computes the number of "on" pixels in ConvexImage.

  if ~computedStats.ConvexArea
    computedStats.ConvexArea = 1;
    
    if ndims(L) > 2
      NoNDSupport('ConvexArea');
      return
    end
    
    [stats, computedStats] = ComputeConvexImage(L, stats, computedStats);
    
    for k = 1:length(stats)
      stats(k).ConvexArea = sum(stats(k).ConvexImage(:));
    end
  end

%%%
%%% ComputeFilledArea
%%%
function [stats, computedStats] = ComputeFilledArea(L,stats,computedStats)
%   Computes the number of "on" pixels in FilledImage.

  if ~computedStats.FilledArea
    computedStats.FilledArea = 1;
    
    [stats, computedStats] = ComputeFilledImage(L,stats,computedStats);

    for k = 1:length(stats)
      stats(k).FilledArea = sum(stats(k).FilledImage(:));
    end
  end

%%%
%%% ComputeConvexImage
%%%
function [stats, computedStats] = ComputeConvexImage(L, stats, computedStats)
%   Uses ROIPOLY to fill in the convex hull.

  if ~computedStats.ConvexImage
    computedStats.ConvexImage = 1;
    
    if ndims(L) > 2
      NoNDSupport('ConvexImage');
      return
    end
    
    [stats, computedStats] = ComputeConvexHull(L, stats, computedStats);
    [stats, computedStats] = ComputeBoundingBox(L, stats, computedStats);
    
    for k = 1:length(stats)
      M = stats(k).BoundingBox(4);
      N = stats(k).BoundingBox(3);
      hull = stats(k).ConvexHull;
      if (isempty(hull))
        stats(k).ConvexImage = false(M,N);
      else
        firstRow = stats(k).BoundingBox(2) + 0.5;
        firstCol = stats(k).BoundingBox(1) + 0.5;
        r = hull(:,2) - firstRow + 1;
        c = hull(:,1) - firstCol + 1;
        stats(k).ConvexImage = roipoly(M, N, c, r);
      end
    end
  end

%%%
%%% ComputeCentroid
%%%
function [stats, computedStats] = ComputeCentroid(L, stats, computedStats)
%   [mean(r) mean(c)]

  if ~computedStats.Centroid
    computedStats.Centroid = 1;
    
    [stats, computedStats] = ComputePixelList(L, stats, computedStats);

    
    % Save the warning state and disable warnings to prevent divide-by-zero
    % warnings.
    warning off MATLAB:divideByZero;
    
    for k = 1:length(stats)
      stats(k).Centroid = mean(stats(k).PixelList,1);
    end
    
    % Restore the warning state.
    warning on MATLAB:divideByZero;
  end

%%%
%%% ComputeEulerNumber
%%%
function [stats, computedStats] = ComputeEulerNumber(L,stats,computedStats)
%   Calls BWEULER on 'Image' using 8-connectivity

  if ~computedStats.EulerNumber
    computedStats.EulerNumber = 1;
    
    if ndims(L) > 2
      NoNDSupport('EulerNumber');
      return
    end
    
    [stats, computedStats] = ComputeImage(L, stats, computedStats);
    
    for k = 1:length(stats)
      stats(k).EulerNumber = bweuler(stats(k).Image,8);
    end
  end

%%%
%%% ComputeExtrema
%%%
function [stats, computedStats] = ComputeExtrema(L, stats, computedStats)
%   A 8-by-2 array; each row contains the x and y spatial
%   coordinates for these extrema:  leftmost-top, rightmost-top,
%   topmost-right, bottommost-right, rightmost-bottom, leftmost-bottom,
%   bottommost-left, topmost-left. 
%   reference: Haralick and Shapiro, Computer and Robot Vision
%   vol I, Addison-Wesley 1992, pp. 62-64.

  if ~computedStats.Extrema
    computedStats.Extrema = 1;
    
    if ndims(L) > 2
      NoNDSupport('Extrema');
      return
    end
    
    [stats, computedStats] = ComputePixelList(L, stats, computedStats);
    
    for k = 1:length(stats)
      pixelList = stats(k).PixelList;
      if (isempty(pixelList))
        stats(k).Extrema = zeros(8,2) + 0.5;
      else
        r = pixelList(:,2);
        c = pixelList(:,1);
        
        minR = min(r);
        maxR = max(r);
        minC = min(c);
        maxC = max(c);
        
        minRSet = find(r==minR);
        maxRSet = find(r==maxR);
        minCSet = find(c==minC);
        maxCSet = find(c==maxC);

        % Points 1 and 2 are on the top row.
        r1 = minR;
        r2 = minR;
        % Find the minimum and maximum column coordinates for
        % top-row pixels.
        tmp = c(minRSet);
        c1 = min(tmp);
        c2 = max(tmp);
        
        % Points 3 and 4 are on the right column.
        % Find the minimum and maximum row coordinates for
        % right-column pixels.
        tmp = r(maxCSet);
        r3 = min(tmp);
        r4 = max(tmp);
        c3 = maxC;
        c4 = maxC;

        % Points 5 and 6 are on the bottom row.
        r5 = maxR;
        r6 = maxR;
        % Find the minimum and maximum column coordinates for
        % bottom-row pixels.
        tmp = c(maxRSet);
        c5 = max(tmp);
        c6 = min(tmp);
        
        % Points 7 and 8 are on the left column.
        % Find the minimum and maximum row coordinates for
        % left-column pixels.
        tmp = r(minCSet);
        r7 = max(tmp);
        r8 = min(tmp);
        c7 = minC;
        c8 = minC;
        
        stats(k).Extrema = [c1-0.5 r1-0.5
                            c2+0.5 r2-0.5
                            c3+0.5 r3-0.5
                            c4+0.5 r4+0.5
                            c5+0.5 r5+0.5
                            c6-0.5 r6+0.5
                            c7-0.5 r7+0.5
                            c8-0.5 r8-0.5];
      end
    end
    
  end
  
%%%
%%% ComputeBoundingBox
%%%
function [stats, computedStats] = ComputeBoundingBox(L, stats, computedStats)
%   [minC minR width height]; minC and minR end in .5.

  if ~computedStats.BoundingBox
    computedStats.BoundingBox = 1;
    
    [stats, computedStats] = ComputePixelList(L, stats, computedStats);
      
    num_dims = ndims(L);
    
    for k = 1:length(stats)
      list = stats(k).PixelList;
      if (isempty(list))
        stats(k).BoundingBox = [0.5*ones(1,num_dims) zeros(1,num_dims)];
      else
        min_corner = min(list,[],1) - 0.5;
        max_corner = max(list,[],1) + 0.5;
        stats(k).BoundingBox = [min_corner (max_corner - min_corner)];
      end
    end
  end

%%%
%%% ComputeSubarrayIdx
%%%
function [stats, computedStats] = ComputeSubarrayIdx(L, stats, computedStats)
%   Find a cell-array containing indices so that L(idx{:}) extracts the
%   elements of L inside the bounding box.

  if ~computedStats.SubarrayIdx
    computedStats.SubarrayIdx = 1;
    
    [stats, computedStats] = ComputeBoundingBox(L, stats, computedStats);
    num_dims = ndims(L);
    idx = cell(1,num_dims);
    for k = 1:length(stats)
      boundingBox = stats(k).BoundingBox;
      left = boundingBox(1:(end/2));
      right = boundingBox((1+end/2):end);
      left = left(1,[2 1 3:end]);
      right = right(1,[2 1 3:end]);
      for p = 1:num_dims
        first = left(p) + 0.5;
        last = first + right(p) - 1;
        idx{p} = first:last;
      end
      stats(k).SubarrayIdx = idx;
    end
  end

%%%
%%% ComputeEllipseParams
%%%
function [stats, computedStats] = ComputeEllipseParams(L, stats, ...
                                                    computedStats)  
%   Find the ellipse that has the same normalized second central moments as the
%   region.  Compute the axes lengths, orientation, and eccentricity of the
%   ellipse.  Ref: Haralick and Shapiro, Computer and Robot Vision vol I,
%   Addison-Wesley 1992, Appendix A.


  if ~(computedStats.MajorAxisLength && computedStats.MinorAxisLength && ...
       computedStats.Orientation && computedStats.Eccentricity)
    computedStats.MajorAxisLength = 1;
    computedStats.MinorAxisLength = 1;
    computedStats.Eccentricity = 1;
    computedStats.Orientation = 1;
    
    if ndims(L) > 2
      NoNDSupport({'MajorAxisLength', 'MinorAxisLength', ...
                   'Eccentricity', 'Orientation'});
      return
    end
    
    [stats, computedStats] = ComputePixelList(L, stats, computedStats);
    [stats, computedStats] = ComputeCentroid(L, stats, computedStats);

    % Disable divide-by-zero warning
    warning off MATLAB:divideByZero;
    
    for k = 1:length(stats)
      list = stats(k).PixelList;
      if (isempty(list))
        stats(k).MajorAxisLength = 0;
        stats(k).MinorAxisLength = 0;
        stats(k).Eccentricity = 0;
        stats(k).Orientation = 0;
        
      else
        % Assign X and Y variables so that we're measuring orientation
        % counterclockwise from the horizontal axis.
        
        xbar = stats(k).Centroid(1);
        ybar = stats(k).Centroid(2);
        
        x = list(:,1) - xbar;
        y = -(list(:,2) - ybar); % This is negative for the 
                                 % orientation calculation (measured in the
                                 % counter-clockwise direction).
        
        N = length(x);
        
        % Calculate normalized second central moments for the region. 1/12 is
        % the normalized second central moment of a pixel with unit length.
        uxx = sum(x.^2)/N + 1/12; 
        uyy = sum(y.^2)/N + 1/12;
        uxy = sum(x.*y)/N;
        
        % Calculate major axis length, minor axis length, and eccentricity.
        common = sqrt((uxx - uyy)^2 + 4*uxy^2);
        stats(k).MajorAxisLength = 2*sqrt(2)*sqrt(uxx + uyy + common);
        stats(k).MinorAxisLength = 2*sqrt(2)*sqrt(uxx + uyy - common);
        stats(k).Eccentricity = 2*sqrt((stats(k).MajorAxisLength/2)^2 - ...
                                       (stats(k).MinorAxisLength/2)^2) / ...
            stats(k).MajorAxisLength;
        
        % Calculate orientation.
        if (uyy > uxx)
          num = uyy - uxx + sqrt((uyy - uxx)^2 + 4*uxy^2);
          den = 2*uxy;
        else
          num = 2*uxy;
          den = uxx - uyy + sqrt((uxx - uyy)^2 + 4*uxy^2);
        end
        if (num == 0) && (den == 0)
          stats(k).Orientation = 0;
        else
          stats(k).Orientation = (180/pi) * atan(num/den);
        end
      end
    end
    
    % Restore warning state.
    warning on MATLAB:divideByZero;
  end
  
%%%
%%% ComputeSolidity
%%%
function [stats, computedStats] = ComputeSolidity(L, stats, computedStats)
%   Area / ConvexArea

  if ~computedStats.Solidity
    computedStats.Solidity = 1;
    
    if ndims(L) > 2
      NoNDSupport('Solidity');
      return
    end
    
    [stats, computedStats] = ComputeArea(L, stats, computedStats);
    [stats, computedStats] = ComputeConvexArea(L, stats, computedStats);
    
    for k = 1:length(stats)
      if (stats(k).ConvexArea == 0)
        stats(k).Solidity = NaN;
      else
        stats(k).Solidity = stats(k).Area / stats(k).ConvexArea;
      end
    end
  end

%%%
%%% ComputeExtent
%%%
function [stats, computedStats] = ComputeExtent(L, stats, computedStats)
%   Area / (BoundingBox(3) * BoundingBox(4))

  if ~computedStats.Extent
    computedStats.Extent = 1;
    
    if ndims(L) > 2
      NoNDSupport('Extent');
      return
    end
    
    [stats, computedStats] = ComputeArea(L, stats, computedStats);
    [stats, computedStats] = ComputeBoundingBox(L, stats, computedStats);
    
    for k = 1:length(stats)
      if (stats(k).Area == 0)
        stats(k).Extent = NaN;
      else
        stats(k).Extent = stats(k).Area / prod(stats(k).BoundingBox(3:4));
      end
    end
  end

%%%
%%% ComputeImage
%%%
function [stats, computedStats] = ComputeImage(L, stats, computedStats)
%   Binary image containing "on" pixels corresponding to pixels
%   belonging to the region.  The size of the image corresponds
%   to the size of the bounding box for each region.

  if ~computedStats.Image
    computedStats.Image = 1;

    [stats, computedStats] = ComputeSubarrayIdx(L, stats, computedStats);

    for k = 1:length(stats)
      subarray = L(stats(k).SubarrayIdx{:});
      if ~isempty(subarray)
        stats(k).Image = (subarray == k);
      else
        stats(k).Image = logical(subarray);
      end
    end
  end


%%%
%%% ComputePixelList
%%%
function [stats, computedStats] = ComputePixelList(L, stats, computedStats)
%   A P-by-2 matrix, where P is the number of pixels belonging to
%   the region.  Each row contains the row and column
%   coordinates of a pixel.

  if ~computedStats.PixelList
    computedStats.PixelList = 1;
    
    % Convert the linear indices to subscripts and store
    % the results in the pixel list.  Reverse the order of the first
    % two subscripts to form x-y order.
    In = cell(1,ndims(L));
    for k = 1:length(stats)
      if ~isempty(stats(k).PixelIdxList)
        [In{:}] = ind2sub(size(L), stats(k).PixelIdxList);
        stats(k).PixelList = [In{:}];
        stats(k).PixelList = stats(k).PixelList(:,[2 1 3:end]);
      else
        stats(k).PixelList = zeros(0,ndims(L));
      end
    end
  end

%%%
%%% ComputePerimeterCornerPixelList
%%%
function [stats, computedStats] = ComputePerimeterCornerPixelList(L, ...
                                                    stats, computedStats)
  %   Find the pixels on the perimeter of the region; make a list
  %   of the coordinates of their corners; sort and remove
  %   duplicates.

  if ~computedStats.PerimeterCornerPixelList
    computedStats.PerimeterCornerPixelList = 1;
    
    if ndims(L) > 2
      NoNDSupport('PerimeterCornerPixelList');
      return
    end
    
    [stats, computedStats] = ComputeImage(L, stats, computedStats);
    [stats, computedStats] = ComputeBoundingBox(L, stats, computedStats);

    for k = 1:length(stats)
      perimImage = bwmorph(stats(k).Image, 'perim8');
      firstRow = stats(k).BoundingBox(2) + 0.5;
      firstCol = stats(k).BoundingBox(1) + 0.5;
      [r,c] = find(perimImage);
      % Force rectangular empties.
      r = r(:) + firstRow - 1;
      c = c(:) + firstCol - 1;
      rr = [r-.5 ; r    ; r+.5 ; r   ];
      cc = [c    ; c+.5 ; c    ; c-.5];
      stats(k).PerimeterCornerPixelList = [cc rr];
    end
    
  end

%%%
%%% ComputeConvexHull
%%%
function [stats, computedStats] = ComputeConvexHull(L, stats, computedStats)
%   A P-by-2 array representing the convex hull of the region.
%   The first column contains row coordinates; the second column
%   contains column coordinates.  The resulting polygon goes
%   through pixel corners, not pixel centers.

  if ~computedStats.ConvexHull
    computedStats.ConvexHull = 1;
    
    if ndims(L) > 2
      NoNDSupport('ConvexHull');
      return
    end
    
    [stats, computedStats] = ComputePerimeterCornerPixelList(L, stats, ...
                                                      computedStats);
    [stats, computedStats] = ComputeBoundingBox(L, stats, computedStats);

    for k = 1:length(stats)
      list = stats(k).PerimeterCornerPixelList;
      if (isempty(list))
        stats(k).ConvexHull = zeros(0,2);
      else
        rr = list(:,2);
        cc = list(:,1);
        hullIdx = convhull(rr, cc);
        stats(k).ConvexHull = list(hullIdx,:);
      end
    end
  end

%%%
%%% ComputePerimeter
%%%
function [stats, computedStats] = ComputePerimeter(L, stats, computedStats)

  if ~computedStats.Perimeter
    computedStats.Perimeter = 1;
    
    if ndims(L) > 2
      NoNDSupport('ComputePerimeter');
      return
    end
    
    B = regionboundariesmex(double(L),8);

    for i = 1:length(B)
      boundary = B{i};
      delta = diff(boundary).^2;
      stats(i).Perimeter = sum(sqrt(sum(delta,2)));
    end
  end

%%%
%%% ParseInputs
%%%
function [L,reqStats] = ParseInputs(officialStats, varargin)

  L = [];
  reqStats = [];

  if (length(varargin) < 1)
    eid = sprintf('Images:%s:tooFewInputs',mfilename);
    msg = 'Too few input arguments.';
    error(eid,'%s',msg);
  end

  L = varargin{1};

  if islogical(L)
    eid = 'Images:regionprops:binaryInput';
    msg1 = 'Use bwlabel(BW) or double(BW) convert binary image to ';
    msg2 = 'a label matrix before calling regionprops.';
    msg = sprintf('%s\n%s',msg1,msg2);
    error(eid, '%s', msg);
  end

  iptcheckinput(L, {'numeric'}, {'real', 'integer', 'nonnegative'}, ...
                mfilename, 'L', 1);

  list = varargin(2:end);
  if (~isempty(list) && ~iscell(list{1}) && strcmp(lower(list{1}), 'all'))
    reqStats = officialStats;
    reqStatsIdx = 1:length(officialStats);
    
  elseif (isempty(list) || (~iscell(list{1}) && strcmp(lower(list{1}),'basic')))
    % Default list
    reqStats = {'Area'
                'Centroid'
                'BoundingBox'};
  else
    
    if (iscell(list{1}))
      list = list{1};
    end
    list = list(:);

    officialStatsL = lower(officialStats);
    
    reqStatsIdx = [];
    eid = sprintf('Images:%s:invalidMeasurement',mfilename);
    for k = 1:length(list)
      if (~ischar(list{k}))
        msg = sprint('This measurement is not a string: "%d".', list{k});
        error(eid,'%s',msg);
      end
      
      idx = strmatch(lower(list{k}), officialStatsL);
      if (isempty(idx))
        msg = sprintf('Unknown measurement: "%s".', list{k});
        error(eid,'%s',msg);
        
      elseif (length(idx) > 1)
        msg = sprintf('Ambiguous measurement: "%s".', list{k});
        error(eid,'%s',msg);
        
      else
        reqStatsIdx = [reqStatsIdx; idx];
      end
    end
    
    reqStats = officialStats(reqStatsIdx);
  end

%%%
%%% NoNDSupport
%%%
function NoNDSupport(str)
%   Issue a warning message about lack of N-D support for a given
%   measurement or measurements.
  
  wid = sprintf('Images:%s:measurementNotForN-D',mfilename);

  if iscell(str)
    warn_str = sprintf('%s: %s ', ...
                       'These measurements are not supported if ndims(L) > 2.', ...
                       sprintf('%s ', str{:}));
  else
    warn_str = sprintf('%s: %s', ...
                       'This measurement is not supported if ndims(L) > 2.', ...
                       str);
  end

  warning(wid,'%s',warn_str);

%%%
%%% PreprocessRequestedStats
%%%
function requestedStats = PreprocessRequestedStats(requestedStats)
%   Remove any requested stats that are not supported for N-D input
%   and issue an appropriate warning.

  no_nd_measurements = {'MajorAxisLength'
                      'MinorAxisLength'
                      'Eccentricity'
                      'Orientation'
                      'ConvexHull'
                      'ConvexImage'
                      'ConvexArea'
                      'EulerNumber'
                      'Extrema'
                      'EquivDiameter'
                      'Solidity'
                      'Extent'
                      'Perimeter'};

  bad_stats = find(ismember(requestedStats, no_nd_measurements));
  if ~isempty(bad_stats)
    NoNDSupport(requestedStats(bad_stats));
  end

  requestedStats(bad_stats) = [];
