% Help for the Segment Secondary Propagate Subfunction:
% Category: Object Identification
%
% This is a subfunction implemented in C and MEX to perform the
% propagate algorithm (somewhat similar to watershed).  This help
% documents the arguments and behavior of the propagate algorithm.
% 
% Propagate labels from LABELS_IN to LABELS_OUT, steered by IMAGE and
% limited to MASK.  MASK should be a logical array.  LAMBDA is a
% regularization paramter, larger being closer to Euclidean distance
% in the image plane, and zero being entirely controlled by IMAGE.
% 
% Propagation of labels is by shortest path to a nonzero label in
% LABELS_IN.  Distance is the sum of absolute differences in the image
% in a 3x3 neighborhood, combined with LAMBDA via sqrt(differences^2 +
% LAMBDA^2).
%
% Note that there is no separation between adjacent areas with
% different labels (as there would be using, e.g., watershed).  Such
% boundaries must be added in a postprocess.
%
% See also ALGIDENTIFYSECPROPAGATE.

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
%
% $Revision 1.10 $
