%#ok function LABELS_OUT = AlgSegmentSecPropagateSubfunction(LABELS_IN, IMAGE, MASK, LAMBDA) %#ok
% 
% Help for the Segment Secondary Propagate Subfunction:
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

% The contents of this file are subject to the Mozilla Public License Version 
% 1.1 (the "License"); you may not use this file except in compliance with 
% the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
% 
% 
% The Original Code is the Segment Secondary Propagate Subfunction.
% 
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
% 
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$
