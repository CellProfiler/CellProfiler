% function LABELS_OUT = AlgSegmentSecPropagateSubfunction(LABELS_IN, IMAGE, MASK, LAMBDA)
% 
% Propagate labels from LABELS_IN to LABELS_OUT, steered by IMAGE and
% limited to MASK.  MASK should be a logical array.  LAMBDA is a
% regularization paramter, larger being closer to Euclidean distance
% in the image plane, and zero being entirely controlled by IMAGE.
% 
% Propagation of labels is by shortest path to a nonzero label in
% LABELS_IN.  Distance is the sum of absolute differences in the image
% in a 3x3 neighborhood, combined with LAMBDA via 
% sqrt(differences^2 + LAMBDA^2).
%
% Note that there is no separation between adjacent areas with
% different labels (as there would be using, e.g., watershed).  Such
% boundaries must be added in a postprocess.
