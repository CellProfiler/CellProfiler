% RIDGESEGMENT - Normalises fingerprint image and segments ridge region
%
% Function identifies ridge regions of a fingerprint image and returns a
% mask identifying this region.  It also normalises the intesity values of
% the image so that the ridge regions have zero mean, unit standard
% deviation.
%
% This function breaks the image up into blocks of size blksze x blksze and
% evaluates the standard deviation in each region.  If the standard
% deviation is above the threshold it is deemed part of the fingerprint.
% Note that the image is normalised to have zero mean, unit standard
% deviation prior to performing this process so that the threshold you
% specify is relative to a unit standard deviation.
%
% Usage:   [normim, mask, maskind] = ridgesegment(im, blksze, thresh)
%
% Arguments:   im     - Fingerprint image to be segmented.
%              blksze - Block size over which the the standard
%                       deviation is determined (try a value of 16).
%              thresh - Threshold of standard deviation to decide if a
%                       block is a ridge region (Try a value 0.1 - 0.2)
%
% Returns:     normim - Image where the ridge regions are renormalised to
%                       have zero mean, unit standard deviation.
%              mask   - Mask indicating ridge-like regions of the image, 
%                       0 for non ridge regions, 1 for ridge regions.
%              maskind - Vector of indices of locations within the mask. 
%
% Suggested values for a 500dpi fingerprint image:
%
%   [normim, mask, maskind] = ridgesegment(im, 16, 0.1)
%
% See also: RIDGEORIENT, RIDGEFREQ, RIDGEFILTER

% Peter Kovesi         
% School of Computer Science & Software Engineering
% The University of Western Australia
% pk at csse uwa edu au
% http://www.csse.uwa.edu.au/~pk
%
% January 2005


function [normim, mask, maskind] = ridgesegment(im, blksze, thresh)
    
    im = normalise(im,0,1);  % normalise to have zero mean, unit std dev
    
    fun = inline('std(x(:))*ones(size(x))');
    
    stddevim = blkproc(im, [blksze blksze], fun);
    
    mask = stddevim > thresh;
    maskind = find(mask);
    
    % Renormalise image so that the *ridge regions* have zero mean, unit
    % standard deviation.
    im = im - mean(im(maskind));
    normim = im/std(im(maskind));    
