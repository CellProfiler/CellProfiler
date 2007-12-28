% FREQEST - Estimate fingerprint ridge frequency within image block
%
% Function to estimate the fingerprint ridge frequency within a small block
% of a fingerprint image.  This function is used by RIDGEFREQ
%
% Usage:
%  freqim =  freqest(im, orientim, windsze, minWaveLength, maxWaveLength)
%
% Arguments:
%         im       - Image block to be processed.
%         orientim - Ridge orientation image of image block.
%         windsze  - Window length used to identify peaks. This should be
%                    an odd integer, say 3 or 5.
%         minWaveLength,  maxWaveLength - Minimum and maximum ridge
%                     wavelengths, in pixels, considered acceptable.
% 
% Returns:
%         freqim    - An image block the same size as im with all values
%                     set to the estimated ridge spatial frequency.  If a
%                     ridge frequency cannot be found, or cannot be found
%                     within the limits set by min and max Wavlength
%                     freqim is set to zeros.
%
% Suggested parameters for a 500dpi fingerprint image
%   freqim = freqest(im,orientim, 5, 5, 15);
%
% See also:  RIDGEFREQ, RIDGEORIENT, RIDGESEGMENT
%
% Note I am not entirely satisfied with the output of this function.

% Peter Kovesi 
% School of Computer Science & Software Engineering
% The University of Western Australia
% pk at csse uwa edu au
% http://www.csse.uwa.edu.au/~pk
%
% January 2005

    
function freqim =  freqest(im, orientim, windsze, minWaveLength, maxWaveLength)
    
    debug = 0;
    
    [rows,cols] = size(im);
    
    % Find mean orientation within the block. This is done by averaging the
    % sines and cosines of the doubled angles before reconstructing the
    % angle again.  This avoids wraparound problems at the origin.
    orientim = 2*orientim(:);    
    cosorient = mean(cos(orientim));
    sinorient = mean(sin(orientim));    
    orient = atan2(sinorient,cosorient)/2;

    % Rotate the image block so that the ridges are vertical
    rotim = imrotate(im,orient/pi*180+90,'nearest', 'crop');
    
    % Now crop the image so that the rotated image does not contain any
    % invalid regions.  This prevents the projection down the columns
    % from being mucked up.
    cropsze = fix(rows/sqrt(2)); offset = fix((rows-cropsze)/2);
    rotim = rotim(offset:offset+cropsze, offset:offset+cropsze);

    % Sum down the columns to get a projection of the grey values down
    % the ridges.
    proj = sum(rotim);
    
    % Find peaks in projected grey values by performing a greyscale
    % dilation and then finding where the dilation equals the original
    % values. 
    dilation = ordfilt2(proj, windsze, ones(1,windsze));
    maxpts = (dilation == proj) & (proj > mean(proj));
    maxind = find(maxpts);

    % Determine the spatial frequency of the ridges by divinding the
    % distance between the 1st and last peaks by the (No of peaks-1). If no
    % peaks are detected, or the wavelength is outside the allowed bounds,
    % the frequency image is set to 0
    if length(maxind) < 2
	freqim = zeros(size(im));
    else
	NoOfPeaks = length(maxind);
	waveLength = (maxind(end)-maxind(1))/(NoOfPeaks-1);
	if waveLength > minWaveLength & waveLength < maxWaveLength
	    freqim = 1/waveLength * ones(size(im));
	else
	    freqim = zeros(size(im));
	end
    end

    
    if debug
	show(im,1)
	show(rotim,2);
	figure(3),    plot(proj), hold on
	meanproj = mean(proj)
	if length(maxind) < 2
	    fprintf('No peaks found\n');
	else
	    plot(maxind,dilation(maxind),'r*'), hold off
	    waveLength = (maxind(end)-maxind(1))/(NoOfPeaks-1);
	end
    end
    