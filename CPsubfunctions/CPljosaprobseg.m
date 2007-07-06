function pmask = ljosaprobseg(image, restartprobability, nwalks, seed)
%LJOSAPROBSEG Perform probabilistic segmentation.
% LJOSAPROBSEG(IMAGE, RESTARTPROBABILITY, NWALKS, SEED) segments the
% object identified by SEED in IMAGE using Ljosa and Singh's algorithm
% [doi:10.1109/ICDM.2006.129].
%
% IMAGE can be uint8 or double.  RESTARTPROBABILITY should be a small
% probability, such as 0.001 or 0.0001.  The number of walks, NWALKS,
% only affects the resolution of the result; 1000 is usually
% sufficient.  The seed is a two-column matrix, with one row for each
% pixel in the seed.  The first column contains x-coordinates and the
% second column contains y-coordinates of the seed's pixels.  
%
% The resulting probabilistic mask is clipped at the lowest stationary
% probability within the seed and scaled to [0, 1].
error('ljosaprobseg mexFunction not found');
