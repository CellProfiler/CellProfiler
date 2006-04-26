function FindImageSetNumberGivenImageNameForBatches(handles,fieldname, imagename)

%%% This script helps you find which image set corresponds to a given
%%% image name so you can re-run image analysis on a particular image.
%%% It's not been thoroughly tested.

%%% To use this, load the MATLAB file 'Batch_data.mat' for the batch.
%%% Then find the fieldname in handles.Pipeline that corresponds to
%%% the images of interest, like this:
%%% Example fieldname:
%%% fieldname = 'FileListf00d0';

CharListOfFilenames = char(handles.Pipeline.(fieldname));

%%% The following looks for an image file name and tells you which
%%% image set it was.

clear Found,
for i = 1:length(CharListOfFilenames),
    Found{i} = findstr(CharListOfFilenames(i,:),imagename);
end

for i=1:length(Found),
    if Found{i}==1,
        i
    end
end