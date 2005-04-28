function SplitUpAviMovie

Pathname = '/Users/cellprofileruser/Documents/AnneCarpenter/2005_03_30JamesWhittleTimeLapse';
Filename = 'GOOD_GFPHistone2noIntro.avi';
FramesPerSplitMovie = 100;


AviMovieInfo = aviinfo(fullfile(Pathname,Filename));

NumSplitMovies = ceil(AviMovieInfo.NumFrames/FramesPerSplitMovie);

LastFrameRead = 0;
for i = 1:NumSplitMovies
    [Pathname,FilenameWithoutExtension,Extension,ignore3] = fileparts(fullfile(Pathname,Filename));
    NewFileAndPathName = fullfile(Pathname, [FilenameWithoutExtension, '_', num2str(i),Extension]);
    LastFrameToReadForThisFile = min(i*FramesPerSplitMovie,AviMovieInfo.NumFrames);
    LoadedRawImages = aviread(fullfile(Pathname,Filename), LastFrameRead+1:LastFrameToReadForThisFile);
    try movie2avi(LoadedRawImages,NewFileAndPathName)
    catch error('problem encountered during save')
        return
    end
    LastFrameRead = i*FramesPerSplitMovie;
end