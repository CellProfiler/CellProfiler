function CombineFourImageQuadrants

PathName = uigetdir(pwd,'Select the folder with the four images in .mat format');
if PathName == 0
    return
end
%%% Uses subfunction below.
FileNames = RetrieveImageFileNames(PathName,'N')


%%% Loads the images.
load(fullfile(PathName,FileNames{1}));
TopLeftOfWellImage = MeanImage;
load(fullfile(PathName,FileNames{2}));
TopRightOfWellImage = MeanImage;
load(fullfile(PathName,FileNames{3}));
BottomRightOfWellImage = MeanImage;
load(fullfile(PathName,FileNames{4}));
BottomLeftOfWellImage = MeanImage;

%%% Rearranges so that the top/left bottom/right refers to the portion
%%% of one field of view.
BottomRightImage = TopLeftOfWellImage;
BottomLeftImage = TopRightOfWellImage;
TopLeftImage = BottomRightOfWellImage;
TopRightImage = BottomLeftOfWellImage;

%%% Determines sizes.
[TopLeftImageHeight, TopLeftImageWidth] = size(TopLeftImage);
[TopRightImageHeight, TopRightImageWidth] = size(TopRightImage);
[BottomLeftImageHeight, BottomLeftImageWidth] = size(BottomLeftImage);
[BottomRightImageHeight, BottomRightImageWidth] = size(BottomRightImage);

if TopLeftImageHeight ~= TopRightImageHeight
    errordlg('The heights of the top two images is not the same')
    return
end
if BottomLeftImageHeight ~= BottomRightImageHeight
    errordlg('The heights of the bottom two images is not the same')
    return
end
if TopLeftImageWidth ~= BottomLeftImageWidth
    errordlg('The widths of the two left images is not the same')
    return
end
if TopRightImageWidth ~= BottomRightImageWidth
    errordlg('The widths of the two right images is not the same')
    return
end
WholeImageWidth = TopLeftImageWidth + TopRightImageWidth;
WholeImageHeight = TopLeftImageHeight + BottomLeftImageHeight;

%%% Preallocates the array to be the proper size.
WholeImage(WholeImageHeight, WholeImageWidth) = 0;

WholeImage(1:TopLeftImageHeight,1:TopLeftImageWidth) = TopLeftImage;
figure, subplot(2,2,1), imagesc(WholeImage), colormap(gray), title(FileNames{3})
WholeImage(1:TopRightImageHeight,TopLeftImageWidth+1:WholeImageWidth) = TopRightImage;
subplot(2,2,2), imagesc(WholeImage), colormap(gray), title(FileNames{4})
WholeImage(TopLeftImageHeight+1:WholeImageHeight,1:BottomLeftImageWidth) = BottomLeftImage;
subplot(2,2,3), imagesc(WholeImage), colormap(gray), title(FileNames{2})
WholeImage(TopRightImageHeight+1:WholeImageHeight,BottomLeftImageWidth+1:WholeImageWidth) = BottomRightImage;
subplot(2,2,4), imagesc(WholeImage), colormap(gray), title(FileNames{1})
MeanImage = WholeImage;

FileName = inputdlg('What do you want to call the resulting file? (.mat will be added automatically)','Name the resulting file',1,{'WholeMeanImage'})
save([FileName{1},'.mat'], 'MeanImage')
msgbox('The file was saved to the current directory')

function FileNames = RetrieveImageFileNames(Pathname,recurse)
%%% Lists all the contents of that path into a structure which includes the
%%% name of each object as well as whether the object is a file or
%%% directory.
FilesAndDirsStructure = dir(Pathname);
%%% Puts the names of each object into a list.
FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
%%% Puts the logical value of whether each object is a directory into a list.
LogicalIsDirectory = [FilesAndDirsStructure.isdir];
%%% Eliminates directories from the list of file names.
FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);
if isempty(FileNamesNoDir) == 1
    FileNames = [];
else
    %%% Makes a logical array that marks with a "1" all file names that start
    %%% with a period (hidden files):
    DiscardLogical1 = strncmp(FileNamesNoDir,'.',1);
    %%% Makes logical arrays that mark with a "1" all file names that have
    %%% particular suffixes (mat, m, m~, and frk). The dollar sign indicates
    %%% that the pattern must be at the end of the string in order to count as
    %%% matching.  The first line of each set finds the suffix and marks its
    %%% location in a cell array with the index of where that suffix begins;
    %%% the third line converts this cell array of numbers into a logical
    %%% array of 1's and 0's.   cellfun only works on arrays of class 'cell',
    %%% so there is a check to make sure the class is appropriate.  When there
    %%% are very few files in the directory (I think just one), the class is
    %%% not cell for some reason.
    DiscardsByExtension = regexpi(FileNamesNoDir, '\.(m|m~|frk~|xls|doc|rtf|txt|csv)$', 'once');
    if strcmp(class(DiscardsByExtension), 'cell')
        DiscardsByExtension = cellfun('prodofsize',DiscardsByExtension);
    else
        DiscardsByExtension = [];
    end
    %%% Combines all of the DiscardLogical arrays into one.
    DiscardLogical = DiscardLogical1 | DiscardsByExtension;
    %%% Eliminates filenames to be discarded.
    if isempty(DiscardLogical) == 1
        FileNames = FileNamesNoDir;
    else FileNames = FileNamesNoDir(~DiscardLogical);
    end
end
if(strcmp(upper(recurse),'Y'))
    DirNamesNoFiles = FileAndDirNames(LogicalIsDirectory);
    DiscardLogical1Dir = strncmp(DirNamesNoFiles,'.',1);
    DirNames = DirNamesNoFiles(~DiscardLogical1Dir);
    if (length(DirNames) > 0)
        for i=1:length(DirNames),
            MoreFileNames = RetrieveImageFileNames(fullfile(Pathname,char(DirNames(i))), recurse);
            for j = 1:length(MoreFileNames)
                MoreFileNames{j} = fullfile(char(DirNames(i)), char(MoreFileNames(j)));
            end
            if isempty(FileNames) == 1
                FileNames = MoreFileNames;
            else
                FileNames(end+1:end+length(MoreFileNames)) = MoreFileNames(1:end);
            end
        end
    end
end
