function CPTestRebasePreferences(Directory,TestDirectory)
%%%
%%% Given a root directory of "Directory" (defaults to pwd), change
%%% all of the CellProfilerPreferences.mat files so that their
%%% image directories point to ExampleImages in this directory hierarchy
%%% and so their output directories are the directory containing
%%% CellProfilerPrefreences.mat and so that their module directory
%%% is the module directory of the hierarchy.
%%%
if nargin < 1
    Directory=pwd;
end

if nargin < 2
    TestDirectory = fullfile(Directory,'Test');
end


listing = dir(TestDirectory);
listing=listing([listing.isdir] & (~strcmp({listing.name},'Framework')));
if ~isempty(listing)
    for i=1:length(listing)
        entry = listing(i);
        if entry.name(1)=='.' || strcmp(entry.name,'Framework')
            continue
        end
        if entry.isdir
            RebasePreferences(fullfile(TestDirectory,entry.name),Directory);
        end
    end
end

function RebasePreferences(TestDirectory, BaseDirectory)
PreferencesFile = fullfile(TestDirectory,'CellProfilerPreferences.mat');
if exist(PreferencesFile,'file')
    x=load(PreferencesFile);
    ei = strfind(x.SavedPreferences.DefaultImageDirectory,'ExampleImages');
    if ~isempty(ei)
        ei = ei+length('ExampleImages')+1;
        ExampleImagesDirectory = fullfile(BaseDirectory,'ExampleImages');
        NewDirectory=fullfile(ExampleImagesDirectory,x.SavedPreferences.DefaultImageDirectory(ei:end));
        if ~ strcmp(NewDirectory,x.SavedPreferences.DefaultImageDirectory)
            disp(sprintf('%s: Changed image directory from %s to %s',x.SavedPreferences.DefaultImageDirectory,NewDirectory));
            x.SavedPreferences.DefaultImageDirectory = NewDirectory;
        end
    end
    x.SavedPreferences.DefaultOutputDirectory=TestDirectory;
    x.SavedPreferences.DefaultModuleDirectory=fullfile(BaseDirectory,'Modules');
    save(PreferencesFile,'-struct','x','SavedPreferences');
end
listing = dir(TestDirectory);
for i=1:length(listing)
    entry = listing(i);
    if entry.isdir && entry.name(1)~='.'
        RebasePreferences(fullfile(TestDirectory,listing(i).name), BaseDirectory);
    end
end
