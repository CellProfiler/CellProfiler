function CPTestFilterExpectedOUT(Directory)
%
% Remove fields from the ExpectedOUT.mat file that shouldn't be expected
% to match.

if nargin < 1
    Directory = fullfile(pwd,'Test');
end


listing = dir(Directory);
for i=find(arrayfun(@(x) x.isdir && x.name(1)~='.' && (~strcmp(x.name,'Framework')),listing))
    entry = listing(i);
    if entry.name(1)=='.' || strcmp(entry.name,'Framework')
        continue
    end
    if entry.isdir
        FilterExpected(fullfile(Directory,entry.name));
    end
end


function FilterExpected(Directory)
ExpectedOUTFile = fullfile(Directory,'ExpectedOUT.mat');
if exist(ExpectedOUTFile,'file')
    x=load(ExpectedOUTFile);
    if isfield(x.handles.Measurements,'Image')
        names=fieldnames(x.handles.Measurements.Image);
        names=names(cellfun(@(x) strncmp(x,'PathName_',9) || strncmp(x,'ModuleError_',12),names));
        x.handles.Measurements.Image = rmfield(x.handles.Measurements.Image,names);
    end
    save(ExpectedOUTFile,'-struct','x','handles');
end
listing = dir(Directory);
for i=1:length(listing)
    entry = listing(i);
    if entry.isdir && entry.name(1)~='.'
        FilterExpected(fullfile(Directory,listing(i).name));
    end
end
