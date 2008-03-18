function parse_filenames( image_file )
%PARSE_FILENAMES Parse image numbers and filenames from a concatenated image.CSV file
%                   into a file with well, site, and wavelength info.
%
%  parse_filenames('/PATH/image.CSV') 
%
% OUTPUT: '/PATH/image_well_info.csv'
%
% NB!! Assumes a concatenated image.csv file!  The order will be incorrect
% otherwise!!!
%
% For now, this only works for files & filenames of this construction ->
% image number, "..._A01_s1_w2...", etc.

%% READ IMAGE.CSV FILE
fid = fopen(image_file);
C = textscan(fid, '%u16 %s %*[^\n]','delimiter',',');
fclose(fid)

image_num = C{1};
filename = C{2};

%% PARSE filename
%% for filenames of this construction -> "..._A01_s1_w2..."
[PreFilename, remain] = strtok(filename,'_');
[well, remain] = strtok(remain,'_');
[site, remain] = strtok(remain,'_');
for i = length(remain):-1:1
    w = remain{i};
    wavelength{i,1} = w(2:3);
end

%% TREATMENT column
% NB will need to be changed for each platemap!
% 
% NOT DONE.  Will be more complicated and require user inputs depending on 
%           platemap layout


%% OUTPUT .csv file
[pathstr, name, ext] = fileparts(image_file);
well_file = fullfile(pathstr, [name '_well_info.csv']);
if ~exist(well_file,'file')
    %% Note: cannot use xlswrite, because Mac's can't run Excel COM server
    %%  nor dlmwrite because it outputs one character at a time
    
    M = [well, site, wavelength];
    fid = fopen(well_file,'w');
    for i = 1:size(M,1)
       fprintf(fid, '%d,%s,%s\n',image_num(i),M{i,1},M{i,2});
    end
    fclose(fid);
else 
    disp(['Cannot write ' well_file ' since it already exists'])
end