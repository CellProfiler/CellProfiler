function parse_filenames(varargin)
%PARSE_FILENAMES Parse image numbers and filenames from a concatenated image.CSV file
%   into a file with well, site, and wavelength info.  
%
%   parse_filenames
%   parse_filenames('/PATH/image.CSV')
%
%   OUTPUT: '<same directory as input files>/image_well_info.csv'
%
% User can subsequntly add columns indicating treatment conditions manually, 
% and subsequently upload to a database meta-data table.

error(nargchk(0, 1, nargin, 'string'))
if nargin < 1
	image_dir = CPuigetdir('HOME', 'Choose the directory where your *image.CSV files are located');
    if image_dir == 0
        return
    end
else
    image_dir = varargin{1};
end

%DEBUG example dir
% image_dir = '/Volumes/imaging_analysis/2008_04_15_Lithium_Neurons_JenPan/2008_08_expts/output_plate1';

if ~ispc
%     cd(image_dir)
    files = dir(fullfile(image_dir,'*_image.CSV'));
%     !cat `ls *_image.CSV | grep image` >image.CSV
else
    error('Does not work on PCs yet')
end

%% READ IMAGE.CSV FILE
image_num = [];
filename = '';
h_bar = CPwaitbar(0,'Parsing files...');

for idx = 1:length(files)
    this_file = fullfile(image_dir,files(idx).name);
    fid = fopen(this_file);
    C = textscan(fid, '%u16 %s %*[^\n]','delimiter',',','BufSize',8192);
    image_num = cat(1,image_num,C{1});
    filename = cat(1,filename,C{2});
    fclose(fid);
    waitbar(idx./length(files))
end
close(h_bar)

if ~all(sort(image_num) == (1:length(image_num))')
    CPwarndlg('The list of image numbers is not complete from 1:end')
end

%% PARSE filename
choice = menu({'Choose the filename construction which matches your files',['Example: ' filename{1}]},...
    '..._A01_s1_w2...',...
    'PANDORA_123456789_A01f01d0.TIF');
switch choice
    case 1
        %% for filenames of this construction -> "..._A01_s1_w2..."
        [PreFilename, remain] = strtok(filename,'_');
        [well, remain] = strtok(remain,'_');
        [site, remain] = strtok(remain,'_');
        for i = length(remain):-1:1
            w = remain{i};
            wavelength{i,1} = w(2:3); %#ok<AGROW>
        end
    case 2
        %% for filenames of this construction -> 'PANDORA_123456789_A01f01d0.TIF'
        [PreFilename, remain] = strtok(filename,'_');
        [date, remain] = strtok(remain,'_');
        [WellFieldChannel, remain] = strtok(remain,'_');
        [well, remain] = strtok(WellFieldChannel,'f');
        [site, wavelength] = strtok(remain,'d');
        
end

%% Split off WELL into 2 columns (e.g. 'A01' -> 'A' and '01')
wellCharArray = cell2mat(well);
if size(wellCharArray,2) ~= 3, error('wells do not all have 3 characters'), end
row = wellCharArray(:,1);      %% 'A'
col = wellCharArray(:,2:3);    %% '01'

if ~all(isletter(row)), error('Problem with well row format.  Should all be letters'), end
if any(isletter(col)), error('Problem with well column format.  Should all be character numbers'), end

rowCellArray = cellstr(row);
colCellArray = cellstr(col);

%% OUTPUT .csv file
well_file = fullfile(image_dir, 'image_well_info.csv');
if ~exist(well_file,'file')
    %% Note: cannot use xlswrite, because Mac's can't run Excel COM server
    %%  nor dlmwrite because it outputs one character at a time
    
    M = [rowCellArray, colCellArray, site, wavelength];
    
    %% Sort by image_num
    [image_num_sorted,IDX] = sort(image_num);
    M_sorted = M(IDX,:);
    fid = fopen(well_file,'w');
    for i = 1:size(M_sorted,1)
       fprintf(fid, '%d,%s,%s,%s,%s\n',image_num_sorted(i),...
           M_sorted{i,1},M_sorted{i,2},M_sorted{i,3},M_sorted{i,4});
    end
    fclose(fid);
    disp(['DONE!  An image_well_info.csv file was written to the same directory a your image.csv file'])
else 
    disp(['Cannot write ' well_file ' since it already exists'])
end