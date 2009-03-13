%% Read in data, Scored by Well, possibly with >1 plate
%%
%% Example line:
%% Well, PLATE, PHENOTYPE1_COUNT, PHENOTYPE2_COUNT, PVALUE_PHENOTYPE1, PVALUE_PHENOTYPE2, ENRICHMENT
%% A01,2002-01-W01-02-01-CN00002412-B,3198,2352,0.986617752,0.013382248,1.867619843
%%
%% NOTE!  You must manually strip off the header lines of CPA output .csv file before you run this.  
%% The way newlines are encoded now in the header make it complicated to do this programmatically.   

clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USER SETS THESE
numRows = 4;  
numCols = 5; 
suptitle_all = {'PHENOTYPE1_COUNT', 'PHENOTYPE2_COUNT', 'PVALUE_PHENOTYPE1', 'PVALUE_PHENOTYPE2', 'ENRICHMENT'};
file = '/Users/dlogan/Desktop/CPA_output.csv';
output_folder = 'Data_array_CSV_files'; %% placed in same path as CPA_output.csv 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[pathstr, name, ext, versn] = fileparts(file);

fid = fopen(file,'r');
% fid = fopen('/Users/dlogan/Desktop/My_Enrichment_Data.csv','r');
C = textscan(fid, '%s %s %f %f %f %f %f',...
    'delimiter',',',...
    'Headerlines',0);
fclose(fid);

%% Checks
ALPHABET = 'ABCDEFGHIJKLMNOP';
assert(max(str2double(cellfun(@(x) x(2:3),C{1},'UniformOutput',0))) == 24,'Number of columns ~= 24 (Note: This script only works with 384 well plates yet')
assert(all(cell2mat(regexp(cellfun(@(x) x(1),C{1},'UniformOutput',0), '[A-P]'))),'Number of rowss ~= 16 (Note: This script only works with 384 well plates yet')


%% Sort by plate
plate_list = unique(C{2});

Data = cell2mat(C(3:end));
% max_Data = max(Data);
% min_Data = min(Data);

for idxPlate = length(plate_list):-1:1
    for idxRow = 16:-1:1
        for idxCol = 24:-1:1
            
            idxColStr = CPtwodigitstring(idxCol);
            wellID = [ALPHABET(idxRow) num2str(idxColStr)];
            
            %% Find proper row
            matched_wells = strcmp(C{1}, wellID);
            matched_plates = strcmp(C{2}, plate_list(idxPlate));
            matched_well_and_plate =  find(matched_wells & matched_plates);
            
            if isempty(matched_well_and_plate)
                Data_array(idxRow,idxCol,idxPlate,:) = NaN;
            elseif length(matched_well_and_plate) == 1
                Data_array(idxRow,idxCol,idxPlate,:) = Data(matched_well_and_plate,:);
            else
                error([matched_well_and_plate ' did not match exactly one row'])
            end
        end
    end
end

%% PLOT
xticks = 1:4:24;
yticks = 1:3:16;
iptsetpref('ImshowAxesVisible','on')

%% Create folder for output
output_folder = fullfile(pathstr,output_folder);
mkdir(output_folder)

for idxDataType = size(Data,2):-1:1
    for idxPlate = length(plate_list):-1:1

        figure(idxDataType);
        subplot(numRows,numCols,idxPlate)
        thisData = Data_array(:,:,idxPlate,idxDataType);
        imshow(thisData, [min(thisData(:)) max(thisData(:))])
        colormap(hot)
        if idxPlate == length(plate_list)
            h_suptitle = suptitle(suptitle_all{idxDataType});
            set(h_suptitle,'Interpreter','none')
            h_colorbar = colorbar;
            set(h_colorbar, 'Position', [.95 .1 .02 .2])
            %             colorbar('delete')
        end

        title(plate_list{idxPlate})
        set(gca,'XTick',1:3:24)
        set(gca,'YTick',1:3:16)
        set(gca, 'YTickLabel', 'A|D|G|J|M|P')
        
        %% Export data to Excel
        csvwrite(fullfile(output_folder,[name '_' suptitle_all{idxDataType} '_' plate_list{idxPlate} '.csv']),...
            Data_array(:,:,idxPlate,idxDataType))
    end
end
