function norm_arrays_brokenfolders
% Script to assure that the normalization values are in the correct order
% with respect to the Batch_data.mat, especialy when plates can be broken
% across folders
% M. Bray, modified by D. Logan 2009.08.12

keep_going = 'Continue';

while strcmp(keep_going,'Continue')
    
    % Load the files
    [text_filename,text_pathname] = uigetfile('*.txt');
    disp(text_filename)
    [well,val] = textread(fullfile(text_pathname,text_filename),'%s %f\n','delimiter','\t');
    [batch_filename,batch_pathname] = uigetfile('*.mat');
    disp(batch_pathname)
    load(fullfile(batch_pathname,batch_filename));
    
    % Extract well row and column info
    c = regexp(handles.Pipeline.FileListOrigNuclei,'.*_(?<WellRow>[A-P])(?<WellColumn>[0-9]{1,2})','tokens','once');
    
    % Join row and column to get well name
    d = cellfun(@cell2mat,c,'UniformOutput',false);
    
    [ignore,idx] = ismember(d,well);
    val = val(idx);
    
    % Write re-arranged normalization values to file
    [pathname,name,ext] = fileparts(fullfile(text_pathname,text_filename));
    fid = fopen(fullfile(text_pathname,[name,'_modified',ext]),'wt');
    fprintf(fid,'DESCRIPTION Normalization factors for CorrNuclei\n');
    fprintf(fid,'%.6f\n',val);
    fclose(fid);
    
    % Insert re-arranged normalization values in handles and save
    loadtext_name = 'LoadedText_normalization';
    val = reshape(cellstr(num2str(val,'%.6f')),size(handles.Measurements.Image.(loadtext_name)));
    handles.Measurements.Image.(loadtext_name) = val;
    [pathname,name,ext] = fileparts(fullfile(batch_pathname,batch_filename));
    % Move batch_data to batch_data_old
    movefile(fullfile(batch_pathname,batch_filename),fullfile(batch_pathname,[name,'_old',ext]));
    % Save new batch_data
    save(fullfile(batch_pathname,batch_filename),'handles','number_of_image_sets');
    
    keep_going = CPquestdlg(['Done with ' fullfile(text_pathname,[name,'_modified',ext]) ...
        '.  Continue with other files, or quit to stop execution?'],'Continue?','Continue','Quit','Continue');
    
end