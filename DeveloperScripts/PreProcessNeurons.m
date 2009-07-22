%% Preprocess neural images with I + tophat - bothat, as done in 
%% Zhang et al 2007 JNMeth "A novel tracing algorithm for high throughput
%% imaging Screening of neuron-based assays"

clear

doLogTransform = 1;

%%% CHOOSE ONE SECTION BELOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% JEN PAN images
% strel_size = 4;
% PathRoot = '/Volumes/Image08/hcs/';
% PathName = 'KuaiLetian/ADSAHippo7DIVLithiumMap2/2008-04-03/7190';
% OutputPathRoot = '/Volumes/imaging_analysis/2008_04_15_Lithium_Neurons_JenPan/';
% OutputPathName = PathName;
% search_string = '.*_s[0-9][a-zA-Z0-9].*.tif';
%%%% 

%%%% JEN PAN images from 2008-05-09
% strel_size = 4;
% PathRoot = '/Volumes/Image08/hcs/';
% PathName = 'JenPan/ADSAPlate0423-1-4x/2008-05-09/8069';
% OutputPathRoot = '/Volumes/imaging_analysis/2008_04_15_Lithium_Neurons_JenPan/image08_ADSAPlate0423-1-4x_8069/';
% OutputPathName = PathName;
% search_string = '.*_s[0-9]_w[0-9][a-zA-Z0-9].*.tif';
%%%% 

%%%% JEN PAN images analyzed 2009-05-11 : SERVER
% strel_size = 4;
% PathRoot = '/Volumes/psych_neuro/people/jpan/Jen_Pan_dendrite_Li/';
% PathName = '080716/plate 2';
% OutputPathRoot = '/Volumes/imaging_analysis/2008_04_15_Lithium_Neurons_JenPan/';
% OutputPathName = [PathName '_pre_processed_with_tophat_minus_bottomhat_size4'];
% search_string = '.*_[B-G][0-9][0-9].*.TIF'; %% Rows A and H are empty
%%%% 

%%%% JEN PAN images analyzed 2009-05-11 : LOCAL
strel_size = 4;

% PathRoot = '/Users/dlogan/Projects/2008_04_15_Lithium_Neurons_JenPan/images/';
% PathName = '080716/plate 2';
% OutputPathRoot = '/Users/dlogan/Projects/2008_04_15_Lithium_Neurons_JenPan/images/';
% OutputPathName = [PathName '_pre_processed_with_tophat_minus_bottomhat_size4'];
% if doLogTransform
%     OutputPathName = [OutputPathName '_logTransformed'];
% end
% search_string = '.*_[B-G][0-9][0-9].*.TIF'; %% Rows A and H are empty
% 
% 
% PathRoot = '/Volumes/Image08/hcs/';
% PathName = 'JenPan/ADSAPlate0423-1-4x/2008-05-09/8069';
% OutputPathRoot = '/Volumes/imaging_analysis/2008_04_15_Lithium_Neurons_JenPan/image08_ADSAPlate0423-1-4x_8069/';
% OutputPathName = PathName;
% search_string = '.*_s[0-9]_w[0-9][a-zA-Z0-9].*.tif';


PathRoot = '/Users/dlogan/Projects/2008_04_15_Lithium_Neurons_JenPan/images/';
PathName = '080716/plate 2';
OutputPathRoot = '/Users/dlogan/Projects/2008_04_15_Lithium_Neurons_JenPan/images/';
OutputPathName = [PathName '_pre_processed_with_tophat_minus_bottomhat_size4'];
search_string = '.*_[B-G][0-9][0-9].*.TIF'; %% Rows A and H are empty

%%%% 


%%%% LETIAN KUAI images
% strel_size = 2;
% PathRoot = '/Volumes/IMAGING_ANALYSIS/2007_06_28_Neuron_Projects,_Steve_Haggarty_Lab/LK-1056-060816-Nrg1-2105-Day2_Plate_914/TimePoint_1';
% PathName = '';
% OutputPathRoot = PathRoot;
% OutputPathName = '/preprocessed';
% search_string = '^LK.*[A-Z][0-9][0-9].TIF';
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% MAIN
SE = strel('disk',strel_size);

Files = dir([PathRoot PathName]);
idx = 1;
for i = 1:length(Files)
    Str = regexp(Files(i).name,search_string);
    if ~isempty(Str)
        Good{idx} = Files(i).name;
        idx = idx + 1;
    end
end

for i = length(Good):-1:1
    disp(i)
    I = imread(fullfile([PathRoot PathName],Good{i}));
    
    %% Log transform
    if doLogTransform
        if max(I(:)) > 0,
            ImageNoZeros = max(I, min(I(I > 0)));
            I = log2(double(ImageNoZeros));
        else
            I = zeros(size(I));
        end
    end

    J = imsubtract(imadd(I,imtophat(I,SE)), imbothat(I,SE));
    colormap(gray)

    [foo,OutputFileBase] = fileparts(Good{i});
    if ~exist([OutputPathRoot OutputPathName],'dir')
        mkdir([OutputPathRoot OutputPathName])
    end
    imwrite(J,fullfile([OutputPathRoot OutputPathName],[OutputFileBase '.png']),'png')

end
