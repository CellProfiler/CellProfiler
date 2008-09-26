function differences = compare_measurements(folder)
% This is an ridicuously inelegant script to compare measurement results
% between the old (pre-5122) and the new (> 5701). It takes a folder as 
% input and outputs a strcuture with fields corresponding to the shared
% measurments between the two handles, with a value equal to the RMS 
% difference between them.
% 
% It assumes that the two results are in DefaultOUT.mat and 
% DefaultOUT__1.mat for a given example directory. 
% 
% It uses objdiff, a user-contributed fxn found by David Logan
%
% Written by Mark Bray, last edited on 8/29/08

cd(['C:\CellProfiler\ExampleImages\',folder]);
load('DefaultOUT.mat'); handles_new = handles;
load('DefaultOUT__1.mat'); handles_old = CP_convert_old_measurements(handles);
obj = objdiff(handles_old.Measurements, handles_new.Measurements);
fn1 = fieldnames(obj);
for i = fn1',
    if ~any(cellfun('isempty',obj.(i{:}))),
        prefix = 'Location';
        if any(strncmp(fieldnames(obj.(i{:}){2}),prefix,length(prefix))),
            obj.(i{:}){1}.Location_Center_X = obj.(i{:}){1}.Location_CenterX;
            obj.(i{:}){1}.Location_Center_Y = obj.(i{:}){1}.Location_CenterY;
        end
        for prefix = {'AreaOccupied','Intensity'}
            if any(strncmp(fieldnames(obj.(i{:}){2}),prefix,length(prefix{:}))),
                idx = find(strncmp(fieldnames(obj.(i{:}){2}),prefix,length(prefix{:})));
                fn2 = fieldnames(obj.(i{:}){2});
                for j = 1:length(idx),
                    k = findstr(fn2{idx(j)},'_');
                    str1 = fn2{idx(j)}(1:k(1)-1); str2 = fn2{idx(j)}(k(1)+1:k(2)-1); str3 = fn2{idx(j)}(k(2)+1:end);
                    obj.(i{:}){2}.([str1,'_',str3,'_',str2]) = obj.(i{:}){2}.(fn2{idx(j)});
                end
            end
        end
        fn2 = intersect(fieldnames(obj.(i{:}){1}),fieldnames(obj.(i{:}){2}));
        if ~isempty(fn2),
            features{find(strcmp(i,fn1))} = {i,fn2};
            for j = fn2',
                if length(obj.(i{:}){1}.(j{:})) > 1,
                    d = [];
                    for k = 1:length(obj.(i{:}){1}.(j{:})),
                        d = cat(1,d,obj.(i{:}){1}.(j{:}){k} - obj.(i{:}){2}.(j{:}){k});
                    end
                else
                    if iscell(obj.(i{:}){1}.(j{:}){:}),
                        if strcmp(j{:},'IdentityOfNeighbors'),  % Neighbor measurement
                            d = sort(cat(2,obj.(i{:}){1}.(j{:}){:}{:})) - sort(cat(2,obj.(i{:}){1}.(j{:}){:}{:}));
                        end
                    else    % Straight-up vector array
                        d = cell2mat(obj.(i{:}){1}.(j{:})) - cell2mat(obj.(i{:}){2}.(j{:}));
                    end
                end
                diffmeas{find(strcmp(i,fn1)),find(strcmp(j,fn2))} = sqrt(mean(d.^2));
            end
        end
    end
end
if exist('features','var'),
    k = cellfun('isempty',features);
    diffmeas(k,:) = [];
    features(k) = [];
    for i = 1:length(features),
        for j = 1:length(features{i}{2}),
            differences.(features{i}{1}{:}).(features{i}{2}{j}) = diffmeas(i,j);
        end
    end
else
    differences = [];
end