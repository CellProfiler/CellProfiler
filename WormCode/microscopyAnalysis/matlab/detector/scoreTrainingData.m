function scoreTrainingData(imagefilename,datafilename,sz,score_min,score_max)
% A matlab script that scores an image used in training data 
% into jboost data and scores it.
% Lines output is in datafilename
% only segments with score between score_min and score_max are generated.
% it calls the python-generated procedures: 
%   * getRotatedImagesAndMask : read the mask and images and rotates the image
%   * calcF : maps a set of boxes into a feature vector.

    fprintf('rotating images\n')
    [I,D,mask] = getRotatedImagesAndMask(imagefilename);

    [x1,y1,x2,y2,weight,label] = textread([imagefilename '.tif.bix.lines'], '%f,%f,%f,%f,%f,%s');
    linelist(:,1)=x1;
    linelist(:,2)=y1;
    linelist(:,3)=x2;
    linelist(:,4)=y2;
    
    fprintf('sz=%f\n',sz);

    fprintf('calculating features\n');
    datafile = fopen(datafilename,'w');
    maskCell{size(mask,1)} = mask;
    maskTemplate = mask;
    score = zeros(length(x1),1);
    for i=1:length(x1)
        [Iboxes,Dboxes,labels,weights] = extractBoxes(I,D,linelist(i,:),label(i),weight(i),-sz);
        if(isempty(Iboxes));continue;end
        
        curSize = size(Iboxes{1},1);
        if length(maskCell)<curSize 
            curMask = imresize(maskTemplate,[curSize,curSize])>0.5;
            maskCell{curSize} = curMask;
        elseif isempty(maskCell{curSize})
            curMask = imresize(maskTemplate,[curSize,curSize])>0.5;
            maskCell{curSize} = curMask;
        else
            curMask = maskCell{curSize};
        end

        f= calcF(Iboxes(1,:),Dboxes(1,:),curMask);
        defined = (f==f);
        score(i) = calcScore(f,defined);

        if(score(i)>score_max || score(i)<score_min);continue;end
        if(isempty(strmatch('+1',label(i))))
            fprintf(datafile,'%.2f %.2f %.2f %.2f #ff0000 0 0\n',...
                linelist(i,1),linelist(i,2),linelist(i,3),linelist(i,4));
        else
            fprintf(datafile,'%.2f %.2f %.2f %.2f #00ff00 0 0\n',...
                linelist(i,1),linelist(i,2),linelist(i,3),linelist(i,4));
        end
        
    end

    fclose(datafile);