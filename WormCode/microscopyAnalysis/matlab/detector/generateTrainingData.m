function generateTrainingData(imagefilename,imageLinesFileName,datafilename,sz)
% A matlab script that transforms a preprocessed image and a lines file
% into a dataset for jboost
% it calls the python-generated procedures: 
%   * getRotatedImagesAndMask : read the mask and images and rotates the image
%   * calcF : maps a set of boxes into a feature vector.

    fprintf('rotating images\n')
    [I,D,mask] = getRotatedImagesAndMask(imagefilename);

%     [x1,y1,x2,y2,weight,label] = textread([imagefilename '.tif.bix.lines'], '%f,%f,%f,%f,%f,%s');
    
    [x1,y1,x2,y2,weight,label] = textread(imageLinesFileName, '%f,%f,%f,%f,%f,%s');
    
    linelist(:,1)=x1;
    linelist(:,2)=y1;
    linelist(:,3)=x2;
    linelist(:,4)=y2;
    
    fprintf('sz=%f\n',sz);
    [Iboxes,Dboxes,labels,weights] = extractBoxes(I,D,linelist,label,weight,-sz);
    maskCell{size(mask,1)} = mask;
    maskTemplate = mask;

    fprintf('calculating features\n');
    datafile = fopen(datafilename,'a');
    for i=1:size(Iboxes,1)
        curSize = size(Iboxes{i,1},1);
        if length(maskCell)<curSize 
            curMask = imresize(maskTemplate,[curSize,curSize])>0.5;
            maskCell{curSize} = curMask;
        elseif isempty(maskCell{curSize})
            curMask = imresize(maskTemplate,[curSize,curSize])>0.5;
            maskCell{curSize} = curMask;
        else
            curMask = maskCell{curSize};
        end

        f= calcF(Iboxes(i,:),Dboxes(i,:),curMask);
        
        fprintf(datafile,'%-.2f,',f);
        fprintf(datafile,'%.2f,%s;\n',weights(i),labels{i});
    end

    fclose(datafile);
