function scores = scoreBoxesOneSize_vect(imagefilename,sz,angle_step,step_sz,numQuadrants)
% calculate the scores for all boxes of a single size

    fprintf('Starting scoreBoxesOneSize_vect(%s,%d,%d,%d)\n',imagefilename,sz,angle_step,step_sz);
    [I,D,mask] = getRotatedImagesAndMask(imagefilename);
    
    fprintf('finished rotating images\n')
    tic;

    scores = calculateScores_vect(I,D,mask,sz,step_sz,angle_step,numQuadrants);

    toc;
    fprintf('finished Scoring\n')
