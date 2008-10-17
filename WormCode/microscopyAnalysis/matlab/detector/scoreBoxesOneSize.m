function scores = scoreBoxesOneSize(imagefilename,sz,angle_step,step_sz,numQuadrants)
% calculate the scores for all boxes of a single size

    fprintf('Starting scoreBoxesOneSize(%s,%d,%d,%d)\n',imagefilename,sz,angle_step,step_sz);
    [I,D,mask] = getRotatedImagesAndMask(imagefilename);
    
    fprintf('finished rotating images\n')

    scores = calculateScores(I,D,mask,sz,step_sz,angle_step,numQuadrants);

    fprintf('finished Scoring\n')
