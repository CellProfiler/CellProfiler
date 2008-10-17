class detectorFeatures:
    mask = 'EdgeMasks1'
    size = 49
    boxSize = (49,)
    angle_step=10

    # The new format for the pre-processing is meant to allow for a sequence of processing steps, 
    # processing of different files, and different types (stk and tif at this point)
    # The assumption is that the deriv files are generated after all of the imagefiles have been generated.
    # Each list consists of tuples, the tuples are to be processed in order.
    # Each tuple is of the form (old-extension,new-extension,matlabFunc), the pre-processor takes as input a file with the 
    # first extension, runs the matlab function called matlabFunc and generates an output file. 
    # The name of the output file is new-extension appended to the file name.
    #
    # For example ('YFP.stk','YFP.mean.tif','averStk(') takes as input a file "filenameYFP.stk" , applies the matlab function
    # 'averStk' to it and generates a file called "filenameYFP.mean.tif"
    # note the open-paren in 'overStk(', this is a hack that allows passing a parameter to the matlab function, which we are not using for aLoG
    
    PreProcFunctions = ['smooth','scaleImage','aLoG','averStk','deriv']

    imageProcessing = [('Phase.tif','Phase.norm.tif','scaleImage('),('Phase.norm.tif','Phase.smooth.tif','smooth('),
                  ('Phase.tif','Phase.alog4.tif','aLoG(4,'),('Phase.tif','Phase.alog2.tif','aLoG(2,'),('Phase.tif','Phase.alog3.tif','aLoG(3,'),\
                  ('YFP.stk','YFP.mean.tif','averStk('),('YFP.mean.tif','YFP.smooth.tif','smooth('),\
                  ('YFP.mean.tif','YFP.alog3.tif','aLoG(3,'),('YFP.mean.tif','YFP.alog2.tif','aLoG(2,'),('YFP.mean.tif','YFP.alog4.tif','aLoG(4,'),\
                  ('Cherry.stk','Cherry.mean.tif','averStk('),('Cherry.mean.tif','Cherry.smooth.tif','smooth('),\
                  ('Cherry.mean.tif','Cherry.alog3.tif','aLoG(3,'),('Cherry.mean.tif','Cherry.alog2.tif','aLoG(2,'),('Cherry.mean.tif','Cherry.alog4.tif','aLoG(4,'),\
                  ]             # BW images
    derivProcessing = [('Phase.smooth.tif','Phase.deriv.tif','deriv('),\
                  ('YFP.smooth.tif','YFP.deriv.tif','deriv('),\
                  ('Cherry.smooth.tif','Cherry.deriv.tif','deriv('),\
                  ]             # (mag,angle) images

    imagefiles = [item[1] for item in imageProcessing]
    derivfiles = [item[1] for item in derivProcessing]
    
    numQuadrants = 4

# 
#             mask,image ,function,parameter for function 
    features = ( (1   ,'Phase.norm.tif','hist'  ,(0,51,102,153,204,255)),\
                 (1   ,'Phase.alog2.tif','hist'  ,(0,40,100,255)),\
                 (1   ,'Phase.alog3.tif','hist'  ,(0,40,100,255)),\
                 (1   ,'Phase.alog4.tif','hist'  ,(0,40,100,255)),\
                 (1   ,'YFP.mean.tif','hist'  ,(0,51,102,153,204,255)),\
                 (1   ,'YFP.alog2.tif','hist'  ,(0,40,100,255)),\
                 (1   ,'YFP.alog3.tif','hist'  ,(0,40,100,255)),\
                 (1   ,'YFP.alog4.tif','hist'  ,(0,40,100,255)),\
                 (1   ,'Cherry.mean.tif','hist'  ,(0,51,102,153,204,255)),\
                 (1   ,'Cherry.alog2.tif','hist'  ,(0,40,100,255)),\
                 (1   ,'Cherry.alog3.tif','hist'  ,(0,40,100,255)),\
                 (1   ,'Cherry.alog4.tif','hist'  ,(0,40,100,255)),\
                 \
                 (2   ,'Phase.norm.tif','hist'  ,(0,51,102,153,204,255)),\
                 (2   ,'Phase.alog2.tif','hist'  ,(0,40,100,255)),\
                 (2   ,'Phase.alog3.tif','hist'  ,(0,40,100,255)),\
                 (2   ,'Phase.alog4.tif','hist'  ,(0,40,100,255)),\
                 (2   ,'YFP.mean.tif','hist'  ,(0,51,102,153,204,255)),\
                 (2   ,'YFP.alog2.tif','hist'  ,(0,40,100,255)),\
                 (2   ,'YFP.alog3.tif','hist'  ,(0,40,100,255)),\
                 (2   ,'YFP.alog4.tif','hist'  ,(0,40,100,255)),\
                 (2   ,'Cherry.mean.tif','hist'  ,(0,51,102,153,204,255)),\
                 (2   ,'Cherry.alog2.tif','hist'  ,(0,40,100,255)),\
                 (2   ,'Cherry.alog3.tif','hist'  ,(0,40,100,255)),\
                 (2   ,'Cherry.alog4.tif','hist'  ,(0,40,100,255)),\
                 \
                 (3   ,'Phase.deriv.tif','deriv',8),\
                 (3   ,'YFP.deriv.tif','deriv',8),\
                 (3   ,'Cherry.deriv.tif','deriv',8),\
                 )

    classifyStep = 5

    labels={'randomSeg':'-1','cellEdge':'+1'}
    lineColors={"cellEdge": "#000000",\
                         "randomSeg":"#ffff00"} # mapping from line type to line colors in the svm
    lineWeights = {"cellEdge":1,\
                   "randomSeg":1}  
