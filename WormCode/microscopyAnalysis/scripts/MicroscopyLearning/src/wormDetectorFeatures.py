class detectorFeatures:
    size = -2
    boxSize = (15,20,25,30,35,40,50)
    imgSizeX = 1388
    imgSizeY = 1036
    angle_step=30
    numQuadrants = 2
    

    PreProcFunctions = ['normalizeImage','aLoG','deriv']

    imageProcessing = [('.tif','.norm.tif','normalizeImage(10,'),\
                       ('.norm.tif','.alog1.tif','aLoG(1,'),\
                       ('.norm.tif','.alog2.tif','aLoG(2,'),\
                       ('.norm.tif','.zlog2.tif','aLoG(2,'),\
#                       ('red.tif','red.norm.tif','scaleImage('),\
#                       ('red.tif','red.alog0p5.tif','aLoG(0.5,'),\
#                       ('red.tif','red.alog2.tif','aLoG(2,'),\
                      ]

    derivProcessing = [('.tif','.deriv.tif','deriv('),\
#                       ('red.tif','red.deriv.tif','deriv(')
                    ]
    imagefiles = [item[1] for item in imageProcessing]
    derivfiles = [item[1] for item in derivProcessing]
    


#             mask,image ,function,parameter for function 
#    features = ((1   ,'.norm.tif','hist'  ,(0,51,102,153,204,255)),\
#                (1   ,'.alog0p5.tif','hist'  ,(0,51,102,153,204,255)),\
#                (1   ,'.alog2.tif','hist'  ,(0,51,102,153,204,255)),\
#                (1   ,'red.norm.tif','hist'  ,(0,51,102,153,204,255)),\
#                (1   ,'red.alog0p5.tif','hist'  ,(0,51,102,153,204,255)),\
#                (1   ,'red.alog2.tif','hist'  ,(0,51,102,153,204,255)),\
#                \
#                (2   ,'.norm.tif','hist'  ,(0,51,102,153,204,255)),\
#                (2   ,'.alog0p5.tif','hist'  ,(0,51,102,153,204,255)),\
#                (2   ,'.alog2.tif','hist'  ,(0,51,102,153,204,255)),\
#                (2   ,'red.norm.tif','hist'  ,(0,51,102,153,204,255)),\
#                (2   ,'red.alog0p5.tif','hist'  ,(0,51,102,153,204,255)),\
#                (2   ,'red.alog2.tif','hist'  ,(0,51,102,153,204,255)),\
#                \
#                (3   ,'.deriv.tif','deriv',8),\
#                (3   ,'red.deriv.tif','deriv',8),\
#                )

    mask = 'wormMasks2'
    features = ((1   ,'.norm.tif','perc'  ,(5,20,35,50,65,80,95)),\
                (1   ,'.alog1.tif','perc'  ,(5,20,35,50,65,80,95)),\
                (1   ,'.alog2.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (1   ,'red.norm.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (1   ,'red.alog0p5.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (1   ,'red.alog2.tif','perc'  ,(5,20,35,50,65,80,95)),\
                \
                (2   ,'.norm.tif','perc'  ,(5,20,35,50,65,80,95)),\
                (2   ,'.alog1.tif','perc'  ,(5,20,35,50,65,80,95)),\
                (2   ,'.alog2.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (2   ,'red.norm.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (2   ,'red.alog0p5.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (2   ,'red.alog2.tif','perc'  ,(5,20,35,50,65,80,95)),\
                \
                (5   ,'.norm.tif','perc'  ,(5,20,35,50,65,80,95)),\
                (5   ,'.alog1.tif','perc'  ,(5,20,35,50,65,80,95)),\
                (5   ,'.alog2.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (5   ,'red.norm.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (5   ,'red.alog0p5.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (5   ,'red.alog2.tif','perc'  ,(5,20,35,50,65,80,95)),\
                \
                (3   ,'.deriv.tif','deriv',8),\
                (4   ,'.deriv.tif','deriv',8),\
#                (3   ,'red.deriv.tif','deriv',8),\
#                (4   ,'red.deriv.tif','deriv',8),\
                )
 
#    mask = 'wormMasks1'
#    features = ((1   ,'.norm.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (1   ,'.alog0p5.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (1   ,'.alog2.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (1   ,'red.norm.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (1   ,'red.alog0p5.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (1   ,'red.alog2.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                \
#                (3   ,'.norm.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (3   ,'.alog0p5.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (3   ,'.alog2.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (3   ,'red.norm.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (3   ,'red.alog0p5.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                (3   ,'red.alog2.tif','perc'  ,(5,20,35,50,65,80,95)),\
#                \
#                (2   ,'.deriv.tif','deriv',8),\
#                (2   ,'red.deriv.tif','deriv',8),\
#                )
#    
    
    labels={'randomSeg':'-1','wormSeg':'+1','correct':'+1','incorrect':'-1'}
    lineColors={"wormSeg": "#0088ff",\
                "randomSeg":"#ffff00",\
                "correct": "#00ff00",\
                "incorrect": "#ff0000"\
                } # mapping from line type to line colors
    
    lineWeights = {"wormSeg": 1,\
                "randomSeg": 0.5,\
                "correct": 1,\
                "incorrect": 1\
                }

    classifyStep = 1

    matlabPostProcessing = 'wormHighScoreOutLinesVect'
