import sys,os,os.path
from learningManager import learningManager
from wormDetectorFeatures import detectorFeatures


def main():
    args = sys.argv[1:]
    if len(args)==1:
        runName = args[0]
        imageFile = args[0:]
    elif len(args)>1:
        runName = args[0]
        imageFile = args[1:]
    else:
        runName = "test"
        imageFile = ("norm001",)
#        sys.exit("wormClassifyManager.py [runName=(ImageFile)] ImageFiles")
    
    manager = learningManager()
    manager.initialize(imageFile,runName)
    manager.detectorFeatureVar = detectorFeatures()
    manager.preprocessImage()
    print "done Preprocessing\n"
    print "done preparing data for JBoost"
    manager.scoreImageVect()
    manager.addOutLines()
    print 'done'

if __name__ == "__main__":
    main()

