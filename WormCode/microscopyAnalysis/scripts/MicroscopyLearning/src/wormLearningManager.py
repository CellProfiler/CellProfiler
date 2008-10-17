import sys,os,os.path
from learningManager import learningManager
from wormDetectorFeatures import detectorFeatures
from wormDigitalNotebook import wormCreateLines
from wormDigitalNotebookStage2 import wormLinesSecStage
from vectorizePredictor import vectPredict


def main():
    args = sys.argv[1:]
    
    if len(args)==1:
        runName = args[0]
        imageFile = args[0:]
    elif len(args)>1:
        runName = args[0]
        imageFile = args[1:]
    else:
#        runName = "test"
#        imageFile = ("norm001",)
        sys.exit("wormLearningManager.py [runName=(ImageFile)] ImageFiles")
    
    run_worm(imageFile,runName)

def run_worm(imageFile,runName):
    manager = learningManager()
    manager.initialize(imageFile,runName)
    manager.detectorFeatureVar = detectorFeatures()
    
    for image in imageFile:
        wormCreateLines(image,runName)
#        wormLinesSecStage(image)

    manager.preprocessImage()
    print "done Preprocessing\n"
    manager.prepareDataForJboost()
    print "done preparing data for JBoost"

#    manager.runJBoostNfold()
#    manager.plotScores()
    manager.runJBoostGenerateScorer()
    manager.renamePredictor()
    vectPredict(manager.procBaseDir+"matlab/calcScore_"+manager.runName+".m",runName)

#    manager.scoreImage(detectorFeatureVar)
#    manager.addOutLines(detectorFeatureVar)
    print 'done'

    print 'done'

if __name__ == "__main__":
    main()


