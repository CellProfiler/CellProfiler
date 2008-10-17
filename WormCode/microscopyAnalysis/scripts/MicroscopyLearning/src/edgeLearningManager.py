import sys,os,os.path
from learningManager import learningManager
from edgeDetectorFeatures import detectorFeatures

"**** Note: Make sure that imageFile is always a list ***"

def main():
    args = sys.argv[1:]
    if len(args)>0:
        imageFile = args[0:]
    else:
        imageFile = ('t28',)

    manager = learningManager()
    manager.initialize(imageFile)
    detectorFeatureVar = detectorFeatures()
    manager.preprocessImage(detectorFeatureVar)
    print "done Preprocessing\n"
    manager.prepareDataForJboost(detectorFeatureVar)
    print "done preparing data for JBoost"
    manager.runJBoostNfold()
    manager.plotScores()
    manager.runJBoostGenerateScorer()
    manager.renamePredictor()
    manager.scoreImage(detectorFeatureVar)
    print 'done'

if __name__ == "__main__":
    main()

