import sys,os,os.path
from learningManager import learningManager
from wormDetectorFeatures import detectorFeatures


def main():
    args = sys.argv[1:]
    if len(args)>0:
        imageFile = args[0:]
    else:
        sys.exit("Image File is not specified")
    prepareData(imageFile)

def prepareData(imageFile):
    manager = learningManager()
    manager.initialize(imageFile)
    manager.detectorFeatureVar = detectorFeatures()
    manager.preprocessImage()
    print "done Preprocessing\n"
    manager.prepareDataForJboost()
    print "done preparing data for JBoost"

if __name__ == "__main__":
    main()


