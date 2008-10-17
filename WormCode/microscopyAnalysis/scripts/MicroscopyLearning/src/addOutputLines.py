import sys,os,os.path
from learningManager import learningManager
from wormDetectorFeatures import detectorFeatures


def main():
    args = sys.argv[1:]
    if len(args)>0:
        runName = args[0]
        imageFile = args[1:]
    else:
        sys.exit("No file given")
    run_worm(runName,imageFile)

def run_worm(runName,imageFile):
    manager = learningManager()
    manager.initialize(imageFile,runName)
    manager.detectorFeatureVar = detectorFeatures()
    manager.addOutLines()
    print 'done'

if __name__ == "__main__":
    main()
