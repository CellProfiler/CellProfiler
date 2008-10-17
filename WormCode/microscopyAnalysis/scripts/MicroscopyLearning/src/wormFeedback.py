from DigitalNotebook import *
from wormDetectorFeatures import detectorFeatures
from wormLearningManager import run_worm
from configLM import *

if __name__ == "__main__":

    args = sys.argv[1:]
    if len(args) != 1:
        filename = 'norm1_fdbk1'
    else:
        filename = args[0]
    
    bixFilename = procBaseDir+"images/"+filename + ".tif.bix"
        
    dn = DNXML(bixFilename,detectorFeatures())
    shutil.copy(bixFilename, bixFilename+".fdbk")

    dn.removeInvalidLines()
    dn.findBoxesWorm()
#    for lineLength in range(5,25,2):
#        dn.addRandomLines(50,lineLength,'randomSeg')
    dn.addLinesToXML()
    dn.writeXML(bixFilename)
    dn.writeLines(bixFilename+'.lines')
    
    run_worm(filename)

