from DigitalNotebook import *
from wormDetectorFeatures import detectorFeatures
from configLM import *
from wormDNGeom import wormDNGeom

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print 'usage: python wormDigitalNotebookStage2.py imageName'
        sys.exit(-1)
    wormLinesSecStage(args[0])
        
def wormLinesSecStage(imageName):
    filename = procBaseDir+"images/"+imageName +".tif.bix"
    dn = wormDNGeom(filename,detectorFeatures())
    
    dn.secStageGenLines(procBaseDir+"images/"+imageName)
    dn.writeLines(filename+'.lines2Stage')

if __name__ == "__main__":
    main()