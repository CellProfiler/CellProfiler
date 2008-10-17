from DigitalNotebook import *
from wormDetectorFeatures import detectorFeatures
from configLM import *
from wormDNGeom import wormDNGeom

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print 'usage: python wormDigitalNotebook.py imageName'
        sys.exit(-1)
    wormCreateLines(args[0],"test")
        
def wormCreateLines(imageName,runName):

    origBixFile = procBaseDir+"images/"+imageName +".tif.bix"
    newBixFile =  procBaseDir+"images/"+imageName +".tif.bixDN_"+runName
    shutil.copy(origBixFile,origBixFile+"."+runName)
    shutil.copy(origBixFile,newBixFile)

    dn = wormDNGeom(newBixFile,detectorFeatures())
    
    dn.addValidityLines()
    dn.addNamedLines("incorrect")
    dn.addNamedLines("correct")
    dn.markCrossedNegative()
    dn.removeEarlierLines("wormSeg")
    dn.removeEarlierLines("randomSeg")
    for lineLength in range(15,25,2):
        dn.addRandomLines(1000,lineLength,'randomSeg')
    dn.findBoxesWorm()
    dn.addLinesToXML()
    dn.writeXML(newBixFile)
    dn.writeLines(newBixFile+'.lines')

if __name__ == "__main__":
    main()