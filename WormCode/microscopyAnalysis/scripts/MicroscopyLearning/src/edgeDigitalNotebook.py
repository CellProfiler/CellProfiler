from DigitalNotebook import *
from edgeDetectorFeatures import detectorFeatures

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        filename = "/Users/yoavfreund/Desktop/Galit_Lahav/Raw_tifs/28/B2_w1Phase_-_camera_s2_t28.TIF.bix"
    elif len(args) == 1:
        filename = args[0]
    else:
       print 'usage: python edgeDigitalNotebook.py inFile.xml'
       sys.exit(-1)

    dn = DNXML(filename,detectorFeatures())
    shutil.copy(filename, filename+".old")
    dn.removeEarlierLines()
    dn.findBoundaryBoxesPolygon('cellEdge')
    dn.addRandomLines(895,5,'randomSeg')
    dn.addLinesToXML()
    dn.writeXML(filename)
    dn.writeLines(filename+'.lines')


