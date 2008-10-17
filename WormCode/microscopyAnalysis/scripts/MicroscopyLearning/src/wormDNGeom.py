import sys, os, shutil,os.path
from stat import *
from xml.dom import minidom, Node
from PIL import Image
from math import atan2,pi
from DigitalNotebook import DNXML

class wormDNGeom(DNXML):
    """ This class has worm specific geometric functions. This is to clean up DigitalNotebook.py
    """
    
    def addValidityLines(self):
        """ Adds line that have the validity tag.
        """
        for gobject in self.bfiParsed.documentElement.getElementsByTagName('gobject'):
            gobject_name = gobject.attributes["name"].value
            validityLine = False
            for properties in gobject.childNodes:
                if properties.nodeType == Node.ELEMENT_NODE and properties.hasAttribute('name'):
                    if properties.attributes['name'].value == 'validity':
                        validity = properties.attributes['value'].value
                        validityLine = True
                        
            if validityLine:
                for vertices in gobject.getElementsByTagName('vertex'):
                    if(vertices.attributes['index'].value=='0'):
                        x1 = float(vertices.attributes['x'].value)
                        y1 = float(vertices.attributes['y'].value)
                    elif(vertices.attributes['index'].value=='1'):
                        x2 = float(vertices.attributes['x'].value)
                        y2 = float(vertices.attributes['y'].value)
                    
                self.linelist[validity].append((x1,y1,x2,y2))
                dump = self.bfiParsed.documentElement.removeChild(gobject)
    
    def addNamedLines(self,name):
        """ Adds lines with name 
        """
        if not self.linelist.has_key(name):
            self.linelist[name]=[]

        for gobject in self.bfiParsed.documentElement.getElementsByTagName('gobject'):
            if gobject.attributes["name"].value == name:
                
                for vertices in gobject.getElementsByTagName('vertex'):
                    if(vertices.attributes['index'].value=='0'):
                        x1 = float(vertices.attributes['x'].value)
                        y1 = float(vertices.attributes['y'].value)
                    elif(vertices.attributes['index'].value=='1'):
                        x2 = float(vertices.attributes['x'].value)
                        y2 = float(vertices.attributes['y'].value)
                    
                self.linelist[name].append((x1,y1,x2,y2))
                dump = self.bfiParsed.documentElement.removeChild(gobject)
        
        
    def findBoxesWorm(self):
        """ For all the worms in the bix file, creates boxes of appropriate size    
        """
        self.updateObjectList()
        if not self.objectList.has_key('Worm'):
            return

        for worms in self.objectList['Worm']:
            for wormGobjects in worms.getElementsByTagName('gobject'):
                if wormGobjects.attributes['name'].value == 'outer edge 1':
                    vList1 = self.getVertices(wormGobjects)
                elif wormGobjects.attributes['name'].value == 'outer edge 2':
                    vList2 = self.getVertices(wormGobjects)
                elif wormGobjects.attributes['name'].value == 'median line':
                    vListm = self.getVertices(wormGobjects)
            self.addMedianBoxLines(vList1,vList2,vListm)



    def findBoundaryBoxesPolyline(self):
        """ For all polylines in the bix file, creates lines orthogonal to each segment. 
        """

        self.updateObjectList()
        if not self.objectList.has_key('polyline'):
            return

        for polygons in self.objectList['polyline']:
            vList = self.getVertices(polygons)
            for segNo in range(len(vList)-1):
                self.addOrthLines(vList[segNo],vList[segNo+1],5,boxSizes)
    

    def markCrossedNegative(self):
        """ all wormSegs that cross lines name markNegative are marked as negative
        """
        self.addNamedLines("markNegative")

        for gobject in self.bfiParsed.documentElement.getElementsByTagName('gobject'):
            if gobject.attributes["name"].value == "wormSeg":
                
                for vertices in gobject.getElementsByTagName('vertex'):
                    if(vertices.attributes['index'].value=='0'):
                        x1 = float(vertices.attributes['x'].value)
                        y1 = float(vertices.attributes['y'].value)
                    elif(vertices.attributes['index'].value=='1'):
                        x2 = float(vertices.attributes['x'].value)
                        y2 = float(vertices.attributes['y'].value)
                
                for negLine in self.linelist["markNegative"]:
                    v1 = (negLine[0],negLine[1])
                    v2 = (negLine[2],negLine[3])
                    u1 = (x1,y1)
                    u2 = (x2,y2)    
                    if self.doIntersect(v1,v2,u1,u2):
                        self.linelist["incorrect"].append((x1,y1,x2,y2))
                        break
                    
                dump = self.bfiParsed.documentElement.removeChild(gobject)
        
        del self.linelist["markNegative"]
        
    def addMedianBoxLines(self,vList1,vList2,vListm):
        """ Specialized methods for worms. Box length is defined by vlist1, vlist2 and vlistm
        """

        if not self.linelist.has_key('wormSeg'):
            self.linelist['wormSeg']=[]

        for segNo in range(len(vListm)-1):
            (x1,y1) = vListm[segNo]
            (x2,y2) = vListm[segNo+1]
            
            segLength = int(pow((x1-x2)**2 + (y1-y2)**2,0.5))
            if segLength <= 0: continue


            xstep = (x2-x1)/segLength
            ystep = (y2-y1)/segLength

            (x,y) = (x1,y1)
            for i in range(1,segLength):
                intersect1 = False
                
                lineSizes = range(0,25)
                lineSizes.extend(range(-1,-25,-1))
                for linelength in lineSizes:
                    
                    xline=ystep*linelength
                    yline=-xstep*linelength

                    if linelength<0:
                        nextStep = -1
                    else:
                        nextStep = 1
                        
                    xlineNext = ystep*(linelength+nextStep)
                    ylineNext = -xstep*(linelength+nextStep)
                    
                    v1 = (x+xline,y+yline)
                    v1Next = (x+xlineNext,y+ylineNext)

                    for i in range(len(vList1)-1):
                        intersectCur = self.doIntersect(v1,(x,y),vList1[i],vList1[i+1])
                        intersectNext = self.doIntersect(v1Next,(x,y),vList1[i],vList1[i+1])
                         
                        if intersectCur ^ intersectNext:
                            intersect1 = True
                            break
                        
                    if intersect1: break


                intersect2 = False
    
                lineSizes = range(0,25)
                lineSizes.extend(range(-1,-25,-1))
                
                for linelength in lineSizes:
                    xline=ystep*linelength
                    yline=-xstep*linelength

                    if linelength<0:
                        nextStep = -1
                    else:
                        nextStep = 1
                        
                    xlineNext = ystep*(linelength+nextStep)
                    ylineNext = -xstep*(linelength+nextStep)
                    
                    v2 = (x+xline,y+yline)
                    v2Next = (x+xlineNext,y+ylineNext)
                    
                    for i in range(len(vList2)-1):
                        intersectCur = self.doIntersect((x,y),v2,vList2[i],vList2[i+1])
                        intersectNext = self.doIntersect((x,y),v2Next,vList2[i],vList2[i+1])
                        
                        if intersectCur ^ intersectNext:
                            intersect2 = True
                            break
                        
                    if intersect2: break

                if intersect1 and intersect2:
                    self.linelist['wormSeg'].append((v1[0],v1[1],v2[0],v2[1]))
                    
                x=x+xstep
                y=y+ystep
                    

    def doIntersect(self,v1,v2,u1,u2):
        ''' Tells if segment v1,v2) intersects (u1,u2)
        '''
        (v1x,v1y) = v1
        (v2x,v2y) = v2
        (u1x,u1y) = u1
        (u2x,u2y) = u2

        cross1 = (v1y-u2y)*(u1x-u2x) - (u1y-u2y)*(v1x-u2x)
        cross2 = (v2y-u2y)*(u1x-u2x) - (u1y-u2y)*(v2x-u2x)
        cross3 = (u1y-v2y)*(v1x-v2x) - (v1y-v2y)*(u1x-v2x)
        cross4 = (u2y-v2y)*(v1x-v2x) - (v1y-v2y)*(u2x-v2x)

        if (cross1*cross2)<0 and (cross3*cross4)<0:
            return True
        else:
            return False
        
    def getLabel(self,vertex1,vertex2):
        """ If a line segment is close to a wormSeg then return 1
        """
        (x1,y1) = vertex1
        (x2,y2) = vertex2
        for seg in self.linelist["wormSeg"]:
            (a1,b1,a2,b2) = seg[0:4]
            if( ( (a1-x1)**2+(b1-y1)**2)< 8 and ( (a2-x2)**2+(b2-y2)**2)< 8): 
                return True
            elif( ( (a1-x2)**2+(b1-y2)**2)< 8 and ( (a2-x1)**2+(b2-y1)**2)< 8):
                return True
            
        return False
            
    def secStageGenLines(self,imagefilename):
        """ Generates lines for 2nd stage Matlab
        """
        self.removeEarlierLines("wormSeg")
        print "removed Worm Seg"
        self.removeEarlierLines("randomSeg")
        print "removed Random Seg"
        self.removeEarlierLines("incorrect")
        self.removeEarlierLines("correct")
        self.removeEarlierLines("markNegative")
        print "removed Mark Negative"
        self.findBoxesWorm()
        
        self.linelist["incorrect"] = []
        self.linelist["correct"] = []
        outFile = open(imagefilename+"OutLines",mode="r")
        allLines = outFile.readlines()
        
        count = 0
        for line in allLines:
            parts = line.split()
            x1 = float(parts[0])
            y1 = float(parts[1])
            x2 = float(parts[2])
            y2 = float(parts[3])
            
            if(self.getLabel((x1,y1),(x2,y2))):
                self.addLines((x1,y1),(x2,y2),'correct')
            else:
                self.addLines((x1,y1),(x2,y2), "incorrect")
            
            if(count%5000 == 0):
                print count
            count = count+1