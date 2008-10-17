import sys, os, shutil,os.path
from stat import *
from xml.dom import minidom, Node
from math import atan2,pi



class DNXML:
    """ A class for performing the low-level processing of XML files from the Digital annotator
    """

    def __init__(self,filename,detectorFeatureVar):
        """reads a bix file and parses the graphical annotations (bfi) 
        """
        from xml.dom import minidom
        self.filename = filename
        xmldoc = minidom.parse(filename)
        self.xmldoc = xmldoc    
        self.bfiParsed = self.getBfi()
        self.updateObjectList()
        self.linelist={}                   # initialize internal dictionary of linelist (one list per class)

        self.minx = self.miny = 30
        self.maxx = detectorFeatureVar.imgSizeX-30
        self.maxy = detectorFeatureVar.imgSizeY-30
        
        self.detectorFeature = detectorFeatureVar
        for linetype in self.detectorFeature.labels.keys():
            self.linelist[linetype] = []
        
    def getBfi(self):             
        """ Finds graphics node and parses it
        Returns the parsed XML
        """
        parent = self.xmldoc.documentElement
        for node in parent.childNodes:        # search for the graphics node
            if node.nodeType == Node.ELEMENT_NODE and node.attributes['type'].value== 'graphics':
                valueNode = node.getElementsByTagName('value')[0] # parse graphics node
                return minidom.parseString(valueNode.childNodes[0].data)

    def updateObjectList(self):
        """ Updates the Object list
            does not return anything
        """
        gobjects = self.bfiParsed.documentElement.getElementsByTagName('gobject')
        objectList = {}
        for gobject in gobjects:
            type = gobject.attributes["name"].value
            if objectList.has_key(type):
                objectList[type].append(gobject)
            else:
                objectList[type] = [gobject]
        self.objectList = objectList

    def getVertices(self,gobject):
        """ return list of index,x,y  tuples corresponding to all of the vertices in gobject """
        vertices = gobject.getElementsByTagName('vertex')
        list = [];
        for vertex in vertices:
            x = float(vertex.attributes['x'].value)
            y = float(vertex.attributes['y'].value)
            i = int(vertex.attributes['index'].value)
            list.insert(i, (x, y))
        return list    



    def objectCounts(self):
        """count the number of each type of object
        Returns a list of (type,count) tuples
        """
        return ( (key,len(self.objectList[key])) for key in self.objectList.keys())

    def getObjectList(self,type):
        """ return the list of objects of given type
        """
        if self.objectList.has_key(type):
            return self.objectList[type]
        else:
            return []

    def updateBfiParsed(self):
        """ Updates the CDATA_SECTION_NODE part of the graphics node
            Used before writeXML
        """
        parent = self.xmldoc.documentElement
        newBfi = self.bfiParsed.toxml()
        for node in parent.childNodes:        # search for the graphics node
            if node.nodeType == Node.ELEMENT_NODE and node.attributes['type'].value== 'graphics':
                valueNode = node.getElementsByTagName('value')[0] 
                dataLen = len(valueNode.childNodes[0].data)
                valueNode.childNodes[0].deleteData(0,dataLen)
                valueNode.childNodes[0].appendData(newBfi)
    
    def findBoundaryBoxesPolygon(self,linetype):
        """ For all polygons in the bix file, creates lines orthogonal to each segment. 
        """
        self.updateObjectList()
        if not self.objectList.has_key('polygon'):
            return

        for polygons in self.objectList['polygon']:
            vList = self.getVertices(polygons)
            # find if polygon vertices are listed clockwise or counter-clockwise, 
            # this is done so that line segments always point out
            rotation=0;
            for segNo in range(len(vList)):
                (x1,y1) = vList[segNo]
                (x2,y2) = vList[segNo-1]
                (x3,y3) = vList[segNo-2]
                angle1=atan2((y1-y2),(x1-x2))
                angle2=atan2((y2-y3),(x2-x3))
                if(abs(angle1-angle2)>pi):
                    if (angle1<0):
                        angle1=angle1+2*pi
                    elif (angle2<0):
                        angle2=angle2+2*pi
                    else:
                        sys.exit('error in findBoundaryBoxesPolygon: angle1=%f, angle2=%f'%(angle1,angle2))                    
                rotation = rotation+angle1-angle2
            #print "rotation=%f\n" % (rotation/(2*pi))
            
            if(rotation>0):
                for segNo in range(len(vList)):
                    self.addOrthLines(vList[segNo-1],vList[segNo],5,linetype)
            else:
                for segNo in range(len(vList)):
                    self.addOrthLines(vList[segNo],vList[segNo-1],5,linetype)
 

    
    def removeEarlierLines(self,name=None):
        """ Removes lines that were added by previous runs of DigitalNotebook.py
        """
        if name == None:
            for gobject in self.bfiParsed.documentElement.getElementsByTagName('gobject'):
                gobject_name = gobject.attributes["name"].value
                if  self.linelist.has_key(gobject_name):
                    dump = self.bfiParsed.documentElement.removeChild(gobject)
        else:
            for gobject in self.bfiParsed.documentElement.getElementsByTagName('gobject'):
                if gobject.attributes["name"].value == name:
                    dump = self.bfiParsed.documentElement.removeChild(gobject)
            

    
    def addLinesToXML(self):
        """ add to the objects a set of lines
        Each line is added as gobject to bfiParsed
        """
        from exampleGobjects import exLine

        for linetype in self.linelist.keys():
            lines = self.linelist[linetype]
            for line in lines:
                
                lineObject = minidom.parseString(exLine).documentElement
                lineObject.setAttribute('name', linetype)
                
                # modify the line object
                for properties in lineObject.childNodes:
                # changing the color below
                    if properties.nodeType == Node.ELEMENT_NODE and properties.hasAttribute('name'): 
                        if properties.attributes['name'].value == 'color':
                            if(len(line)==5):
                                color = line[4]
                            else:
                                color = self.detectorFeature.lineColors[linetype]
                            properties.setAttribute('value',color)
                            # modify the line coordinates
                    if properties.nodeType == Node.ELEMENT_NODE and properties.nodeName == 'vertex':
                        index = int(properties.attributes['index'].value)
                        properties.setAttribute('x','%.2f' % line[index*2])
                        properties.setAttribute('y','%.2f' % line[index*2+1])
                            
                self.bfiParsed.documentElement.appendChild(lineObject)


    def addRandomLines(self,number,linelength,linetype):
        """Create randomly orientd line segments the range of x and y used by the
        line segments orthogonal to the polygones
        """
        from random import random
        from math import sin,cos,pi

        if not self.linelist.has_key(linetype):
            self.linelist[linetype]=[]

        for i in range(number):
            x = int(self.minx+random()*(self.maxx-self.minx))
            y = int(self.miny+random()*(self.maxy-self.miny))
            angle=random()*2*pi
            xline = cos(angle)*linelength/2
            yline = sin(angle)*linelength/2
            self.linelist[linetype].append((x-xline,y-yline,x+xline,y+yline))
    
    def addLines(self,vertex1,vertex2,linetype,color=""):
        """ Add line from vertex1 to vertex2        """

        if not self.linelist.has_key(linetype):
            self.linelist[linetype]=[]

        self.linelist[linetype].append((vertex1[0],vertex1[1],vertex2[0],vertex2[1],color))

 
    def addOrthLines(self,vertex1,vertex2,linelength,linetype):
        """Creates lines orthogonal to line joining vertex1 and vertex2 
           at each pixel length along the line
        """

        if not self.linelist.has_key(linetype):
            self.linelist[linetype]=[]

        (x1,y1) = vertex1
        (x2,y2) = vertex2

        segLength = int(pow((x1-x2)**2 + (y1-y2)**2,0.5))
        if segLength <=  0: return

        xstep = (x2-x1)/segLength
        ystep = (y2-y1)/segLength
        xline=ystep*linelength
        yline=-xstep*linelength
        (x,y) = (x1,y1)
        for i in range(1,segLength):
            x=x+xstep
            y=y+ystep
            self.linelist[linetype].append((x-xline,y-yline,x+xline,y+yline,))
                
    def writeXML(self, filename):
        """ write object into an XML file
        filename: the output file name
        """
        ff = open(filename,'w')
        self.updateBfiParsed()
        self.xmldoc.writexml(ff)
        ff.close

    def writeLines(self,filename):
        """ write list of box defning lines as a tab separated ascii file.
        filename: the output file name"""
        ff = open(filename,'w')
        labels = self.detectorFeature.labels
        weights = self.detectorFeature.lineWeights
        for linetype in self.linelist.keys():
            linelist = self.linelist[linetype]
            print "%s %d" % (linetype,len(linelist))
            for line in linelist:
                ff.write(",".join("%.1f" % a for a in line[0:4])+","+str(weights[linetype])+","+labels[linetype]+"\n")
        ff.close()
            
