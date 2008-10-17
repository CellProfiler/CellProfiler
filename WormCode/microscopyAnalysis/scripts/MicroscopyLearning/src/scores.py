#! /usr/bin/env python

''' Example Usage:
    scripts.py --info=data.test.boosting.info --iter=5,10,90 --out=data.test.scores.png
    For matlab plotting: After it runs, it creates 3  new files, matlabScoreScript.m, matlabScoreDataPos.m and matlabScoreDataNeg.m
    In matlab change the directory where these 3 files are and run matlabScoreScript. This script will create the image file as given by --out to the scripts.py
'''
import string
import getopt
import sys
import os
import re
#from pylab import *
#from matplotlib import rcParams
import random

SEPARATOR = ':'

class scores:
    def initialize(self):
        self.iter = None
        self.posScores=[]
        self.negScores=[]
        self.posBins = []
        self.negBins = []
        self.posVals = []
        self.negVals = []

def usage():
    print 'Usage: scores.py    '
    print '\t--info=filename  margins file as output by jboost (-a -2 switch)'
    print '\t--iter=i,j,k,...  the iterations to inspect, no spaces between commas'
    print '\t--out=output image file'


def equalBin(scores):

    if len(scores)<100:
        numBins = len(scores)/5
    else:
        numBins = 15

    binCenter = []
    binVals = []
    prev = 0;

    for ii in range(numBins):
        next = int(len(scores)*(ii+1)/numBins)-1
        binCenter.append( (scores[next]+scores[prev])/2)
        binVals.append(1/float(scores[next]-scores[prev])/float(len(scores)))
        prev = next
       
    return binCenter,binVals


def buildHist(all_scores,outFile):

    for curScores in all_scores:

#        print curScores.posScores[0], curScores.posScores[-1]
        curScores.posScores = [ss+random.uniform(-0.01,0.01) \
                for ss in curScores.posScores]
        curScores.negScores = [ss+random.uniform(-0.01,0.01) \
                for ss in curScores.negScores]
        curScores.posScores.sort()
        curScores.negScores.sort()

        curScores.posBins, curScores.posVals = equalBin(curScores.posScores)
        curScores.negBins, curScores.negVals = equalBin(curScores.negScores)

def drawMatplotlib(all_scores,outFile):
            
    hold(True)
    title('Score Distribution')
    xlabel('Scores')
    ylabel('Distribution')

    for curScores in all_scores:

        posColorStr = (random.random(),random.random(),random.random())
        negColorStr = (random.random(),random.random(),random.random())
        plot(curScores.posBins,curScores.posVals,
                label = 'Pos iter:%d' % (curScores.iter),color = posColorStr)
        plot(curScores.negBins,curScores.negVals,
                label = 'Neg iter:%d' % (curScores.iter),color = negColorStr)

    legend()
    savefig(outFile,format='png')


def drawMatlab(all_scores,outFile,matlab):
    matfile = open('matlabScoreScript.m',mode='w')
    scorefilePos = open('matlabScoreDataPos.m',mode='w')
    scorefileNeg = open('matlabScoreDataNeg.m',mode='w')

    matfile.writelines("h = figure;\n")
    matfile.writelines("hold('on')\n")
    matfile.writelines("title('Score Distribution')\n")

    for curScores in all_scores:
        for score in curScores.posBins:
            scorefilePos.writelines(' %f ' % score)
        scorefilePos.writelines('\n')
        for score in curScores.posVals:
            scorefilePos.writelines(' %f ' % score)
        scorefilePos.writelines('\n')

        for score in curScores.negBins:
            scorefileNeg.writelines(' %f ' % score)
        scorefileNeg.writelines('\n')
        for score in curScores.negVals:
            scorefileNeg.writelines(' %f ' % score)
        scorefileNeg.writelines('\n')

    matfile.writelines("posscores = load('matlabScoreDataPos.m');\n")
    matfile.writelines("negscores = load('matlabScoreDataNeg.m');\n")

    matlabCmd = '''
for i = 1:(size(posscores,1)/2)
    plot(posscores(2*i-1,:),posscores(2*i,:),'b');
end
for i = 1:(size(negscores,1)/2)
    plot(negscores(2*i-1,:),negscores(2*i,:),'r');
end
'''

    matfile.writelines(matlabCmd)
    matfile.writelines("saveas(h,'%s');\n" % (outFile))
    matfile.writelines("quit;\n")
    
    matfile.close()
    scorefileNeg.close()
    scorefilePos.close()

    sysCmd = '%s -nosplash -nojvm -nodesktop -r matlabScoreScript' % (matlab)
    if os.system(sysCmd):
        print 'Matlab run failed'

    os.remove('matlabScoreScript.m')
    os.remove('matlabScoreDataPos.m')
    os.remove('matlabScoreDataNeg.m')

def plotJboostScores(info,iters,outFile,matlab="matlab"):

    f = open(info,mode='r')
    lines = f.readlines()
    regEx = re.compile('^iteration=(\d*):\s*elements=(\d*):')
    all_scores = []

    curLineNo = 0
    while len(lines)>curLineNo:

        firstLine = lines[curLineNo]
        curLineNo = curLineNo+1
        if regEx.search(firstLine):
            matches = regEx.search(firstLine)
            iterNo,numElements =  map(int,matches.groups())
        else:
            print 'Info file is not properly formatted'
            sys.exit(2)

        if iterNo in iters:
            recordScores = True
            temp = scores()
            temp.initialize()
            all_scores.append(temp)
            all_scores[-1].iter = iterNo
        else:
            recordScores = False

        for elemNo in range(numElements):
            curLine = lines[curLineNo].strip()
            curLineNo = curLineNo+1
            if recordScores:
                curScoreSplit = curLine.split(SEPARATOR)
                if len(curScoreSplit) == 7:
                    curLabel = curScoreSplit[5]
                elif len(curScoreSplit) == 5:
                    curLabel = curScoreSplit[3]

                curElem = curScoreSplit[0]
                curMargin = curScoreSplit[1]
                curScore = curScoreSplit[2]

                if int(curLabel) == -1:
                    all_scores[-1].negScores.append(float(curScore))            
                elif int(curLabel) == 1:
                    all_scores[-1].posScores.append(float(curScore))            
                else:
                    print 'Formatting error'

    buildHist(all_scores,outFile)
    drawMatlab(all_scores,outFile,matlab)



if __name__ == "__main__":
    try:
        opts, args= getopt.getopt(sys.argv[1:], '', ['info=', 'iter=', 'out='])
    except getopt.GetoptError, inst:
        print 'Received an illegal argument:', inst
        usage()
        sys.exit(2)

    info = iter = outFile = None

    for opt,arg in opts:
        if (opt == '--info'):
            info = arg
        elif (opt == '--out'):
            outFile = arg
        elif (opt == '--iter'):
            iter = arg

    if(info == None or iter == None or outFile == None):
	    print 'Need info, iter and out file'
	    usage()
	    sys.exit(2)

    iters = map(int, [x.strip() for x in iter.split(',')])

    plotJboostScores(info,iters,outFile)

