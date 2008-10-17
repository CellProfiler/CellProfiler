import sys,os,os.path,shutil
from vectorizePredictor import vectPredict

class learningManager:
    """ A class that contains methods that manage the Digital Notebook, the matlab scripts, jboost and the various files
    that constitute the active learning process for object detection in microscopy image analysis.
    """

    def initialize(self,imageFile,runName =None):
        """
        Define location of relevant files and directories.
        matlabDir: the directory where the matlab scripts reside
        matlab: the command for starting up matlab
        imageDir: the directory where the image files and .bix (XML) files reside
        DNTemplateLocation: the filename from which DigitalNotebook expects to read it's template file
        DNTemplateFile: the template file that this manager wants Digital Notebook to use.
        """

        from configLM import *

        self.imageFile = imageFile
        
# runName is useful when learning using many images. (mainly for worms)
# defaults to imageFile when no runName is given.

        if runName ==None:
            self.runName = imageFile
        else:
            self.runName = runName
        
        if not os.path.isdir(procBaseDir+"/jboost"):
            os.mkdir(procBaseDir+"/jboost")
        if not os.path.isdir(procBaseDir+"/matlab"):
            os.mkdir(procBaseDir+"/matlab")

        self.procBaseDir = procBaseDir
        self.sourceBaseDir = sourceBaseDir
        self.matlabDir = matlabDir
        self.JBoostDir = JBoostDir
        self.JBoostWorkDirBase=JBoostWorkDirBase
        self.imageDir = imageDir

        self.DNTemplateFile = DNTemplateFile
        self.hostname = hostname
        self.matlab = matlab
        if DNTemplateLocation:
            self.DNTemplateLocation = os.path.abspath(DNTemplateLocation)
        else:
            self.DNTemplateLocation = None

       # Set the file cix/gobject_templates.tmpl and/pr .cix files
        # to the configuration for the specific type of annotation
        import shutil

        if self.DNTemplateLocation != None:
            shutil.copyfile(DNTemplateLocation, DNTemplateLocation+'.bck') # back up old template file
            shutil.copyfile(DNTemplateFile, DNTemplateLocation) # copy desired template file to where DN will look for it.

    def generateBoxes():
        """ Wait until bix file has changed and then generate line-for-box annotations based on the the
        annotations added by the user
        """

    def createBoxList():
        """translate annotation file into an ascii file with two list of boxes: positive examples and negative examples
        this will be the input file for matlab
        """

    def preprocessImage(self):
        """ run matlab pre-processing steps on an image: edge detection, laplacian of gaussians etc.
        """
        
        detectorFeatureVar = self.detectorFeatureVar
        for curImage in self.imageFile:
            imageFilePath = self.imageDir+curImage
    
            scriptname = 'preProcessing'+curImage
            script = "filename = '"+imageFilePath+"';\n";
            
            generateSomething = False
            for (inext,outext,function) in detectorFeatureVar.imageProcessing+detectorFeatureVar.derivProcessing:
                if function.find('(')==-1:
                    sys.exit('learningManager.preprocessImage: preProcessing function "'+function+'" does not include "("')
                elif detectorFeatureVar.PreProcFunctions.count(function[:function.find('(')]) == 0:
                    sys.exit('learningManager.preprocessImage: preProcessing function "'+function+'" does not appear in PreProcFucntions')
                elif os.path.isfile(imageFilePath+outext):
                    print "earningManager.preprocessImage: file "+imageFilePath+outext+" already exists, not generating it\n";
                else:
                    script = script+("%s[filename '%s'],[filename '%s']);\n" % (function,inext,outext))
                    generateSomething = True
                           
                           
            self.createMatlabRunFile(self.procBaseDir+'matlab/'+scriptname+'.m',script)
            if generateSomething:
                outputlines = self.runMatlab(self.procBaseDir+'matlab/',scriptname)
        
    def prepareDataForJboost(self):
        """
        create directory for jboost
        Prepare Run matlab script to create training set for jboost.
        create spec file
        """
        dirname = self.runName
        runName = self.runName
        imageDir = self.imageDir
        detectorFeatureVar = self.detectorFeatureVar
        
        i=0
        while(os.listdir(self.JBoostWorkDirBase).count(dirname+(".%d" % i))>0):
            i=i+1
        self.JBoostWorkDir = self.JBoostWorkDirBase+dirname+(".%d" % i)+"/"
        os.mkdir(self.JBoostWorkDir)
        
        from buildFeatureGenerator import buildCalcF,buildCalcFVect,\
            buildgetRotateImagesAndMask,buildGenerateTrainingDataCaller
        
        if os.path.exists(self.JBoostWorkDir+self.runName+".data"):
            os.remove(self.JBoostWorkDir+self.runName+".data")
        #remove earlier data file.

        buildCalcF(self.procBaseDir+"matlab/",detectorFeatureVar,self.runName)
        buildCalcFVect(self.procBaseDir+"matlab/",detectorFeatureVar,self.runName)
        buildgetRotateImagesAndMask(self.procBaseDir+"matlab/",self.matlabDir,\
                                    detectorFeatureVar,self.runName)
        
        for curImage in self.imageFile:
            
            matlabScriptDir =self.procBaseDir+"matlab/" 
            matlabSourceDir = self.matlabDir
            imageFileName = self.procBaseDir+"images/"+curImage
            imageLinesFileName = self.procBaseDir+"images/"+curImage + ".tif.bixDN_"+runName+".lines"
            dataFileName = self.JBoostWorkDir+self.runName+".data"
            
            matlabScriptName = matlabScriptDir+'callGenerateTrainingData_'+curImage+"_"+runName+'.m'
            buildGenerateTrainingDataCaller(matlabScriptName, matlabScriptDir, matlabSourceDir\
                                            ,imageFileName,imageLinesFileName\
                                            ,dataFileName,detectorFeatureVar,self.runName)
            outputlines = self.runMatlab(self.procBaseDir+'matlab/',\
                                         'callGenerateTrainingData_'+curImage+"_"+runName)
            
        from writeSpecFile import writeSpecFile
        writeSpecFile(self.JBoostWorkDir+dirname+".spec",detectorFeatureVar)

    def setJBoostWorkDir(self):
        """ if JBoostWorkDir not set, set it to be the highest indexed directory (well... not exactly ... fix this)"""
        try:
            self.JBoostWorkDir
        except AttributeError:
            dirname = self.runName
            i=0
            while(os.listdir(self.JBoostWorkDirBase).count(dirname+(".%d" % i))>0):
                i=i+1
            self.JBoostWorkDir = self.JBoostWorkDirBase+dirname+(".%d" % (i-1))+"/"
        
            
    def runJBoostNfold(self):
        """ run jboost using N fold cross validation, assigning equal total weight to
        positives and to negatives,
        """
        self.setJBoostWorkDir()
        filename = self.runName
        command = self.JBoostDir+"/scripts/nfold.py --folds=3 --data="+self.JBoostWorkDir+filename+".data"\
                  +" --spec="+self.JBoostWorkDir+filename+".spec"\
                  +" --round=250 --tree=ADD_ALL --generate --dir="+self.JBoostWorkDir+"nfold"\
                  +" --booster=LogLossBoost"
        print "jboost command=|"+command+"|\n"
        status = os.system(command) 
        if status != 0:
            print "Jboost run failed"
            sys.exit(2)
        
    def runJBoostGenerateScorer(self):
        """ Run jboost on the whole training set to generate the scoring function
        """
        self.setJBoostWorkDir()
        filename = self.runName
        command = "java -Xmx500M jboost.controller.Controller -b LogLossBoost -numRounds 250"\
                  +" -S "+self.JBoostWorkDir+filename\
                  +" -n "+self.JBoostWorkDir+filename+".spec"\
                  +" -t "+self.JBoostWorkDir+filename+".data"\
                  +" -T "+self.JBoostWorkDir+filename+".data"\
                  +" -m "+self.procBaseDir+"matlab/calcScore_"+self.runName+".m"
        
        print "jboost command=|"+command+"|\n"
        status = os.system(command) 
        if status != 0:
            print "JBoost run failed"
            sys.exit(2)
            
    def renamePredictor(self):
        """ Rename the function inside calcScore.m from predict to calcScore """
        filename=self.procBaseDir+"matlab/calcScore_"+self.runName+".m"
        if not os.path.isfile(filename):
            exit("did not find "+filename)
    
        infile=open(filename)
        text = infile.read()
        infile.close()
        text = text.replace('predict(','calcScore_'+self.runName+'(')
        outfile=open(filename,"w")
        outfile.write(text)
        outfile.close()
    
 

    def plotScores(self):
        from scores import *
        ''' Score plots'''
        filename = self.runName
        for folds in range(3):
            plotJboostScores(\
        "%snfold/ADD_ALL/trial%d.test.boosting.info"% (self.JBoostWorkDir,folds),\
         (90,) , "%s%s_test_%d.png" %(self.JBoostWorkDir,filename,folds),\
         self.matlab)
            plotJboostScores(\
        "%snfold/ADD_ALL/trial%d.train.boosting.info"% (self.JBoostWorkDir,folds),\
         (90,) , "%s%s_train_%d.png" %(self.JBoostWorkDir,filename,folds),\
         self.matlab)
            
    def runMatlab(self,dirname,scriptname):
        """ run matlab as a unix command
        commands: the list of command lines for matlab
        Returns the output from the matlab commands
        """
        logfile = scriptname+".log"
        cwd=os.getcwd()
        os.chdir(dirname)               # yoav: matlab is not happy with running a script not in the current directory
        command=self.matlab+' -logfile '+logfile+' -nojvm -nosplash -nodisplay -r '+scriptname
        print "matlab command=|"+command+"|\n"
        print "current directory is:"+os.getcwd()+"\n"
        self.matlabStatus = os.system(command) 
        os.chdir(cwd)                   # switch back to original working directory
        if self.matlabStatus != 0:
            print "Matlab terminated with error: check logfile "+ dirname+logfile
            sys.exit(1)

    def createMatlabRunFile(self,scriptname,commands):
        fi = open(scriptname,mode='w')
        print self.matlabDir
        fi.write('cd '+self.matlabDir+'\n')
        fi.writelines(commands)
        fi.write('quit\n')
        fi.close()
        

    def scoreImage(self):
        """ run matlab detector on an image and generate detection map
        This is an initial scorer, more elaborate scorers should, at least initially, be placed
        in sub-classes (for edge, CElegans etc.)
        """

        matlabScriptDir = self.procBaseDir+"matlab/"
        matlabSourceDir = self.matlabDir
        detectorFeatureVar = self.detectorFeatureVar
        for curImage in self.imageFile:
            imagefilename = self.procBaseDir+"images/"+curImage
            
            size = detectorFeatureVar.size
            angleStep = detectorFeatureVar.angle_step
            numQuadrants = detectorFeatureVar.numQuadrants
            stepSize = detectorFeatureVar.classifyStep
            
            scriptname = 'callScoreBoxes'+curImage+"_"+self.runName
            mscript = open(matlabScriptDir+scriptname+'.m','w')
            
            mscript.write("'starting callScoreBoxes'\n")
            mscript.write("path('"+matlabScriptDir+"',path);   % point path back to this directory\n")
            mscript.write("cd "+matlabSourceDir+"\n")
            for i in detectorFeatureVar.boxSize:
    
                mscript.write("scores{%d} = scoreBoxesOneSize('%s',%d,%d,%d,%d);\n" %\
                          (i,imagefilename,i,angleStep,stepSize,numQuadrants))
                mscript.write("save "+matlabScriptDir+curImage+"_"+runName+"_"+"Scores.mat\n")
    
            try :
                detectorFeatureVar.matlabPostProcessing
                mscript.write("%s(scores,'%s_%s_OutLines',%d,%.2f);\n" % \
                    (detectorFeatureVar.matlabPostProcessing,imagefilename,runName,\
                     detectorFeatureVar.classifyStep,detectorFeatureVar.size))
            except NameError:
                pass
    
            mscript.write("quit\n")
            
            mscript.close()
            
            outputlines = self.runMatlab(self.procBaseDir+"matlab/",scriptname)
    

    def scoreImageVect(self):
        """ run matlab detector on an image and generate detection map
        This is an initial scorer, more elaborate scorers should, at least initially, be placed
        in sub-classes (for edge, CElegans etc.)
        """


        matlabScriptDir = self.procBaseDir+"matlab/"
        matlabSourceDir = self.matlabDir
        detectorFeatureVar = self.detectorFeatureVar
        for curImage in self.imageFile:
            imagefilename = self.procBaseDir+"images/"+curImage
            
            size = detectorFeatureVar.size
            angleStep = detectorFeatureVar.angle_step
            numQuadrants = detectorFeatureVar.numQuadrants
            stepSize = detectorFeatureVar.classifyStep
            
            scriptname = 'callScoreBoxes'+curImage+"_"+self.runName
            mscript = open(matlabScriptDir+scriptname+'.m','w')
            
            mscript.write("fprintf('starting callScoreBoxes\\n');\n")
            mscript.write("path('"+matlabScriptDir+"',path);   % point path back to this directory\n")
            mscript.write("cd "+matlabSourceDir+"\n")
            mscript.write("fprintf('Rotating Images..\\n');\n")
            mscript.write("\t[I,D,mask]=getRotatedImagesAndMask('%s');\n" % (imagefilename))
            
            for i in detectorFeatureVar.boxSize:
    
                mscript.write("\tscores{%d} = calculateScoresVect(I,D,mask,%d,%d,%d,%d,'%s');\n" %\
                              (i,i,stepSize,angleStep,numQuadrants,self.runName))
                mscript.write("\tsave "+matlabScriptDir+curImage+"_"+self.runName+"_"+"Scores.mat\n")
    
            try :
                detectorFeatureVar.matlabPostProcessing
                mscript.write("%s(scores,'%s_%s_OutLines',%d,%.2f);\n" % \
                    (detectorFeatureVar.matlabPostProcessing,imagefilename,runName,\
                     detectorFeatureVar.classifyStep,detectorFeatureVar.size))
            except NameError:
                pass
    
            mscript.write("quit\n")
            
            mscript.close()
            
            outputlines = self.runMatlab(self.procBaseDir+"matlab/",scriptname)
    


    def addOutLines(self):
        """ Add's lines that are generated by matlab post processing to the 
            .bix file
        """
        from DigitalNotebook import *

        detectorFeatureVar = self.detectorFeatureVar
        for curImage in self.imageFile:
            imagefilename = self.procBaseDir+"images/"+curImage
            bixFile = self.procBaseDir+"images/"+curImage+".tif.bix."+self.runName
            newBixFile = self.procBaseDir+"images/"+curImage+".tif.bix.LM_"+self.runName
    
            if not os.path.isfile(bixFile):
                print bixFile+" Does not exist"
                print "Not adding lines to bix file"
    
            shutil.copy(bixFile,newBixFile)
            dn = DNXML(newBixFile,detectorFeatureVar)
    
            outFile = open(imagefilename+"_"+self.runName+"_OutLines",mode='r')
            for line in outFile:
                parts = line.split()
                x1 = float(parts[0])
                y1 = float(parts[1])
                x2 = float(parts[2])
                y2 = float(parts[3])
                curColor = parts[4]
                dn.addLines((x1,y1),(x2,y2),'wormSeg',color=curColor)
            
            dn.addLinesToXML()
            dn.writeXML(newBixFile)


    def getUserFeedback():
        """ run digital notebook to view detections and get the user to mark correct and incorrect ones.
        Second development phase.
        """
        

