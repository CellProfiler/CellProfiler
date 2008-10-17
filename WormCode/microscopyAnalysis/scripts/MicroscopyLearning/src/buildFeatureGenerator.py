import sys,shutil

def buildGenerateTrainingDataCaller(matlabScriptName, matlabScriptDir,matlabSourceDir,imagefilename,\
                                    imageLinesFileName,datafilename,detectorfeatures,runName):
    "create matlab script file: callGenerateTrainingData"
    
    comment="% callGenerateTrainingData calls  generateTrainingData in the fixed matlab scripts directory\n"+\
            "% This is the script that is called when python fires up matab to generate the training set \n\n"
            
    mscript = open(matlabScriptName,'w')

    mscript.write(comment)
    mscript.write("path('"+matlabScriptDir+"',path);   % point path back to this directory\n")
    mscript.write("cd "+matlabSourceDir+"\n")
    mscript.write("generateTrainingData('%s', '%s', '%s',%.2f);\n" %\
                   (imagefilename,imageLinesFileName, datafilename,detectorfeatures.size))
    mscript.write("quit\n")
    

def buildgetRotateImagesAndMask(matlabScriptDir,matlabSourceDir,detectorFeatures,runName):
    " create matlab function file: getRotatedImagesAndMask.m"
    comment = "% getRotatedImagesAndMask receives an image filename and outputs rotated versions of all of the pre-processed versions of this file\n"+\
              "% and the mask matrix that should be used for calculating this feature set\n\n"
    
    mask = detectorFeatures.mask
    imagefiles = detectorFeatures.imagefiles
    derivfiles = detectorFeatures.derivfiles
    angle_step = detectorFeatures.angle_step

    mscript = open(matlabScriptDir+'getRotatedImagesAndMask.m','w')

    mscript.write("function [I,D,mask] = getRotatedImagesAndMask(imagefilename)\n"+comment)
    
    mscript.write("    load(['%smasks/%s.mat']);\n\n" % (matlabSourceDir,mask))

    if (90 % angle_step) != 0:
        sys.exit('buildgetRotateImagesAndMask:    angle_step = %f must divide 90' % angle_step)
    mscript.write('    global as; as=%3d; %% angle step\n\n' % angle_step)
        
    mscript.write('    [I,D] = rotateAll({...\n')
    for ext in imagefiles:
        mscript.write("\t\t\t[imagefilename '"+ext+"'], ...\n")
    mscript.write('}, ...\n\t\t{...\n')
    for ext in derivfiles:
        mscript.write("\t\t\t[imagefilename '"+ext+"'], ...\n")
    mscript.write("});\n")

    mscript.write('\n%%% end %%%\n')
    mscript.close()
    shutil.copy(matlabScriptDir+'getRotatedImagesAndMask.m', matlabScriptDir+'getRotatedImagesAndMask_'+runName+'.m')
    
def buildCalcF(matlabScriptDir,detectorFeatures,runName):
    "create matlab function file: calcF.m"
    comment = "% calcF receives a set of boxes and calculates from it a feature vector\n\n"

    imagefiles = detectorFeatures.imagefiles
    derivfiles = detectorFeatures.derivfiles
    features = detectorFeatures.features

    calcFileName = matlabScriptDir+'calcF_'+runName+'.m'
    mscript = open(calcFileName,'w')

    mscript.writelines(("function f=calcF(Iboxes,Dboxes,mask)\n",
                        comment,
                        "    f=[];\n"))

    firstDboxUse = [True for e in derivfiles] # a list of flags to identify the first time that a Deriv box is used so that it is Normalized once
    for fp in features:
        dname = fp[1]

        if imagefiles.count(dname)>0:       # image is a BW image
            index=imagefiles.index(dname)
            if fp[2] == 'hist':
                mscript.write("\t"+"f=[f; histFeatures(["+",".join("%d" % x for x in fp[3])+"],"+\
                              "Iboxes{"+("%d" % (index+1))+"},mask(:,:,"+("%d" % fp[0])+"))];\n")
            elif fp[2] == 'perc':
                mscript.write("\t"+"f=[f; percFeatures(["+",".join("%d" % x for x in fp[3])+"],"+\
                              "Iboxes{"+("%d" % (index+1))+"},mask(:,:,"+("%d" % fp[0])+"))];\n")
            else:
                sys.exit('buildCalcF: BW feature function named '+fp[2]+' not recognized');
        elif derivfiles.count(dname)>0:       # image is a deriv image
            index=derivfiles.index(dname)
            if firstDboxUse[index]:
                firstDboxUse[index]=False
                mscript.write("\n\t"+"[magnitude,nmag] = edgeBoxMag(Dboxes{"+\
                              ("%d" % (index+1))+"}.mag);\n"+\
                              "\t"+"f = [f; magnitude];\n")
            if fp[2] == 'deriv':
                mscript.write("\t"+"f=[f; edgeFeatures("+("%d" % fp[3])+",Dboxes{"+("%d" % (index+1))+"}.angle,"+\
                                  "nmag,mask(:,:,"+("%d" % fp[0])+"))];\n")
            else:
                exit('buildCalcF: Deriv feature function named '+fp[2]+' not recognized');    
        else:
            sys.exit('buildCalcF: could not find image named '+ dname)

    mscript.write('\n%%% end %%%\n')
    mscript.close()
    shutil.copy(calcFileName, matlabScriptDir+'calcF.m')

    
def buildCalcFVect(matlabScriptDir,detectorFeatures,runName):
    "create matlab function file: calcFVect.m"
    comment = "% calcF receives a set of boxes and calculates from it a feature vector\n\n"

    imagefiles = detectorFeatures.imagefiles
    derivfiles = detectorFeatures.derivfiles
    features = detectorFeatures.features

    calcFileName = matlabScriptDir+'calcFVect'+runName+'.m'
    mscript = open(calcFileName,'w')

    mscript.writelines(("function f=calcFVect%s(Iboxes,Dboxes,mask)\n" % (runName),
                        comment,
                        "f = [];\n"))

    firstDboxUse = [True for e in derivfiles] # a list of flags to identify the first time that a Deriv box is used so that it is Normalized once
    for fp in features:
        dname = fp[1]

        if imagefiles.count(dname)>0:       # image is a BW image
            index=imagefiles.index(dname)
            if fp[2] == 'hist':
                mscript.write("\t"+"f=cat(3,f,"+\
                 "histFeaturesVect(Iboxes{%d}," % (index+1)+ \
                 "["+",".join("%d" % x for x in fp[3])+"],"+\
                 "mask(:,:,%d)" % fp[0]+"));\n")
            elif fp[2] == 'perc':
                mscript.write("\t"+"f=cat(3,f,"+\
                 "percFeaturesVect(Iboxes{%d}," % (index+1)+ \
                 "["+",".join("%d" % x for x in fp[3])+"],"+\
                 "mask(:,:,%d)" % fp[0]+"));\n")
            else:
                sys.exit('buildCalcF: BW feature function named '+fp[2]+' not recognized');
                
        elif derivfiles.count(dname)>0:       # image is a deriv image
            index=derivfiles.index(dname)
            if firstDboxUse[index]:
                firstDboxUse[index]=False
                mscript.write("\n\t"+"stdMag = edgeBoxMagVect(Dboxes{%d}.mag,mask);\n" % (index+1)+\
                              "\t"+"f = cat(3,f,stdMag);\n")
            if fp[2] == 'deriv':
                mscript.write("\t"+"f=cat(3,f,"+\
                    "edgeFeaturesVect(%d,Dboxes{%d}.angle,Dboxes{%d}.mag,stdMag,mask(:,:,%d)));\n" % \
                    (fp[3],index+1,index+1,fp[0]))
            else:
                exit('buildCalcFVect: Deriv feature function named '+fp[2]+' not recognized');    
        else:
            sys.exit('buildCalcFVect: could not find image named '+ dname)

    mscript.write('\n%%% end %%%\n')
    mscript.close()
    shutil.copy(calcFileName, matlabScriptDir+'calcFVect.m')
