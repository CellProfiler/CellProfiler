import os,os.path


pwd = os.getcwd()
(sourceBaseDir,src) = os.path.split(pwd)
(sourceBaseDir,package) = os.path.split(sourceBaseDir)
(sourceBaseDir,scripts) = os.path.split(sourceBaseDir)

if not (src =='src' and scripts=='scripts'):
    exit('scripts mush be run from the python source directory, not from '+pwd)

matlabDir=sourceBaseDir+'/matlab/detector/'

matlab = '/Applications/MATLAB_R2008a/bin/matlab'
JBoostDir = '/Users/mkabra/jboost-1.4/'
procBaseDir = '/Users/mkabra/worms_run/'
matlabDir = sourceBaseDir+'/matlab/detector/'
hostname = 'Moksha'
DNTemplateLocation = None
DNTemplateFile = None

#if os.name == 'nt':
#    matlab = 'matlab'
#    DNTemplateLocation = None
#    JBoostDir = 'C:/Documents\ and\ Settings/mkabra/Desktop/jboost-1.4/'
#    procBaseDir = 'C:/Documents\ and\ Settings/mkabra/Desktop/worms_run/'
#    matlabDir = sourceBaseDir+'/matlab/detector/'
#    DNTemplateFile = sourceBaseDir+'/DigitalNotebook/MCF7.tmpl'
#    hostname = 'windows'
#
#else:
#    
#    hostname = os.uname()[1]
#    print "hostname = %s\n" % hostname
#    if hostname=='biospike.ucsd.edu' :
#        matlab='/usr/local/bin/matlab'
#        DNTemplateLocation = None
#        JBoostDir='/usr/local/jboost-1.4/'
#        procBaseDir = '/home/yfreund/process_images/galit/28/'
#        
#    elif hostname.find('Macintosh-110')>-1 or hostname.find('med.harvard.edu')>-1:   # Yoav's laptop
#        matlab='/Applications/MATLAB_R2007b/bin/matlab'
#        DNTemplateLocation='/Applications/DigitalNotebook 3/cix/gobject_templates.tmpl'
#        JBoostDir='/Users/yoavfreund/projects/jboost-1.4'
#        procBaseDir = '/Users/yoavfreund/Desktop/Galit_Lahav/Raw_tifs/28/'
#    
#    elif hostname=='gold':
#        matlab='/broad/tools/apps/matlab76/bin/matlab'
#        DNTemplateLocation = None    
#        JBoostDir = '/home/radon01/mkabra/jboost/jboost-1.4'
#        procBaseDir = '/imaging/analysis/People/Mayank/tmp/'
#    elif hostname.find('node')>-1:
#        matlab='/broad/tools/apps/matlab76/bin/matlab'
#        DNTemplateLocation = None    
#        JBoostDir = '/home/radon01/mkabra/jboost/jboost-1.4'
#        procBaseDir = '/imaging/analysis/People/Mayank/worms_run/'
#    elif hostname.find('gm0b2')>-1:
#        matlab ='/Applications/MATLAB_R2008a/bin/matlab'
#        DNTemplateLocation = None
#        JBoostDir = '/Users/mkabra/jboost-1.4'
#        procBaseDir = '/Users/mkabra/wormVect/'
    
JBoostWorkDirBase=procBaseDir+'jboost/'
imageDir=procBaseDir+'images/'

print "procBaseDir = %s\n" % procBaseDir

os.environ['CLASSPATH'] = JBoostDir+'/dist/jboost.jar:'+JBoostDir+'/lib/concurrent.jar'
