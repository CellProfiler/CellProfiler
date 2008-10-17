import getopt, sys
import glob, shutil
import os, os.path
import re, time

def usage():
    print '''Submits a number of jobs to lsf
    Inputs: -s "script": script to run
            -p script parameters: Script Parameter String -- optional 
            -d dir: Directory to execute in -- optional 
            -f files: Regexp of files on which to execute the script on
            -g groupname: jobs in lsf will be submitted to this group
            -l lsf parameters: extra parameters for lsf -- optional 
            -o log file name -- optional: default lsfJobs.log
    This script will create the directory dir if it does not exist.
    Inside this directory, for each file a new directory is created with "dir_" appended
    The jobs submitted are "bjobs -g groupname [lsf parameters] 'script [script parameters] filename'"
'''

class jobClass:
    pass

def main():
    try:
        opts,args = getopt.getopt(sys.argv[1:], "s:p:d:f:g:l:o:")
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    for o,a in opts:
        if o=='-s':
            script = a
        elif o=='-p':
            scriptParameter = a
        elif o=='-d':
            workDir = a
        elif o=='-f':
            fileRegex = a
        elif o=='-g':
            lsfGroup = a
        elif o=='-l':
            lsfParams = a
        elif o=="-o":
            logFile = a

    try :
        lsfParams
    except NameError:
        lsfParams = ""
    try:
        scriptParameter
    except NameError:
        scriptParameter = ""
    try:
        workDir
    except NameError:
        print "No working dir specified! Using current directory"
        workDir = "."
    try:
        lsfGroup
    except NameError:
        print "Need a group name"
        usage()
        sys.exit(2)
    else:
        lsfGroup = " -g "+lsfGroup+" "
    try: 
        script
    except NameError:
        print "No script specified"
        usage()
        sys.exit(2)
    try:
        fileRegex 
    except NameError:
        print "Need a file name"
        usage()
        sys.exit(2)
    try:
        logFile
    except NameError:
        logFile= "lsfJobs.log"

    workDir = os.path.abspath(workDir)
    if not os.path.isdir(workDir):
        os.makedirs(workDir,mode=0755)
        
    fLog = open(logFile,mode = 'w')
    startDir = os.getcwd()
    files = glob.glob(fileRegex)
    checkStatus = {}
    for file in files:
        job = jobClass()
        fileDir,fileName = os.path.split(file)
        fileBase,fileExt = os.path.splitext(fileName)

        job.fileName = fileName
        job.fileBase = fileBase
        job.fileExt = fileExt
        job.dirName = workDir+"/dir_"+fileBase
        if not os.path.isdir(job.dirName):
            os.mkdir(job.dirName)
        shutil.copy(file,job.dirName)
        os.chdir(job.dirName)
        
        job.jobName = fileBase
        job.outFile = fileBase+"_lsfout"
        job.errFile = fileBase+"_lsferr"

        try:
            os.remove(job.errFile)
        except OSError:
            pass
        try:
            os.remove(job.outFile)
        except OSError:
            pass

#create the lsf job cmd string
        lsfParams = lsfParams+" -oo "+job.outFile+" -eo "+job.errFile+" -J "+job.jobName
        job.lsfcmd = 'bsub '+lsfGroup+lsfParams+" ' "+script+" "+scriptParameter+" "+fileName+" ' "

        fcmd = os.popen(job.lsfcmd,'r')
        job.cmdout = fcmd.readlines()
        if fcmd.close():
            fLog.writelines("lsf submission of job failed for %s/%s\n" % (fileDir, fileName))

        fLog.writelines(job.cmdout)
        fLog.flush()

        checkStatus[file] = job
        os.chdir(startDir)




                #Now keep on checking the status.
    errPattern = re.compile('^Exited with exit code ')
    okPattern = re.compile('^Successfully completed.')

    while True:
        if not checkStatus.keys(): break

        for file,job in checkStatus.items():
            try:
                f = open(job.dirName+"/"+job.outFile,mode='r')
            except IOError:
                continue
                
            job.output = f.readlines()
            for line in job.output:

                if errPattern.search(line):
                    errCode = re.search('^Exited with exit code \d+',line).group()
                    fLog.writelines('LSF job for %s failed with error code %s\n' %(job.fileBase,errCode))
                    print 'Job for %s done' % job.fileBase
                    del checkStatus[file]

                if okPattern.search(line):
                    fLog.writelines('LSF job for %s completed succesfully\n' %(job.fileBase))
                    print 'Job for %s done' % file
                    del checkStatus[file]
                    os.remove(job.dirName+"/"+job.errFile)
                    os.remove(job.dirName+"/"+job.outFile)

            f.close()

        fLog.flush()
        time.sleep(1)

    fLog.close()


if __name__ == "__main__":
    main()
