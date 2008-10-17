import sys,os,os.path,re

def vectPredict(fileName,runName):
    
    matFile = open(fileName,'r')
    outMatFile = open(fileName.replace('.m','Vect.m'),'w')
    
    cmntRe = re.compile('(^\s*%.*$|^\s*$)')
    funcRe = re.compile('^\s*function')
    predInitRe = re.compile('^\s*pred = zeros')
    ifdefRe = re.compile('^\s*if def')
    ifvalRe = re.compile('^\s*if val')
    endRe = re.compile('^\s*end')
    elseRe = re.compile('^\s*else')
    predSumRe = re.compile('pred = pred +')
 
    printNextLine = False
 
    ifNo = 0
    end2 = False
    
    for line in matFile:
        
        line = line.lstrip()
        if(endRe.search(line)):
            if(end2):
                ifNo = ifNo-1
                end2 = False
            else:
                end2 = True
            continue
        if(ifdefRe.search(line)):
            continue
        if(printNextLine):
            for i in range(ifNo):
                outMatFile.writelines("  ")
            outMatFile.writelines("\t")
            outMatFile.writelines(line)
            printNextLine = False
            continue
        
        if(predSumRe.search(line)):
            modLine = line.replace('= pred + ','= pred + sel%d.*'%ifNo)
            for i in range(ifNo):
                outMatFile.writelines("  ")
            outMatFile.writelines(modLine)
            printNextLine = True
            continue
        
        if(ifvalRe.search(line)):
            ifNo = ifNo+1
            modLine = line.replace('if val(','tempSel%d = val(:,' %(ifNo))
            modLine = modLine.replace('\n',';\n')
            for i in range(ifNo):
                outMatFile.writelines("  ")
            outMatFile.writelines(""+modLine)
            for i in range(ifNo):
                outMatFile.writelines("  ")
            outMatFile.writelines("sel%d=sel%d&tempSel%d;\n" %(ifNo,ifNo-1,ifNo))
            continue
        
        if(elseRe.search(line)):
            for i in range(ifNo):
                outMatFile.writelines("  ")
            outMatFile.writelines("sel%d=sel%d&(~tempSel%d);\n" %(ifNo,ifNo-1,ifNo))
            continue
        
        if(cmntRe.search(line)):
            outMatFile.writelines(line)
            continue
        if(funcRe.search(line)):
            funcLine= line.replace('predict(','calcScore_'+runName+'Vect(')
            funcLine = funcLine.replace('calcScore(','calcScore_'+runName+'Vect(')
            funcLine = funcLine.replace('calcScore_%s(' %(runName),'calcScore_'+runName+'Vect(')
            funcLine = funcLine.replace(', def','')
            outMatFile.writelines(funcLine)
            continue
        if(predInitRe.search(line)):
            outMatFile.writelines('pred = zeros(size(val,1),1);\n')
            outMatFile.writelines('sel0=true(size(val,1),1);\n')
            continue
    
    
    matFile.close()
    outMatFile.close()
    
    

if __name__ == "__main__":
    runName = sys.argv[1]
    fileName = sys.argv[2]
    
    vectPredict(fileName,runName)