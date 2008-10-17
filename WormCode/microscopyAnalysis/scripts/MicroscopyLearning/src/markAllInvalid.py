import sys,os,re,shutil

def main():
    args = sys.argv[1:]
    if len(args)>0:
        xmlFile = args[0]
    else:
        sys.exit('No File Given')
    
    shutil.copy(xmlFile,xmlFile+".woValidity")
    
    inFile = open(xmlFile+".woValidity",'r')
    outFile = open(xmlFile,'w');
    
    wormSegRex = re.compile('<gobject type="polyline" name="wormSeg" >')
    validRex = re.compile('<tag value=".*correct" name="validity"/>')
    checkValidity = False
    
    for line in inFile:
        
        if(checkValidity):
            checkValidity = False
            if not validRex.search(line):
                outFile.writelines('        <tag value="incorrect" name="validity"/>\n')
        
        if(wormSegRex.search(line)):
            checkValidity = True
        else:
            checkValidity = False
            
        outFile.writelines(line)
    outFile.close()
    inFile.close()
    
if __name__ == "__main__":
    main()
