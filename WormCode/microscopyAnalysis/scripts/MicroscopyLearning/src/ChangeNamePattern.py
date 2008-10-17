""" 
change all the file names under a given directory
"""
import re,sys,os,os.path,string
    
dirname = '/Users/yoavfreund/Desktop/Galit_Lahav/'
if not os.path.isdir(dirname):
    sys.exit(dirname+' is not a directory')
    
p=re.compile('.*(Phase|YFP|Cherry).*(t\d+)(.tif|.stk)$',re.IGNORECASE)

for root, dirs, files in os.walk(dirname):
    for name in files:
        if p.match(name) != None:
            (type,time,ext)=p.findall(name)[0]
            newname = time+type+string.lower(ext)
            print name+" -> "+newname+"\n"
            os.rename(os.path.join(root,name),os.path.join(root,newname))
        
