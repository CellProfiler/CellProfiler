""" 
given a directory name, this scripts walks the directory under this directory 
name and changes the space in each name to an underscore
"""
import sys,os,os.path

args = sys.argv[1:]
if len(args) != 1:
    sys.exit('Usage: replaceSpacesInFilenames <dir>')
    
dirname = args[0]
if not os.path.isdir(dirname):
    sys.exit(dirname+' is not a directory')
    
for root, dirs, files in os.walk(dirname):
    for name in files+dirs:
        newname = name.replace(' ','_')
        os.rename(os.path.join(root,name),os.path.join(root,newname))
        
