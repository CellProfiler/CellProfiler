import yaml
import os
import sys

def trim_extension(filename):
    return os.path.splitext(filename)[0]

if len(sys.argv) < 2:
    print "sort|comp|comp2 file1 [file2]"
else:
    operation = sys.argv[1]

    if operation == "sort":
        file = sys.argv[2]
        loaded = yaml.load(open(file))
        yaml.dump(loaded, open(trim_extension(file) + "_sorted.yaml","w"), default_flow_style=False)
    elif operation == "comp" or operation == "comp2":
        fileA = sys.argv[2]
        fileB = sys.argv[3]

        loadedA = yaml.load(open(fileA))
        loadedB = yaml.load(open(fileB))

        def diff(a,b):
            for k,v in a.iteritems():
                if k in b:
                    bn = b[k]
                    if type(v) is not dict:
                        if v != bn:
                            print k,": ", v, " -> ", bn
                    else:
                        diff(v,bn)
                else:
                    print k, ": ", v, " -> NONE"
                
                

                
        diff(loadedA,loadedB)
        if operation == "comp2":
            diff(loadedB,loadedA)
    else:
        raise Exception("Wrong option")