import sys,os,os.path,re,math

#filename = '/Users/yoavfreund/Desktop/Galit_Lahav/Raw_tifs/28/jboost/t28.0/ADD_ALL/trial1.output.tree'

args = sys.argv[1:]
if len(args) != 1:
    sys.exit('Usage: ParseTree <name>.output.tree')
    
filename = args[0]

infile = open(filename,'r')

overall_p = re.compile('(\d+)\s+\[(R[\d\.\:]*)\] (Splitter|prediction) = (.*)')
node_p = re.compile('(:(\d+)|\.(\d+))')
splitter_p = re.compile('InequalitySplitter. (\S+ < \d+\.\d{0,2})')
rule_p = re.compile('(\S*)\.tif')
prediction_p = re.compile('BinaryPrediction. p\(1\)= (-?\d+\.\d{0,4})')
which_predictor_p = re.compile(':(\d+)$')
count=0
list={}
contribution={}
for line in infile:
    #print line
    if overall_p.match(line)== None:
        continue
    [(iter,node,ps,content)] = overall_p.findall(line)
    iter = int(iter)
    #print "iter='%d' node='%s' ps='%s' content='%s'\n" % (iter,node,ps,content)
    if ps=='Splitter':
        [rule] = splitter_p.findall(content)
        list[iter] = {'node':node,'rule':rule}
        [image] = rule_p.findall(rule)
        strength = math.exp(-iter/50.0)
        if contribution.has_key(image):
            contribution[image] +=  strength;
        else:
            contribution[image] =  strength;
    elif ps=='prediction':
        [prediction] = prediction_p.findall(content)
        if(iter==0):
            list[iter] = {'node':node,'prediction':prediction}
        else:
            [which_pred] = which_predictor_p.findall(node)
            list[iter][which_pred]=prediction
    else:
        sys.exit('bad line:\n'+line)
    count=count+1
    
print "weak rules, ordered by boosting iteration number:"
for (iter,item) in list.items():
    if iter==0:
        print "root. Prediction = %s" % item['prediction']
    else:
        print "%d: %s if (%s) %s:%s" % (iter,item['node'],item['rule'],item['0'],item['1'])
    
print "\n\nContributions of different pre-preocessed images:\n"

c = contribution.items()
c.sort(cmp = lambda x,y: -cmp(x[1],y[1]))
print "\n".join(("%s\t%.3f") % x for x in c)
    
    
    
