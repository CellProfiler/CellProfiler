import re,sys,os,os.path

filename = 'TwikiPage.html'

infile = open(filename,'r')

imagelink_p = re.compile('<img src="([^"]*)" alt="([^"]*)"'+" width='([^']*)' height='([^']*)' />")

for line in infile:
    if len(imagelink_p.findall(line))==1:
        [(long,short,width,height)] = imagelink_p.findall(line)
        print "<a href='%s'><img src='%s' alt='%s' width='300' height='200' /></a>" % (long,long,short)
    else:
        print line[:-1]
        

        
    
