def writeSpecFile(specFileName,detectorFeatureVar):

    mask=detectorFeatureVar.mask
    size=detectorFeatureVar.size
    angle_step=detectorFeatureVar.angle_step
    imagefiles=detectorFeatureVar.imagefiles
    derivfiles=detectorFeatureVar.derivfiles
    features=detectorFeatureVar.features
    labels  =detectorFeatureVar.labels

    specfile = open(specFileName,'w')
    specfile.write("""exampleTerminator=;
attributeTerminator=,
maxBadExa=0 

""")

    firstDboxUse = [True for e in derivfiles] # a list of flags to identify the first time that a Deriv box is used so that it is Normalized once

    for fp in features:
        if(fp[2]=='hist'):
            p = fp[3]
            for i in range(len(p)-1):
                specfile.write(fp[1]+("M%d_" % fp[0])+("b%d_%d" % (p[i],p[i+1]))+"\tnumber\n")
        elif(fp[2]=='perc'):
            p = fp[3]
            for i in range(len(p)):
                specfile.write(fp[1]+("M%d_" % fp[0])+("%d" % p[i])+"\tnumber\n")
        elif(fp[2]=='deriv'):
            index=derivfiles.index(fp[1])
            if firstDboxUse[index]:
                firstDboxUse[index]=False
                specfile.write(fp[1]+"_mag\tnumber\n")
            for i in range(fp[3]):
                specfile.write(fp[1]+("M%d" % fp[0])+("d%d" % i)+"\tnumber\n")
    
    specfile.write("weight\tnumber\n")                       
    specfile.write("labels\t("+",".join(labels.values())+")")
    specfile.close()

