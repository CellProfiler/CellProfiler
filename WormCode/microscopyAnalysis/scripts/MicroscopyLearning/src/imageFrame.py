from PIL import Image, ImageSequence
import os.path
import sys
'''Usage imageFrame.py imageName frameNumber'''

def main():
    args = sys.argv[1:]
    if len(args) !=2:
        print "Usage: imageFrame.py imageName frameNumber"
        sys.exit(2)

    imageFile = args[0]
    frameNumber = int(args[1])
    imDir,imName = os.path.split(imageFile)
    if imDir == "":
        imDir = "."
    imBase,imExt = os.path.splitext(imName)

    im = Image.open(imageFile)
    frame = ImageSequence.Iterator(im)[frameNumber]
    frame.save('%s/%s_%04d%s' %(imDir,imBase,frameNumber,imExt))

if __name__ == "__main__":
    main()
