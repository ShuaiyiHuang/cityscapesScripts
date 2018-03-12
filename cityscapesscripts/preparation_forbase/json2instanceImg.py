#!/usr/bin/python
#
# Reads labels as polygons in JSON format and converts them to instance images,
# where each pixel has an ID that represents the ground truth class and the
# individual instance of that class.
#
# The pixel values encode both, class and the individual instance.
# The integer part of a division by 1000 of each ID provides the class ID,
# as described in labels.py. The remainder is the instance ID. If a certain
# annotation describes multiple instances, then the pixels have the regular
# ID of that class.
#
# Example:
# Let's say your labels.py assigns the ID 26 to the class 'car'.
# Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .
# A group of cars, where our annotators could not identify the individual
# instances anymore, is assigned to the ID 26.
#
# Note that not all classes distinguish instances (see labels.py for a full list).
# The classes without instance annotations are always directly encoded with
# their regular ID, e.g. 11 for 'building'.
#
# Usage: json2instanceImg.py [OPTIONS] <input json> <output image>
# Options:
#   -h   print a little help text
#   -t   use train IDs
#
# Can also be used by including as a module.
#
# Uses the mapping defined in 'labels.py'.
#
# See also createTrainIdInstanceImgs.py to apply the mapping to all annotations in Cityscapes.
#

# python imports
import os, sys, getopt
from basic_utils import Cropping
import matplotlib.pyplot as plt
import numpy as np

# Image processing
# Check if PIL is actually Pillow as expected
try:
    from PIL import PILLOW_VERSION
except:
    print("Please install the module 'Pillow' for image processing, e.g.")
    print("pip install pillow")
    sys.exit(-1)

try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)


# cityscapes imports
sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
from annotation import Annotation
from labels     import labels, name2label

# Print the information
def printHelp():
    print('{} [OPTIONS] inputJson outputImg'.format(os.path.basename(sys.argv[0])))
    print('')
    print(' Reads labels as polygons in JSON format and converts them to instance images,')
    print(' where each pixel has an ID that represents the ground truth class and the')
    print(' individual instance of that class.')
    print('')
    print(' The pixel values encode both, class and the individual instance.')
    print(' The integer part of a division by 1000 of each ID provides the class ID,')
    print(' as described in labels.py. The remainder is the instance ID. If a certain')
    print(' annotation describes multiple instances, then the pixels have the regular')
    print(' ID of that class.')
    print('')
    print(' Example:')
    print(' Let\'s say your labels.py assigns the ID 26 to the class "car".')
    print(' Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .')
    print(' A group of cars, where our annotators could not identify the individual')
    print(' instances anymore, is assigned to the ID 26.')
    print('')
    print(' Note that not all classes distinguish instances (see labels.py for a full list).')
    print(' The classes without instance annotations are always directly encoded with')
    print(' their regular ID, e.g. 11 for "building".')
    print('')
    print('Options:')
    print(' -h                 Print this help')
    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')

# Print an error message and quit
def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    printHelp()
    sys.exit(-1)

# Convert the given annotation to a label image

def getInstancewithLabel(annotation, encoding,labelfilter='car'):
    # the size of the image
    size = ( annotation.imgWidth , annotation.imgHeight )
    mysize=1.0*annotation.imgWidth*annotation.imgHeight

    # the background
    if encoding == "ids":
        backgroundId = name2label['unlabeled'].id
    elif encoding == "trainIds":
        backgroundId = name2label['unlabeled'].trainId
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create


    # a drawer to draw into the image


    # a dict where we keep track of the number of instances that
    # we already saw of each class
    nbInstances = {}
    for labelTuple in labels:
        if labelTuple.hasInstances:
            nbInstances[labelTuple.name] = 0
    instanceImg_arr=[]
    Ids_arr=[]
    Sizes_arr=[]
    # loop over all objects
    for obj in annotation.objects:
        label   = obj.label
        polygon = obj.polygon

        #if the object does not belong to filer label, skip it
        if obj.label!=labelfilter:
            continue

        # If the object is deleted, skip it
        if obj.deleted:
            continue

        instanceImg = Image.new("I", size, 0)
        drawer = ImageDraw.Draw(instanceImg)
        # if the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        # also we know that this polygon describes a group
        isGroup = False
        if ( not label in name2label ) and label.endswith('group'):
            label = label[:-len('group')]
            isGroup = True

        if not label in name2label:
            printError( "Label '{}' not known.".format(label) )

        # the label tuple
        labelTuple = name2label[label]

        # get the class ID
        if encoding == "ids":
            id = labelTuple.id
        elif encoding == "trainIds":
            id = labelTuple.trainId

        # if this label distinguishs between individual instances,
        # make the id a instance ID
        if labelTuple.hasInstances and not isGroup and id != 255:
            id = id * 1000 + nbInstances[label]



        # If the ID is negative that polygon should not be drawn
        if id < 0:
            continue



        try:
            drawer.polygon( polygon, fill=255 )
            instanceImgnp=np.asarray(instanceImg)

            #statistic
            ystart, ystop, xstart, xstop = Cropping.get_boundary(instanceImgnp)
            masksize=Cropping.get_sizeofbdary(instanceImgnp)
            masksize_percentage=masksize*1.0/mysize

            # skip truncated one
            if ystart<=10 or xstart<=10 or ystop>=annotation.imgHeight-10 or xstop>=annotation.imgWidth-10:
                continue

            #0.05 gt_Fine_base
            if masksize_percentage>=0.05:
                # print ('id,label,masksize,imgsize',id,label,masksize_percentage,size)
                # plt.imshow(instanceImgnp)
                # plt.show()
                instanceImg_arr.append(instanceImgnp)
                Ids_arr.append(id)
                Sizes_arr.append(masksize_percentage)
                nbInstances[label] += 1

        except:
            print("Failed to draw polygon with label {} and id {}: {}".format(label,id,polygon))
            raise

    return instanceImg_arr,Ids_arr,Sizes_arr,nbInstances[labelfilter]

# A method that does all the work
# inJson is the filename of the json file
# outImg is the filename of the instance image that is generated
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs


def json2instanceArr(inJson,encoding="ids",label_tochose='car'):
    annotation = Annotation()
    annotation.fromJsonFile(inJson)
    instanceImg =  getInstancewithLabel( annotation , encoding, label_tochose)
    return instanceImg

# The main method, if you execute this script directly
# Reads the command line arguments and calls the method 'json2instanceImg'
def main(argv):
    trainIds = False
    try:
        opts, args = getopt.getopt(argv,"ht")
    except getopt.GetoptError:
        printError( 'Invalid arguments' )
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit(0)
        elif opt == '-t':
            trainIds = True
        else:
            printError( "Handling of argument '{}' not implementend".format(opt) )

    if len(args) == 0:
        printError( "Missing input json file" )
    elif len(args) == 1:
        printError( "Missing output image filename" )
    elif len(args) > 2:
        printError( "Too many arguments" )

    inJson = args[0]
    outImg = args[1]

    if trainIds:
        json2instanceImg( inJson , outImg , 'trainIds' )
    else:
        json2instanceImg( inJson , outImg )

# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])
