#!/usr/bin/python
#
# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode the ground truth classes and the
# individual instance of that classes.
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#   b) *labelIds.png          : the class is encoded by its ID
#   c) *instanceIds.png       : the class and the instance are encoded by an instance ID
# 
# With this tool, you can generate option
#   d) *instanceTrainIds.png  : the class and the instance are encoded by an instance training ID
# This encoding might come handy for training purposes. You can use
# the file labes.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Please refer to 'json2instanceImg.py' for an explanation of instance IDs.
#
# Uses the converter tool in 'json2instanceImg.py'
# Uses the mapping defined in 'labels.py'
#

# python imports
from __future__ import print_function
import os, glob, sys

# cityscapes imports
sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
from csHelpers import printError
from json2instanceImg import json2instanceArr
from basic_utils import pathtodir
from scipy.misc import imread, imsave

# The main method
def main():
    # Where to look for Cityscapes
    root='../../../data/cityscape'
    os.environ['CITYSCAPES_DATASET']=root
    store_path=os.path.join(root,'gtFine_car')
    set='train'
    # city="darmstadt"
    # if not os.path.exists(store_path):
    #     os.mkdir(store_path)
    # set_path=os.path.join(store_path,set)
    # if not os.path.exists(set_path):
    #     os.mkdir(set_path)
    # city_path=os.path.join(set_path,city)
    # if not os.path.exists(city_path):
    #     os.mkdir(city_path)

    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    # how to search for all ground truth
    searchFine   = os.path.join( cityscapesPath , "gtFine"   , set , "*", "*_gt*_polygons.json" )


    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()


    # concatenate fine and coarse
    files = filesFine
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        printError( "Did not find any files. Please consult the README." )

    # a bit verbose
    print("Processing {} annotation files".format(len(files)))

    # iterate through files
    progress = -1
    numIns_all=0
    numimg_selected=0
    # print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for i,f in enumerate(files):
        # create the output filename
        progress += 1
        # print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        print('Processing........',i,len(files),i*1.0/len(files))

        # if numIns_all>=1000:
        #     break
        # do the conversion
        try:
            # json2instanceImg( f , dst , "trainIds" )
            print(i,'image............',numIns_all,'ins selected now')
            insImg_arr,insIds_arr,Sizes_arr,num_instances=json2instanceArr(f,'trainIds',label_tochose='car')
            if num_instances==0:
                # progress += 1
                continue
            numIns_all+=num_instances
            for instance,id,size in zip(insImg_arr,insIds_arr,Sizes_arr):

                try:
                    dst = f.replace("_gtFine_polygons.json", "_"+str(id)+".png")
                    # dst = dst.replace("gtFine", "gtFine_base_car")
                    dst = dst.replace("gtFine", "gtFine_base_car_bigger")

                    n_last=len(dst.split('/')[-1])
                    parent_dir=dst[:-n_last]
                    pathtodir(parent_dir)


                    imsave(dst,instance)
                    numimg_selected+=1
                except:
                    print("Failed to save: {}".format(id))
                    raise
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status


        sys.stdout.flush()

    print('Ins count all',numIns_all,',all imgs',len(files),',seleced imgs',numimg_selected,',average instances per img selected',numIns_all*1.0/numimg_selected,',ave ins per img all',numIns_all*1.0/len(files))



# call the main
if __name__ == "__main__":
    main()
