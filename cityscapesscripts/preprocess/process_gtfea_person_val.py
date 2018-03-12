from __future__ import print_function
import os, glob, sys
from basic_utils import *
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import *

# cityscapes imports
sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )

def main():
    datasetname='cityscape'
    # datasetname='cocoamodal'

    cocoamodalPath='../../../data/cocoamodal'
    cityscapesPath='../../../data/cityscape'
    # os.environ['CITYSCAPES_DATASET']=root



    # if 'CITYSCAPES_DATASET' in os.environ:
    #     # cityscapesPath = os.environ['CITYSCAPES_DATASET']
    #     cityscapesPath=root
    # else:
    #     cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    # how to search for all ground truth
    if datasetname=='cocoamodal':
        root=cocoamodalPath
    elif datasetname=='cityscape':
        root=cityscapesPath

    set='val'
    maskset_name = 'gtFine_allperson'
    # maskset_name='gtFine_car'
    # maskset_name='gtFine_allcar'


    if datasetname=='cocoamodal':
        searchFine = os.path.join(root, maskset_name, set, "*.png")
    elif datasetname == 'cityscape':
        searchFine   = os.path.join( root , maskset_name  , set , "*", "*.png" )


    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()

    # concatenate fine and coarse
    files = filesFine
    # files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        print( "Did not find any files. Please consult the README." )
        return

    # a bit verbose
    print("Processing {} gt files".format(len(files)))

    # iterate through files


    for i,f in enumerate(files):

        print('Processing........',i,len(files),i*1.0/len(files))
        gtmask=imread(f)

        # dst_img_temp=f.replace(maskset_name,'leftImg8bit')
        dst_img_temp=f.replace(maskset_name,'leftImg8bit')
        if datasetname=='cityscape':
            dst_img_list=dst_img_temp.split('.')
            dst_img_list[-2]=dst_img_list[-2][:-len('13000')]+'leftImg8bit'
            dst_img='.'.join(dst_img_list)
        elif datasetname=='cocoamodal':
            dst_img_list = dst_img_temp.split('.')
            dst_img_list[-1]='jpg'
            idx_length=len('_'+dst_img_list[-2].split('_')[-1])
            dst_img_list[-2] = dst_img_list[-2][:-idx_length]
            dst_img='.'.join(dst_img_list)
            foldername=dst_img.split('/')[-1].split('.')[0]

        dst_sinmask=f.replace('/'+set+'/','/'+set+'A'+'/')
        dst_sinimg=f.replace('/'+set+'/','/'+set+'B'+'/')

        n_last = len(dst_sinmask.split('/')[-1])
        parent_dir = dst_sinmask[:-n_last]
        pathtodir(parent_dir)

        n_last = len(dst_sinimg.split('/')[-1])
        parent_dir = dst_sinimg[:-n_last]
        pathtodir(parent_dir)

        try:
            img=imread(dst_img)
        except:
            print ('no image found in:{}',dst_img)

        #if singlemask or singleimg not exist, create one
        if not os.path.isfile(dst_sinmask):
            boundary=Cropping.get_boundary(gtmask,expand=0.1)
            sinmask=Cropping.trim(gtmask,boundary)
            imsave(dst_sinmask,sinmask)
        else:
            sinmask=imread(dst_sinmask)
        if not os.path.isfile(dst_sinimg):
            bool_gtmask=gtmask.astype(bool).astype(int)
            boundary=Cropping.get_boundary(gtmask,expand=-1)
            # masked_img=np.expand_dims(bool_gtmask,2)*img
            sinimg=Cropping.trim(img,boundary)
            imsave(dst_sinimg,sinimg)
        else:
            sinimg=imread(dst_sinimg)


    return

if __name__ == "__main__":
    main()