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
    # datasetname='cityscape'
    datasetname='cocoamodal'

    cocoamodalPath='../../../data/cocoamodal'
    cityscapesPath='../../../data/cityscapes'
    # os.environ['CITYSCAPES_DATASET']=root



    # if 'CITYSCAPES_DATASET' in os.environ:
    #     # cityscapesPath = os.environ['CITYSCAPES_DATASET']
    #     cityscapesPath=root
    # else:
    #     cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
    # how to search for all ground truth
    if datasetname=='cocoamodal':
        root=cocoamodalPath
    elif datasetname=='cityscapes':
        root=cityscapesPath

    set='test'

    patchroot=os.path.join(root,'gtFine_car')
    picklefea=datasetname+'_car_0126_fea'+set
    picklepath=datasetname+'_car_0126_path'+set

    if datasetname=='cocoamodal':
        searchFine = os.path.join(root, "gtFine_car", set, "*.png")
    elif datasetname == 'cityscapes':
        searchFine   = os.path.join( root , "gtFine_car"   , set , "*", "*.png" )


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

    # print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    extractObj = extractFeature(pretrained=True,usecuda=True)

    gtfeaObj=gtmaskfeaDataset()
    gtpathObj=gtpathDataset()

    for i,f in enumerate(files):

        print('Processing........',i,len(files),i*1.0/len(files))
        gtmask=imread(f)

        dst_img_temp=f.replace('gtFine_car','leftImg8bit')
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

        dst_sinmask=f.replace('/'+set+'/','/'+set+'singlemask'+'/')
        dst_sinimg=f.replace('/'+set+'/','/'+set+'singleimg'+'/')

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
            boundary=Cropping.get_boundary(gtmask,expand=0)
            sinmask=Cropping.trim(gtmask,boundary)
            imsave(dst_sinmask,sinmask)
        else:
            sinmask=imread(dst_sinmask)
        if not os.path.isfile(dst_sinimg):
            bool_gtmask=gtmask.astype(bool).astype(int)
            boundary=Cropping.get_boundary(gtmask,expand=0)
            # masked_img=np.expand_dims(bool_gtmask,2)*img
            sinimg=Cropping.trim(img,boundary)
            imsave(dst_sinimg,sinimg)
        else:
            sinimg=imread(dst_sinimg)

        #Extract feature
        img_batch=[sinimg]
        maskfeature_matrix= build_maskfeature_matrix(img_batch, extractObj)

        if datasetname=='cityscape':
            relative_path='/'.join(f.split('/')[-2:])
        elif datasetname=='cocoamodal':
            relative_path = f.split('/')[-1]
        else:
            raise

        gtfeaObj.update(maskfeature_matrix)
        gtpathObj.update([relative_path])

        with open(os.path.join(patchroot, picklefea), 'wb') as handle:
            pickle.dump(gtfeaObj, handle)

        with open(os.path.join(patchroot, picklepath), 'wb') as handle:
            pickle.dump(gtpathObj, handle)

    return

if __name__ == "__main__":
    main()