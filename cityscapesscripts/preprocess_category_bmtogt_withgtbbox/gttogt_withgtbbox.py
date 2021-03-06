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

def main(category,set):

    def pathtoparent(full_path):
        n_last = len(full_path.split('/')[-1])
        parent_dir = full_path[:-n_last]
        pathtodir(parent_dir)
        return

    # datasetname='cityscape'
    datasetname='cocoamodal'

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

    # carset_name='gtFine_car'
    # carset_name='gtFine_allcar'
    # carset_name='proposals_bmcomplete_car'
    carset_name=os.path.join('gtFine_amodal',category)
    carset_proposals_name=os.path.join('gtFine_amodal',category)
    # carset_proposals_name='gtFine_visible_car'

    patchroot=os.path.join(root,carset_proposals_name)

    picklename_bbox=set+'_carbbox'

    if datasetname=='cocoamodal':
        searchFine = os.path.join(root, carset_name, set,"*","*.png")
        searchProposals=os.path.join(root,carset_proposals_name,set,"*","*.png")
        # searchFine = os.path.join(root, carset_name, set, "*","*.png")
        # searchProposals=os.path.join(root,carset_proposals_name,set,"*","*.png")
    elif datasetname == 'cityscape':
        searchFine   = os.path.join( root , carset_name  , set , "*", "*.png" )


    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()
    filesProposals=glob.glob(searchProposals)
    filesProposals.sort()
    # concatenate fine and coarse
    files = filesFine


    # quit if we did not find anything
    if not files:
        print( "Did not find any gt files. Please consult the README." )
        return
    if not filesProposals:
        print( "Did not find any gt files. Please consult the README." )
        return

    # a bit verbose
    print("Processing {} gt files".format(len(files)))
    print("Processing {} proposals files".format(len(filesProposals)))

    # iterate through files

    extractObj = extractFeature(pretrained=True,usecuda=True)

    gtfeaObj=gtmaskfeaDataset()
    gtpathObj=gtpathDataset()

    bbox_dict_total={}
    for i,(f_gt,f_posal) in enumerate(zip(files,filesProposals)):
        assert (f_gt==f_posal)
        print('Processing........',i,len(files),i*1.0/len(files))

        gtmask=imread(f_gt)
        posalmask=imread(f_posal)
        # dst_img_temp=f_gt.replace(carset_name,'leftImg8bit')
        dst_img_temp=f_posal.replace(carset_proposals_name,'leftImg8bit')
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

        dst_sin_gtmask=f_posal.replace('/'+set+'/','/'+set+'singlemask_target'+'/')
        dst_sin_posalmask=f_posal.replace('/'+set+'/','/'+set+'singlemask'+'/')
        dst_sinimg=f_posal.replace('/'+set+'/','/'+set+'singleimg'+'/')

        pathtoparent(dst_sin_gtmask)
        pathtoparent(dst_sin_posalmask)
        pathtoparent(dst_sinimg)

        try:
            #trick
            if datasetname=='cocoamodal':
                dst_img=os.path.join(root,'leftImg8bit',set,foldername+'.jpg')
            img=imread(dst_img)
        except:
            print ('no image found in:{}',dst_img)

        #if singlemask or singleimg not exist, create one
        if not os.path.isfile(dst_sin_posalmask):
            boundary=Cropping.get_boundary(gtmask,expand=0.1)
            sin_gtmask=Cropping.trim(gtmask,boundary)
            sin_posalmask=Cropping.trim(posalmask,boundary)
            imsave(dst_sin_posalmask,sin_posalmask)
            imsave(dst_sin_gtmask,sin_gtmask)
        else:
            sin_gtmask=imread(dst_sin_gtmask)
            sin_posalmask=imread(dst_sin_posalmask)
        if not os.path.isfile(dst_sinimg):
            boundary=Cropping.get_boundary(gtmask,expand=-1)
            sinimg=Cropping.trim(img,boundary)
            imsave(dst_sinimg,sinimg)
        else:
            sinimg=imread(dst_sinimg)

        #bbox gt
        # id=dst_sin_posalmask.split('/')[-1]
        # bbox_data=Cropping.get_det_amodaldata(posalmask,gtmask,h_target=224,w_target=224)
        # bbox_dict_total[id]=bbox_data
        # print(bbox_data.keys())

        # #Extract feature
        # img_batch=[sinimg]
        # maskfeature_matrix= build_maskfeature_matrix(img_batch, extractObj)
        #
        # if datasetname=='cityscape':
        #     relative_path='/'.join(f_gt.split('/')[-2:])
        # elif datasetname=='cocoamodal':
        #     relative_path = f_gt.split('/')[-1]
        # else:
        #     raise

        # gtfeaObj.update(maskfeature_matrix)
        # gtpathObj.update([relative_path])
        #
        # with open(os.path.join(patchroot, picklefea), 'wb') as handle:
        #     pickle.dump(gtfeaObj, handle)
        #
        # with open(os.path.join(patchroot, picklepath), 'wb') as handle:
        #     pickle.dump(gtpathObj, handle)

    # with open(os.path.join(patchroot, picklename_bbox), 'wb') as handle:
    #     pickle.dump(bbox_dict_total, handle)



    return

if __name__ == "__main__":
    # for category in ['bus','bottle','bowl','chair','person']:
    for category in ['car','bus','bottle','bowl','chair','person']:
        for set in ['train','val']:
            print('Start.......',category,set)
            main(category=category,set=set)
            print('End.......',category,set)