import pickle
# from preprocess_maskGNN import MyData
import os
from skimage.io import imread,imsave
import numpy as np
import time

from basic_utils import BasicTransform,pathtodir
from utils import gtpathDataset,gtmaskfeaDataset

def annfromfilename(amodal,filename):
    filename_full=filename.split('_')
    id_str=filename_full[2]
    imgid=int(id_str)
    # print 'imgid:',imgid

    annId = amodal.getAmodalAnnIds(imgIds=imgid)
    anns = amodal.loadAnns(annId)
    ann=anns[0]
    # print ann
    return ann

# def get_id(ann,foldername):
#     a=foldername.split('_')
#     id=int(a.split('.')[0])
#     return id
#
# def get_mask(ann,id,h=224,w=224,isvisible=True):
#     index=id-1
#     rle=mask.frPyObjects(ann['regions'][index]['segmentation'],h,w)
#     m=mask.decode(rle)
#     return m

class KNNGraph(object):
    def __init__(self, datasetname,feapicklepath,pathpicklepath,singleimgpath,singlemaskpath,gtroot_all='',  graphmask_dir='',resizedmaskroot='',origmaskroot='', msize=3, knn=5, extractObj=None):

        self.gtroot_all=gtroot_all
        self.maskroot=resizedmaskroot
        self.origmaskroot=origmaskroot
        self.graphmask_dir=graphmask_dir

        self.checkpath(self.graphmask_dir)

        self.gtarray=[]
        self.maskarray=[]
        self.msize=msize
        self.knn=knn
        self.mk=msize*knn
        self.size=224
        # self.extractObj=extractObj

        # self.mask_foldernames,self.mask_masknames=KNNGraph.build_folderdict(self.maskroot)
        # self.mask_masknames=self.build_namedict(self.gtroot_all)
        # self.gt_names=self.build_namedict(self.gtroot_all)

        self.processid=0

        gtfeadataset = self.load_txt(feapicklepath)
        gtpathdataset = self.load_txt(pathpicklepath)

        gtfeadataset=np.array(gtfeadataset,dtype=float)
        gtpathdataset = np.array(gtpathdataset, dtype=object)
        self.gtfeadataset = gtfeadataset
        self.gtpathdataset = gtpathdataset
        self.singleimgpath=singleimgpath
        self.singlemaskpath=singlemaskpath

        assert (len(gtfeadataset)==len(gtpathdataset))

        print(feapicklepath,'num_data:',len(gtfeadataset))

        self.datasetname=datasetname

        return

    def load_txt(self,picklepath):
        datasetObj = pickle.load(open(picklepath))
        # dataset = datasetObj.whole_data
        dataset = datasetObj.whole_list
        return dataset

    @staticmethod
    def load_image(path,isnp=True):
        startt=time.time()

        image_list=[]
        listing=os.listdir(path)
        for imgname in listing:
            assert (imgname.endswith('.png'))
            img=imread(os.path.join(path,imgname))
            image_list.append(img)
        if isnp:
            image_list=np.array(image_list)
        endt = time.time()
        print('cost time', endt - startt, 's', 'length', len(image_list), image_list.shape)
        return image_list

    @staticmethod
    def load_mask(path,isnp=True):
        startt = time.time()

        mask_list=[]
        listing=os.listdir(path)
        for i,foldername in enumerate(listing):
            curpath=os.path.join(path,foldername)
            cur_listing=os.listdir(curpath)

            for j,maskname in enumerate(cur_listing):
                assert (maskname.endswith('.png'))
                mask=imread(os.path.join(curpath,maskname))
                mask_list.append(mask)
        if isnp:
            mask_list=np.array(mask_list)

        endt = time.time()
        print('cost time',endt - startt,'s','length',len(mask_list),mask_list.shape,'path',path)

        return mask_list

    def read_masks(self,singleimg_dir,singlemask_dir,pathlist):
        #eg relative_path: foldername/foldername_i.png
        ratio_set=[]
        singlemask_set=[]
        singleimg_set=[]
        for relative_path in pathlist:

            mask=imread(os.path.join(singlemask_dir,relative_path))
            imgpatch=imread(os.path.join(singleimg_dir,relative_path))
            assert (len(mask.shape)==2)

            #read single imgpatch and gtmask
            singlemask_set.append(mask)
            singleimg_set.append(imgpatch)

            w,h=mask.shape
            ratio=w*1.0/h
            ratio_set.append(ratio)

        ratio_set=np.array(ratio_set)

        return ratio_set,singleimg_set,singlemask_set

    @staticmethod
    def build_folderdict(path):
        listing=os.listdir(path)
        mask_foldername=[]
        mask_maskname=[]
        for i,foldername in enumerate(listing):
            curpath=os.path.join(path,foldername)
            cur_listing=os.listdir(curpath)
            for j, maskname in enumerate(cur_listing):
                mask_foldername.append(foldername)
                mask_maskname.append(maskname)

        return mask_foldername,mask_maskname

    def build_namedict(self,path):
        name_list=[]
        listing=os.listdir(path)
        for imgname in listing:
            name_list.append(imgname)
        return name_list

    @staticmethod
    def calIoU(input, candidates):
        picked_prop=input.astype(bool)
        bool_proposals=[]
        for proposal in candidates:
            bool_proposals.append(proposal.astype(bool))

        intersection_nonzero=[np.count_nonzero(merged) for i,merged in enumerate(picked_prop*bool_proposals)]
        union_nonzero=[np.count_nonzero(merged) for i,merged in enumerate(picked_prop+bool_proposals)]

        all_IoU=1.0*np.array(intersection_nonzero)/np.array(union_nonzero)

        return all_IoU

    def findtopK(self,score,k,descending=True,items=[]):
        if descending:
            arg_indices=(-score).argsort()
        else:
            arg_indices=score.argsort()

        if k>len(score):
            print('k exceed max length,let k be max length')
            k=len(score)

        topk_ind=arg_indices[:k]
        topk_score=score[topk_ind]
        if items!=[]:
            topk_items=items[topk_ind]
            return topk_items,topk_ind,topk_score
        else:
            return topk_ind, topk_score

    def findknn_batch(self, mask, candidate_masks, k):
        all_IoU=self.calIoU(mask, candidate_masks)
        topk_items, topk_ind, topk_score=self.findtopK(score=all_IoU,k=k,descending=True,items=self.gtarray)

        return topk_items,topk_ind,topk_score

    def build_connection(self, mask, candidate_masks, msize, knn):
        nvertice=msize*knn+1
        connection_matrix=np.zeros((nvertice,nvertice),dtype=int)
        topneighbor_masks,topneighbor_ind,topneibor_score=self.findknn_batch(mask, candidate_masks, nvertice - 1)

        masks_batch=np.concatenate((np.expand_dims(mask,0),topneighbor_masks))

        assert (masks_batch.shape==(nvertice,self.size,self.size))
        for rowid in range(nvertice):
            if rowid==0:
                connection_matrix[rowid,1:]=1
                continue
            input=masks_batch[rowid]
            candidates=np.concatenate((masks_batch[0:rowid],masks_batch[rowid+1:]))
            absolute_candind=np.concatenate((np.arange(0,rowid),np.arange(rowid+1,nvertice)))

            score=self.calIoU(input,candidates)
            topk_relativeind, topk_score=self.findtopK(score,knn)

            selected_ind=absolute_candind[topk_relativeind]

            connection_matrix[rowid,selected_ind]=1

        return connection_matrix,topneighbor_ind

    def save(self,names,items):
        for name,item in zip(names,items):
            imsave(name,item)
        return

    def save_graph(self,topneighbor_ind):
        paths=[]
        items=[]

        # mask_foldername=self.mask_foldernames[self.processid]
        mask_foldername=str(self.processid)
        mask_name='cur_'+self.mask_masknames[self.processid]

        cur_graphdir=os.path.join(self.graphmask_dir,mask_foldername,str(self.processid))
        self.checkpath(os.path.join(self.graphmask_dir,mask_foldername))
        self.checkpath(cur_graphdir)

        maskname_tosave=os.path.join(cur_graphdir,mask_name)
        paths.append(maskname_tosave)
        items.append(self.maskarray[self.processid])

        #get visible mask
        # ann=annfromfilename(amodal,self.mask_masknames[self.processid])
        # mask=get_visiblemask(ann,self.mask_masknames[self.processid])


        for rank,gtind in enumerate(topneighbor_ind):
            paths.append(os.path.join(cur_graphdir, str(rank)+'_'+self.gt_names[gtind]))
            items.append(self.gtarray[gtind])

        self.save(paths,items)
        return

    def save_top(self,topneighbor_ind,cur_maskfoldername,whetherresize=False,iscityscape=False):
        paths=[]
        items=[]

        mask_name=self.gtpathdataset[self.processid]
        if iscityscape:
            imgname=mask_name.split('/')[-1].split('.')[0][:-len('_13000')]
            insid=mask_name.split('/')[-1].split('.')[0][-len('13000'):]
            mask_foldername = os.path.join(cur_maskfoldername,imgname)
            cur_graphdir = os.path.join(self.graphmask_dir, mask_foldername, str(insid))
            pathtodir(cur_graphdir)
        else:
            mask_foldername=cur_maskfoldername
            cur_graphdir = os.path.join(self.graphmask_dir, mask_foldername, str(self.processid))
        mask_name_tostore='cur_'+mask_name.split('/')[-1]
        mask_item=imread(os.path.join(self.singlemaskpath,mask_name))
        w,h=mask_item.shape


        self.checkpath(os.path.join(self.graphmask_dir,mask_foldername))
        self.checkpath(cur_graphdir)

        maskname_tosave=os.path.join(cur_graphdir,mask_name_tostore)
        paths.append(maskname_tosave)
        items.append(mask_item)

        for rank,gtind in enumerate(topneighbor_ind):
            paths.append(os.path.join(cur_graphdir, str(rank)+'_'+self.gtpathdataset[gtind].split('/')[-1]))
            item=imread(os.path.join(self.singlemaskpath,self.gtpathdataset[gtind]))
            if whetherresize==True:
                item=BasicTransform.Scale(item,(w,h))
            items.append(item)

        for rank,gtind in enumerate(topneighbor_ind):
            paths.append(os.path.join(cur_graphdir, 'img'+str(rank)+'_'+self.gtpathdataset[gtind].split('/')[-1]))
            item=imread(os.path.join(self.singleimgpath,self.gtpathdataset[gtind]))
            items.append(item)

        self.save(paths,items)

    def get_visiblemask(ann, name):
        #not implemented yet
        return

    def checkpath(self,path):
        if not os.path.exists(path):
            os.mkdir(path)



    def mainnew(self):
        ratio_set, singleimg_set, singlemask_set=self.read_masks(self.singleimgpath,self.singlemaskpath,self.gtpathdataset)

        num_data=len(self.gtfeadataset)

        startt=time.time()
        dists=self.calculate_dists_fast(self.gtfeadataset)
        endt=time.time()
        print('calculate dist cost time',endt-startt)

        ratio_range=[0.9,1.4]
        print('ratio range from',ratio_range)

        topkinds=self.get_topkinds_ratio(dists,self.mk,ratio_range,ratio_set)


        for i in range(num_data):
            startt=time.time()
            topkneighbor=topkinds[i]
            if self.datasetname=='cityscape':
                cur_maskfoldername=self.gtpathdataset[i].split('/')[0]
                iscityscape=True
            else:
                cur_maskfoldername='_'.join(self.gtpathdataset[i].split('.')[0].split('_')[:-1])
                iscityscape=False

            self.save_top(topkneighbor,cur_maskfoldername,whetherresize=True,iscityscape=iscityscape)
            self.processid+=1
            endt=time.time()
            print('Done',i,i*1.0/num_data,'cost',endt-startt)

        return

    def calculate_dists(self,gtfeadataset):

        num_gt=len(gtfeadataset)

        dists=np.zeros((num_gt,num_gt),dtype=float)
        for i in xrange(num_gt):

            dists[i:] = np.sqrt(np.sum((gtfeadataset[i] -gtfeadataset) ** 2, 1))

        return dists

    def calculate_dists_fast(self, data):

        num_gt=len(data)

        x_sq=np.square(data).sum(axis=1)
        y_sq=np.square(data).sum(axis=1)
        xy=data.dot(data.T)
        print('x_sq,y_sq,xy shape',x_sq.shape,y_sq.shape,xy.shape)

        dists=np.sqrt(xy*(-2)+x_sq.reshape(x_sq.shape[0],1)+y_sq)
        print('dists shape',dists.shape)
        assert (dists.shape==(num_gt,num_gt))
        return dists

    def get_topkinds(self,dists,k):
        num_data=dists.shape[0]
        topkinds=np.zeros((num_data,k),dtype=int)
        for i in xrange(num_data):
            ind_knearest=np.argsort(dists[i])[:k]
            topkinds[i]=ind_knearest
        return topkinds

    def get_topkinds_ratio(self,dists,k,ratio_range,ratio_set):
        num_data=dists.shape[0]
        topkinds=[]

        rlow=ratio_range[0]
        rhigh=ratio_range[1]

        ratio_candi_count=0
        knearest_count=0
        for i in xrange(num_data):
            relative_ratiolist=ratio_set[i]*1.0/ratio_set
            candi_ind=np.argwhere((relative_ratiolist<=rhigh) & (relative_ratiolist>=rlow)).flatten()
            candi_dists=dists[i][candi_ind]
            relative_ind_knearest=np.argsort(candi_dists)[:k]
            absolute_ind_knearest=candi_ind[relative_ind_knearest]
            topkinds.append(absolute_ind_knearest)

            ratio_candi_count+=len(candi_ind)
            knearest_count+=len(absolute_ind_knearest)

        print('selected ratio: ',ratio_candi_count,'average: ',ratio_candi_count*1.0/num_data)
        print('selected knearest: ',knearest_count,'average: ',knearest_count*1.0/num_data)

        return topkinds

    def predict(self,dists,k,pathdataset):
        num_data=dists.shape[0]
        predict_y=np.zeros(num_data,k)
        for i in xrange(num_data):
            ind_knearest=np.argsort(dists[i])[:k]
            closest_path=pathdataset[ind_knearest]
            predict_y[i]=closest_path
        return predict_y

    def main(self):
        startt=time.time()
        self.gtarray=KNNGraph.load_image(self.gtroot_all)
        self.maskarray=KNNGraph.load_image(self.gtroot_all)
        for i,mask in enumerate(self.maskarray[:40]):
            connection_matrix,topneighbor_ind=self.build_connection(mask,self.gtarray,self.msize,self.knn)
            print connection_matrix
            self.save_graph(topneighbor_ind)
            self.processid+=1
        endt=time.time()
        print('cost time',endt-startt,'s')


class KNN():
    def __init__(self):
        return


# def load_annotation(dataDir, gtFile):
#     from pycocotools.amodal import Amodal
#
#     dataset = 'coco' # 'coco' or 'bsds'
#     if dataset == 'coco':
#         print("show COCO examples")
#         dataDir=dataDir
#         gtFile = gtFile
#     elif dataset == 'bsds':
#         print("show BSDS examples")
#         dataDir = '../bsds_images/test/'
#         gtFile = '../annotations/BSDS_amodal_test.json'
#     amodal=Amodal(gtFile)
#     return amodal

if __name__ == "__main__":


    print()

    #for coco
    # delete first two rows
    # gtroot_all='../../../data/depth_ordering/FCIS_merge_0102_gt/gtallthings/'+set
    # gtroot_filtered = '../../../data/depth_ordering/FCIS_merge_0102_gt/gtbestmatchthings/' + set

    # set='train'
    # origmaskroot='../../../data/depth_ordering/FCIS_'+set+'2014'
    # newmask_dir = '../../../data/depth_ordering/FCIS_gtpatch_0122/'
    # graphmask_dir=newmask_dir+set+'_'+'ratiomask_fromgt'
    # msize=5
    # knn=3
    #
    # dataDir='/home/hsy/work/hsy/repertory/data/coco/'+set+'2014/'
    # gtFile='/home/hsy/work/hsy/repertory/data/amodal/annotations/COCO_amodal_train2014.json'

    # root='../../../data/depth_ordering/FCIS_gtpatch_0122/'
    # feapicklepath=root+'2class_gtthingsall_0122_fea'+set
    # pathpicklepath=root+'2class_gtthingsall_0122_path'+set
    # singleimgpath=os.path.join(root,set+'singleimgpatch')
    # singlemaskpath=os.path.join(root,set+'singlemaskpatch')

    #For cityscape
    # set='train'
    #
    # root='../../../data/cityscapes/gtFine_car/'
    # graphmask_dir=os.path.join(root,set+'_'+'knn_citytocity_ratiogt')
    #
    #
    # feapicklepath=root+'cityscape_car_0126_fea'+set
    # pathpicklepath=root+'cityscape_car_0126_path'+set
    # singleimgpath=os.path.join(root,set+'singleimg')
    # singlemaskpath=os.path.join(root,set+'singlemask')
    # msize=5
    # knn=3

    #For cocoamodal
    set='test'
    datasetname='cocoamodal'
    root='../../../data/'+datasetname+'/gtFine_car/'
    graphmask_dir=os.path.join(root,set+'_'+'knn_cocotococo_ratiogt')


    feapicklepath=root+datasetname+'_car_0126_fea'+set
    pathpicklepath=root+datasetname+'_car_0126_path'+set
    singleimgpath=os.path.join(root,set+'singleimg')
    singlemaskpath=os.path.join(root,set+'singlemask')

    msize=5
    knn=3

    knngraphObj = KNNGraph(datasetname,feapicklepath,pathpicklepath,singleimgpath,singlemaskpath,graphmask_dir=graphmask_dir,knn=knn,msize=msize)
    knngraphObj.mainnew()