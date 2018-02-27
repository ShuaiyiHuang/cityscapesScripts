import torch
import numpy as np
from basic_utils import BasicTransform,BasicDict

class myTransform(BasicTransform):
    def __init__(self):
        super(BasicTransform, self).__init__()
        return

    @staticmethod
    def batchscale(batchimg, size):
        new_batchimg_list = []
        for img in batchimg:
            new_img = BasicTransform.Scale(img, size)
            new_img_expand = np.expand_dims(new_img, axis=0)
            new_batchimg_list.append(new_img_expand)
        new_batchimg_np = np.concatenate(tuple(new_batchimg_list))
        return new_batchimg_np

    @staticmethod
    def batchtoTensor(batchimg):
        new_batchimg_list = []
        for img in batchimg:
            if (len(img.shape)==2):
                #it is a mask
                img=np.expand_dims(img,2)
            new_img = BasicTransform.toTensor(img)
            new_img_4D = torch.unsqueeze(new_img, dim=0)
            new_batchimg_list.append(new_img_4D)

        new_batchimg = torch.cat(new_batchimg_list)
        return new_batchimg

    def transformmain(self, imgbatch, size=(224, 224)):

        rescale_imgbatch = myTransform.batchscale(imgbatch, size)
            # triky..*225
        tensor_imgbatch = myTransform.batchtoTensor(255 * rescale_imgbatch)

        return tensor_imgbatch

def build_maskfeature_matrix(img_batch, extractObj):

    mytransformObj = myTransform()

    tensor_imgbatch = mytransformObj.transformmain(img_batch, (224, 224))
    # feature = extractObj.extract(tensor_imgbatch)

    bsize=len(tensor_imgbatch)
    feature=[]
    #[bxCxHxW]
    for bid in range(bsize):
        bid=torch.LongTensor([bid])
        cur_img=tensor_imgbatch[bid]
        cur_feature=extractObj.extract(cur_img)
        if len(feature)==0:
            feature=cur_feature
        else:
            feature=torch.cat((feature,cur_feature),dim=0)
            assert (feature.size()[1]==2048)

    if extractObj.usecuda:
        maskfeature_matrix = feature.data.cpu().numpy()
    else:
        maskfeature_matrix = feature.data.numpy()

    assert (len(maskfeature_matrix.shape) == 2 and maskfeature_matrix.shape[1] == 2048)

    return maskfeature_matrix

class gtmaskfeaDataset(BasicDict):
    def __init__(self):
        self.whole_data=[]
        self.ntotal=0

        #Basic
        self.ntotal_folder=0
        self.whole_dict=[]
        self.ntotal_items=0
        self.whole_list=[]
        return
    def update(self,maskfeature_matrix):
        data=maskfeature_matrix
        self.whole_data.append(data)
        self.ntotal+=1
        self.updatedict(maskfeature_matrix)
        return

class gtpathDataset(BasicDict):
    def __init__(self):
        self.whole_data=[]
        self.ntotal=0
        #Basic
        self.ntotal_folder=0
        self.whole_dict=[]
        self.ntotal_items=0
        self.whole_list=[]
        return
    def update(self,new_gtpath_list):

        data=new_gtpath_list
        self.whole_data.append(data)
        self.ntotal+=1
        print('new gtpath list',new_gtpath_list)
        self.updatedict(new_gtpath_list)
        return