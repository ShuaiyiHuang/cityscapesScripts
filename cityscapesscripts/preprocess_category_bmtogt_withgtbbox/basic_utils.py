import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from torchvision import transforms, utils,models
import torch
import torch.nn as nn
from torch.autograd import Variable
from skimage import transform



class ListingDict():
    def __init__(self,root):
        self.root=root
        self.dict={}
        self.whole_file_listing=[]
        self.ntotal=0

    def update(self, newlisting):
        self.whole_file_listing=self.whole_file_listing+newlisting
        self.ntotal+=len(newlisting)

    def build_dict(self,keys,values):
        assert (len(keys)==len(values))
        dict={key:values[i] for i,key in enumerate(keys)}
        return dict

    def build_index(self, rootpath):
        listing = os.listdir(rootpath)
        listing=listing[:2]
        for i, foldername in enumerate(listing):

            cur_path=os.path.join(rootpath, foldername)
            if  os.path.isdir(cur_path):
                filenames=os.listdir(cur_path)
                cur_file_names_lsit=[os.path.join(foldername,name) for name in filenames]
                self.update(cur_file_names_lsit)

        keys=np.arange(self.ntotal)
        values=self.whole_file_listing
        self.dict=self.build_dict(keys,values)

        return self.dict

    def load_one_data(self,key,iftorch=True,ifbatch=False):

        value=self.dict[key]
        path=os.path.join(self.root,value)
        if os.path.exists(path):
            img=imread(path)
        # img = transform.resize(img, (300, 300),mode='symmetric')
        print('max',np.max(img),'min',np.min(img))
        if iftorch:
            print(img.shape)

            img=BasicTransform.Scale(img,(224,224))

            print('after scale max', np.max(img), 'min', np.min(img))
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            img = BasicTransform.toTensor(255*img)
            print('after totensor max', torch.max(img), 'min', torch.min(img))
            print(img.shape)
        # if ifbatch:
        #     if type(img)==torch.FloatTensor:
        #         img=torch.unsqueeze(img,dim=0)
        #     else:
        #         img = np.expand_dims(img, 0)
        print(img.size())
        return img

class BasicTransform(object):
    def __init__(self):
        return

    @staticmethod
    def toTensor(input):
        '''
        Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
        [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].Will be divided by 255.
        '''
        output=transforms.ToTensor()(input)
        return output

    @staticmethod
    def Scale(img,size):
        '''
        :param img: np array [H,W,C]
        :param size: tuple (new_H,new_W)
        :return: np array [new_H,new_W,C] Attention:will rescale [0-255] to [0-1]
        '''
        output=transform.resize(img,size,mode='constant')
        return output

class Ploting():
    def __init__(self):
        return
    def showimg(self,images_batch):
        grid_image = utils.make_grid(images_batch)
        plt.figure()
        print('grid numpy max',np.max(grid_image.numpy()))
        print('grid torch max',torch.max(grid_image))
        # if np.max(grid_image.numpy())==1:
        #[0~1]
        plt.imshow(grid_image.numpy().transpose((1, 2, 0)))
        # else:
        #     plt.imshow(grid_image.numpy().transpose((1, 2, 0)))
        plt.show()

    @staticmethod
    def shownpimg(self,img):
        plt.figure()
        plt.imshow(img)
        plt.show()

class Cropping(object):
    def __init__(self):
        return
    @staticmethod
    def get_boundary(mask, expand=-1):
        '''
        :param mask: input binary mask
        :param expand: 
        :return: boundary of input msk,y-row,x-column
        '''
        h,w=mask.shape

        #expand=-1 means expand 5% scale of original shape


        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        h_bdry=ymax+1-ymin
        w_bdry=xmax+1-xmin
        if expand==-1:
            expand_h=np.minimum(3,int(0.05*h_bdry)) if h_bdry<=200 else 10
            expand_w=np.minimum(3,int(0.05*w_bdry)) if w_bdry<=200 else 10
        else:
            expand_h=int(expand*h_bdry)
            expand_w=int(expand*w_bdry)

        ystart = np.maximum(ymin - expand_h,0)
        ystop = np.minimum(ymax + 1 + expand_h,h)
        xstart = np.maximum(xmin - expand_w,0)
        xstop = np.minimum(xmax + 1 + expand_w,w)
        boundary = [ystart, ystop, xstart, xstop]
        return boundary

    @staticmethod
    def get_detboundary(mask,scale_w=1.0,scale_h=1.0):
        y1, y2, x1, x2=Cropping.get_boundary(mask,expand=0)
        x1_s,y1_s,x2_s,y2_s=int(x1*scale_w),int(y1*scale_h),int(x2*scale_w),int(y2*scale_h)

        det = [x1, y1, x2, y2]
        det_s=[x1_s,y1_s,x2_s,y2_s]
        result=det_s+det
        assert (len(result)==8)
        return np.array(result,dtype=int)

    @staticmethod
    def get_det_data(posalmask, gtmask, h_target=64, w_target=64):
        h_posalmask,w_posalmask=posalmask.shape
        h_gtmask,w_gtmask=gtmask.shape

        scale_h= h_target * 1.0 / h_posalmask
        scale_w=w_target*1.0 / w_posalmask


        result_posal=Cropping.get_detboundary(posalmask,scale_w=scale_w,scale_h=scale_h)
        result_gt=Cropping.get_detboundary(gtmask,scale_w=scale_w,scale_h=scale_h)

        result={'input_s':result_posal[:4],'gt_s':result_gt[:4],
                'input_orig':result_posal[-4:],'gt_orig':result_gt[-4:],
                'height_s':scale_h,'width_s':scale_w}

        return result

    @staticmethod
    def get_det_amodaldata(posalmask, gtmask, h_target=64, w_target=64):
        h_posalmask,w_posalmask=posalmask.shape
        h_gtmask,w_gtmask=gtmask.shape

        scale_h= h_target * 1.0 / h_posalmask
        scale_w=w_target*1.0 / w_posalmask


        result_posal=Cropping.get_detboundary(posalmask,scale_w=scale_w,scale_h=scale_h)
        result_gt=Cropping.get_detboundary(gtmask,scale_w=scale_w,scale_h=scale_h)

        x1_posal,y1_posal,x2_posal,y2_posal=result_posal[:4]
        x1_posal,y1_posal,w_posal,h_posal=x1_posal,y1_posal,x2_posal-x1_posal,y2_posal-y1_posal

        x1_gt,y1_gt,x2_gt,y2_gt=result_gt[:4]
        x1_gt,y1_gt,w_gt,h_gt=x1_gt,y1_gt,x2_gt-x1_gt,y2_gt-y1_gt

        if w_posal==0 or h_posal==0:
            print('w_posal h_posal 0:',w_posal,h_posal)
            target_amodal=np.array([0.0,0.0,0.0,0.0],dtype=float)
            target_fast=np.array([0.0,0.0,0.0,0.0],dtype=float)
        else:
            target_amodal=np.array([(1.0*(x1_posal-x1_gt))/w_posal,(1.0*(y1_posal-y1_gt))/h_posal,
                       (1.0*(w_posal-w_gt))/w_posal,(1.0*(h_posal-h_gt))/h_posal],dtype=float)
            target_fast=np.array([(1.0*(x1_posal-x1_gt))/w_posal,(1.0*(y1_posal-y1_gt))/h_posal,
                       np.log((1.0*w_gt)/w_posal),np.log((1.0*h_gt)/h_posal)],dtype=float)

        result={'input_s':result_posal[:4],'gt_s':result_gt[:4],
                'input_orig':result_posal[-4:],'gt_orig':result_gt[-4:],
                'height_s':scale_h,'width_s':scale_w,
                'target_amodal':target_amodal,
                'target_fast':target_fast}

        return result

    @staticmethod
    def get_sizeofbdary(mask, expand=0):
        ystart, ystop, xstart, xstop=Cropping.get_boundary(mask,expand)
        h=int(ystop-ystart)
        w=int(xstop-xstart)
        size=h*w
        return size

    @staticmethod
    def get_geometric_info(boundary):
        ystart, ystop, xstart, xstop=boundary
        ycenter=int(np.floor((ystart+ystop)/2))
        xcenter=int(np.floor((xstart+xstop)/2))
        h=int(ystop-ystart)
        w=int(xstop-xstart)
        assert (w>=0 and h>=0)
        geometric_info=[ycenter,xcenter,h,w]
        return geometric_info

    @staticmethod
    def trim(input, boundary):
        ystart, ystop, xstart, xstop = boundary[0], boundary[1], boundary[2], boundary[3]
        output = input[ystart:ystop, xstart:xstop]
        return output

    @staticmethod
    def bboxtrim(inputs_array, boundaries_array):
        outputs_array = []
        for input, boundary in zip(inputs_array, boundaries_array):
            output = Cropping.trim(input, boundary)
            outputs_array.append(output)
        return outputs_array

    @staticmethod
    def crop(inputs_array,boundaries_array):
        output_array = Cropping.bboxtrim(inputs_array, boundaries_array)
        return output_array

class extractnodefModel(nn.Module):
    def __init__(self,pretrained=True):
        super(extractnodefModel, self).__init__()
        original_model=models.resnet50(pretrained=pretrained)
        self.features=nn.Sequential(*list(original_model.children())[:-1])
    def forward(self,x):
        x=self.features(x)
        return x

class ClassifyModel(nn.Module):
    def __init__(self,resnet_pretrained=True,img3channel_pretrained=True,whole_model_pretrained=True):
        super(ClassifyModel,self).__init__()

        if whole_model_pretrained:
            model_ft = models.resnet50(pretrained=resnet_pretrained)
            num_ftrs = model_ft.fc.in_features
            model_ft.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
            model_ft.fc = nn.Linear(num_ftrs, 2)

            model_dict = torch.load('/home/hsy/work/hsy/repertory/models/depth_recall_1215/depth_val_epoch7')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            model_ft.load_state_dict(new_state_dict)
            self.features = model_ft
        else:
            if img3channel_pretrained:
                model_ft = models.resnet50(pretrained=resnet_pretrained)
                num_ftrs = model_ft.fc.in_features

                #orig img 3channel params
                origi_dict = model_ft.state_dict()
                param_conv1_pretrained = origi_dict['conv1.weight']

                #modify original resnet to adapt to 5channel and 2 label
                model_ft.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,bias=False)
                model_ft.fc = nn.Linear(num_ftrs, 2)

                #form new 5channel params from orig img 3channel params
                modi_dict = model_ft.state_dict()
                param_conv1_scratch = modi_dict['conv1.weight']
                param_conv1_maskchannel = param_conv1_scratch[:, [3, 4], :, :]
                param_conv1_5channel = torch.cat((param_conv1_pretrained, param_conv1_maskchannel), dim=1)
                assert (torch.equal(param_conv1_5channel[:, :3, :, :],
                                    param_conv1_pretrained) and param_conv1_5channel.size() == param_conv1_scratch.size())
                assert (not torch.equal(param_conv1_5channel[:, :3, :, :], param_conv1_scratch[:, :3, :, :]))

                #update conv1 parmas
                model_ft.conv1.weight.data = param_conv1_5channel

                #check if updated successfully
                param_conv1_imgchannel = model_ft.state_dict()['conv1.weight'][:, :3, :, :]
                assert (torch.equal(param_conv1_imgchannel, param_conv1_pretrained))

                self.features=model_ft
    def forward(self,x):
        scores=self.features(x)
        return scores

class extractedgefModel(nn.Module):
    def __init__(self,whole_model_pretrained=True):
        super(extractedgefModel,self).__init__()
        original_model=ClassifyModel(whole_model_pretrained=whole_model_pretrained)
        #features.childer not .childeren!
        self.features=nn.Sequential(*list(original_model.features.children())[:-1])
    def forward(self,x):
        x = self.features(x)
        return x

class extractFeature():
    def __init__(self,pretrained=True,usecuda=True):
        if usecuda:
            self.nodemodel = extractnodefModel(pretrained).cuda()
            self.edgemodel=torch.nn.DataParallel(extractedgefModel(pretrained),device_ids=[0]).cuda()
        else:
            self.nodemodel=extractnodefModel(pretrained)
            self.edgemodel = extractedgefModel(pretrained)
        self.usecuda=usecuda
        return
    def extract(self,input):
        if self.usecuda:
            input=Variable(input,volatile=True).cuda()
        else:
            input=Variable(input,volatile=True)
        features=self.nodemodel(input)
        bsize=features.size(0)
        features=features.view(bsize,-1)
        return features
    def extractedgef(self,inputs_img,inputs_mask1,inputs_mask2):
        input=torch.cat((inputs_img,inputs_mask1,inputs_mask2),dim=1)
        if self.usecuda:
            input=Variable(input,volatile=True).cuda()
        else:
            input=Variable(input,volatile=True)
        features=self.edgemodel(input)
        bsize=features.size(0)
        features=features.view(bsize,-1)
        return features

class BasicDict(object):
    def __init__(self):
        self.ntotal_folder=0
        self.whole_dict=[]
        self.ntotal_items=0
        self.whole_list=[]
        return
    def updatedict(self,newlist):
        # assert (type(newlist)==list or type(newlist)==np.array)
        self.whole_dict.append(newlist)
        self.ntotal_folder+=len(newlist)

        for item in newlist:
            self.whole_list.append(item)
            self.ntotal_items+=1

def pathtodir(path):
    if not os.path.exists(path):
        l=[]
        p = ""
        l = path.split("/")
        i = 0
        while i < len(l):
            p = p + l[i] + "/"
            i = i + 1
            if not os.path.exists(p):
                os.mkdir(p)

if __name__ == "__main__":
    print()
    # newmask_dir = '../../data/depth_ordering/FCIS_train2014_1229'
    # ListingObject=ListingDict(newmask_dir)
    # PlotingObject=Ploting()
    #
    # mask_dict=ListingObject.build_index(ListingObject.root)
    # print(mask_dict)
    # img=ListingObject.load_one_data(1)
    #
    # PlotingObject.showimg(img)
    #
    # #extract feature
    # pretrained=True
    # model=extractfModel(pretrained)
    # print(model)
    # x=model(Variable(img))
    # print x.size()



