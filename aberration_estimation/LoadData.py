import mat73
import torch
import torchvision
import scipy.io as sio
from utils import prctile_norm
from torch.utils.data import Dataset
import numpy as np
import imageio
import glob
import cv2
import random
import copy

def load_raw_data(inp_path,desired_r,Nnum,cut,minmax_norm,adjustIntParam):
    input_image = np.array(imageio.mimread(inp_path,memtest=False)).astype(np.float64)
    
    # do not read angles outside of the desired radius to save memory
    inp_angles_are_circle = True
    midNA_flag = False # is the input 2pSAM data?
    num_angles = input_image.shape[0]
    if num_angles==49:
        raw_data_r=4
    elif num_angles==81:
        raw_data_r=5
    elif num_angles==113:
        raw_data_r=6
    elif num_angles==317:
        raw_data_r=10
    elif num_angles==13 or num_angles==35:
        midNA_flag = True
    else:
        inp_angles_are_circle = False
    
    if midNA_flag:
        img = input_image
    else:
        if inp_angles_are_circle:
            if desired_r<raw_data_r:
                img = []
                cnt = 0
                for u in range(Nnum):
                    for v in range(Nnum):
                        if (u-Nnum//2)**2+(v-Nnum//2)**2<=raw_data_r**2:
                            if (u-Nnum//2)**2+(v-Nnum//2)**2<=desired_r**2:
                                img.append(input_image[cnt,:,:])
                            cnt = cnt+1
            else:
                img = input_image
        else:
            img = []
            for u in range(Nnum):
                for v in range(Nnum):
                    if (u-Nnum//2)**2+(v-Nnum//2)**2<=desired_r**2:
                        img.append(input_image[u*Nnum+v,:,:])
    input_image = np.array(img)

    # cut edges
    input_image = input_image[:,cut:input_image.shape[1]-cut,cut:input_image.shape[2]-cut]
    
    # normalize
    if minmax_norm:
        input_image = prctile_norm(input_image,0.1)
    else:
        input_image = input_image*adjustIntParam / 65535.0

    return input_image

def angle_sampling(input_image,inds):
    input_image = [input_image[angle,:,:] for angle in inds]
    input_image = np.array(input_image)
    return input_image

def crop_data(input_image,cropped_size,mask_flag,idx=None,idy=None):
    h = input_image.shape[1]
    w = input_image.shape[2]

    if cropped_size<max(h,w):
        half_size = cropped_size//2
        if idx is None or idy is None:
            if mask_flag:
                center_angle = input_image[input_image.shape[0]//2,:,:]
                center_angle = prctile_norm(center_angle)
                fg = cv2.GaussianBlur(center_angle,ksize=[0,0],sigmaX=10,sigmaY=10)
                bg = cv2.GaussianBlur(center_angle,ksize=[0,0],sigmaX=50,sigmaY=50)
                mask = fg-bg
                mask[:half_size+1,:] = 0
                mask[-(half_size+1):,:] = 0
                mask[:,:half_size+1] = 0
                mask[:,-(half_size+1):] = 0
                mask[mask>=0.01] = 1
                mask[mask!=1] = 0
                nonzero_ids_x,nonzero_ids_y = np.nonzero(mask)
                i = np.random.randint(0,len(nonzero_ids_x))
                idx = nonzero_ids_x[i]
                idy = nonzero_ids_y[i]
            else:
                # skip masking to unburden CPU
                idx = np.random.randint(half_size,h-half_size)
                idy = np.random.randint(half_size,w-half_size)
        input_image = input_image[:,idx-half_size:idx+half_size,idy-half_size:idy+half_size]
    
    return input_image, idx, idy

def load_gt(gt_path,zern_index):
    try:
        cur_gt = sio.loadmat(gt_path)
    except NotImplementedError:
        cur_gt = mat73.loadmat(gt_path)
    if zern_index == 'SH':
        cur_gt = np.squeeze(cur_gt['c'])[4:21].astype(np.float32)
    elif zern_index == 'ANSI':
        cur_gt = np.squeeze(cur_gt['c'])[3:21].astype(np.float32)
    else:
        ValueError('Wrong Zernike indexing method!')
    cur_gt = torch.tensor(cur_gt)
    return cur_gt

class MyDataset(Dataset):  
    def __init__(self, args): 
        self.desired_r = args.get('desired_r')
        self.angle_num = args.get('angle_num')
        self.Nnum = args.get('Nnum')
        self.cropped_size = args['patch_size']
        self.mask_flag = args['mask_flag']
        self.minmax_norm = args['minmax_norm']
        self.adjustIntParam = args['adjustIntParam']
        self.cut = args['cut']
        self.load_num = args['load_num']
        inp_path = args['inp_path']
        gt_path = args.get('gt_path')
        self.zern_index = args.get('zern_index')
        self.ang_order = args.get('ang_order')
        # PIL.Image or ndarray(H x W x D) to tensor (D x H x W)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        
        # get all WDF paths
        self.inp_path = []
        for i in range(len(inp_path)):
            cur_path = glob.glob(inp_path[i]+'/*tif')
            print('data path {}: {}'.format(inp_path[i],len(cur_path)))
            self.inp_path.extend(cur_path)
        self.inp_path.sort()

        # load WDFs
        self.inp_data = []
        for i in range(self.load_num):
            cur_data = load_raw_data(self.inp_path[i],self.desired_r,self.Nnum,self.cut,self.minmax_norm,self.adjustIntParam)
            self.inp_data.append(cur_data)
            print('data {} loaded'.format(self.inp_path[i]))

        # get all gt paths
        if gt_path is not None:
            self.gt_path = []
            for i in range(len(gt_path)):
                cur_path = glob.glob(gt_path[i]+'/*mat')
                self.gt_path.extend(cur_path) 
            self.gt_path.sort()

            # load GTs
            self.gt = []
            for i in range(self.load_num):
                cur_gt = load_gt(self.gt_path[i],self.zern_index)
                self.gt.append(cur_gt)
                print('data {} loaded'.format(self.gt_path[i]))
        else:
            self.gt_path = None
        

    def __len__(self):
        return len(self.inp_path)  

    def __getitem__(self, index):
        
        ########################################################
        ################ get input WDF #########################
        ########################################################
        if index < self.load_num:
            input_image = self.inp_data[index]
        else:
            input_image = load_raw_data(self.inp_path[index],self.desired_r,self.Nnum,self.cut,self.minmax_norm,self.adjustIntParam)
        if self.ang_order is None:
            inds = random.sample(range(0, input_image.shape[0]), self.angle_num)
            inds.sort()
        else:
            inds = [x-1 for x in self.ang_order]
        input_image = angle_sampling(input_image,inds)
        input_image,idx,idy = crop_data(input_image,self.cropped_size,self.mask_flag)
        input_image = np.transpose(input_image, (1, 2, 0))
        input = self.transforms(input_image)
        input = input.float()

        ########################################################
        ################ get Zernike label #####################
        ########################################################
        if self.gt_path is not None:
            if index<self.load_num:
                label = self.gt[index]
            else:
                label = load_gt(self.gt_path[index],self.zern_index)
            return input, label
        else:
            return input


class MyDataset_triplet(Dataset):  
    def __init__(self, args): 
        self.desired_r = args.get('desired_r')
        self.angle_num = args.get('angle_num')
        self.Nnum = args.get('Nnum')
        self.cropped_size = args['patch_size']
        self.mask_flag = args['mask_flag']
        self.minmax_norm = args['minmax_norm']
        self.adjustIntParam = args['adjustIntParam']
        self.load_num = args['load_num']
        inp_path = args['inp_path']
        self.zern_index = args.get('zern_index')
        gt_path = args.get('gt_path')
        max_anc_vol_id = args.get('max_anc_vol_id')
        max_anc_psf_id = args.get('max_anc_psf_id')
        self.cut = args['cut']
        self.ang_order = args.get('ang_order')
        # PIL.Image or ndarray(H x W x D) to tensor (D x H x W)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        # get all WDF and GT paths
        self.inp_path = []
        all_path = []
        self.anc_path = []
        if gt_path is not None:
            self.anc_gt_path = []
        for i in range(len(inp_path)):
            tmp = glob.glob(inp_path[i]+'/*tif')
            if gt_path is not None:
                tmp_gt = glob.glob(gt_path[i]+'/*mat')
            cur_path = []
            cur_anc_path = []
            cur_anc_gt_path = []
            if max_anc_vol_id is not None:
                for k in range(len(tmp)):
                    cur_file = tmp[k]
                    vol_id = cur_file.rindex('vol')
                    vol_id = cur_file[vol_id+3:-4]
                    vol_id = int(vol_id)
                    if vol_id<=max_anc_vol_id:
                        if max_anc_psf_id is not None:
                            psf_id = cur_file.rindex('psf')
                            psf_id = cur_file[psf_id+3:psf_id+6]
                            psf_id = int(psf_id)
                            if psf_id<=max_anc_psf_id:
                                cur_anc_path.append(cur_file)
                                if gt_path is not None:
                                    cur_gt_file = cur_file.replace(inp_path[i],gt_path[i])
                                    cur_gt_file = cur_gt_file.replace('tif','mat')
                                    cur_anc_gt_path.append(cur_gt_file)
                        else:
                            cur_anc_path.append(cur_file)
                            if gt_path is not None:
                                cur_gt_file = cur_file.replace(inp_path[i],gt_path[i])
                                cur_gt_file = cur_gt_file.replace('tif','mat')
                                cur_anc_gt_path.append(cur_gt_file)
            elif max_anc_psf_id is not None:
                for k in range(len(tmp)):
                    cur_file = tmp[k]
                    psf_id = cur_file.rindex('psf')
                    psf_id = cur_file[psf_id+3:psf_id+6]
                    psf_id = int(psf_id)
                    if psf_id<=max_anc_psf_id:
                        cur_anc_path.append(cur_file)
                        if gt_path is not None:
                            cur_gt_file = cur_file.replace(inp_path[i],gt_path[i])
                            cur_gt_file = cur_gt_file.replace('tif','mat')
                            cur_anc_gt_path.append(cur_gt_file)
            else:
                cur_anc_path = tmp
                if gt_path is not None:
                    cur_anc_gt_path = tmp_gt
            print('anchor path {}: {}'.format(inp_path[i],len(cur_anc_path)))
            self.anc_path.extend(cur_anc_path)
            if gt_path is not None:
                print('anchor GT path {}: {}'.format(gt_path[i],len(cur_anc_gt_path)))
                self.anc_gt_path.extend(cur_anc_gt_path)
            cur_path = tmp
            print('all data path {}: {}'.format(inp_path[i],len(cur_path)))
            all_path.extend(cur_path)
        # add necessary paths to ensure triplet input based on anchor path
        self.anc_path.sort()
        self.anc_gt_path.sort()
        self.inp_path = copy.deepcopy(self.anc_path)
        if gt_path is not None:
            self.gt_path = self.anc_gt_path
        if len(self.anc_path)<len(all_path):
            vol_list = []
            psf_list = []
            for path_id in range(len(self.anc_path)):
                anc_path = self.anc_path[path_id]
                anc_psf_id = anc_path.rindex('psf')  
                anc_psf_id = anc_path[anc_psf_id+3:anc_psf_id+6]
                anc_vol_id = anc_path.rindex('vol')
                anc_vol_id = anc_path[anc_vol_id+3:-4]
                neg_paths = [path for path in all_path if ('psf'+anc_psf_id not in path) and ('vol'+anc_vol_id in path)]
                common_elements = [item for item in neg_paths if item in self.inp_path]
                if len(common_elements)==0:
                    neg_path = neg_paths[0]
                    psf_id = neg_path.rindex('psf')
                    neg_path_folder = neg_path[0:psf_id-1]
                    neg_path_main_folder = neg_path_folder.rindex('/')
                    neg_path_main_folder = neg_path_folder[0:neg_path_main_folder]
                    psf_id = neg_path[psf_id+3:psf_id+6]
                    psf_list.append(int(psf_id))
                    self.inp_path.extend([neg_path])
                    if gt_path is not None:
                        neg_gt_path = neg_path.replace(neg_path_folder,neg_path_main_folder+'/gt')
                        neg_gt_path = neg_gt_path.replace('tif','mat')
                        self.gt_path.extend([neg_gt_path])
                pos_paths = [path for path in all_path if ('psf'+anc_psf_id in path) and (path != anc_path)]
                common_elements = [item for item in pos_paths if item in self.inp_path]
                if len(common_elements)==0:
                    pos_path = pos_paths[0]
                    vol_id = pos_path.rindex('vol')
                    psf_id = pos_path.rindex('psf')
                    pos_path_folder = pos_path[0:psf_id-1]
                    pos_path_main_folder = pos_path_folder.rindex('/')
                    pos_path_main_folder = pos_path_folder[0:pos_path_main_folder]
                    vol_id = pos_path[vol_id+3:vol_id+6]
                    vol_list.append(int(vol_id))
                    self.inp_path.extend([pos_path])
                    if gt_path is not None:
                        pos_gt_path = pos_path.replace(pos_path_folder,pos_path_main_folder+'/gt')
                        pos_gt_path = pos_gt_path.replace('tif','mat')
                        self.gt_path.extend([pos_gt_path])
            vol_list = list(set(vol_list))
            psf_list = list(set(psf_list))
            print('paths used in training: {}, added {} vols and {} psfs aside from given max vol/psf number'.format(len(self.inp_path),len(vol_list),len(psf_list)))

        # load WDFs
        self.inp_data = []
        for i in range(self.load_num):
            cur_data = load_raw_data(self.inp_path[i],self.desired_r,self.Nnum,self.cut,self.minmax_norm,self.adjustIntParam)
            self.inp_data.append(cur_data)
            print('data {} loaded'.format(self.inp_path[i]))

        
        # load GTs
        self.gt = []
        for i in range(self.load_num):
            cur_gt = load_gt(self.gt_path[i],self.zern_index)
            self.gt.append(cur_gt)
            print('data {} loaded'.format(self.gt_path[i]))


    def __len__(self):
        return len(self.anc_path)  

    def __getitem__(self, index):
        
        ########################################################
        ################ get input WDF #########################
        ########################################################
        anc_path = self.anc_path[index]
        psf_id = anc_path.rindex('psf')
        psf_id = anc_path[psf_id+3:psf_id+6]
        vol_id = anc_path.rindex('vol')
        vol_id = anc_path[vol_id+3:-4]
        neg_paths = [path for path in self.inp_path if ('psf'+psf_id not in path) and ('vol'+vol_id in path)]
        neg_path = random.sample(neg_paths,1)
        neg_path = neg_path[0]
        neg_ind = self.inp_path.index(neg_path)
        pos_paths = [path for path in self.inp_path if ('psf'+psf_id in path) and (path != anc_path)]
        pos_path = random.sample(pos_paths,1)
        pos_path = pos_path[0]
        pos_ind = self.inp_path.index(pos_path)
        
        if index < self.load_num:
            anchor = self.inp_data[index]
        else:
            anchor = load_raw_data(self.inp_path[index],self.desired_r,self.Nnum,self.cut,self.minmax_norm,self.adjustIntParam)
        if pos_ind < self.load_num:
            positive = self.inp_data[pos_ind]
        else:
            positive = load_raw_data(self.inp_path[pos_ind],self.desired_r,self.Nnum,self.cut,self.minmax_norm,self.adjustIntParam)
        if neg_ind < self.load_num:
            negative = self.inp_data[neg_ind]
        else:
            negative = load_raw_data(self.inp_path[neg_ind],self.desired_r,self.Nnum,self.cut,self.minmax_norm,self.adjustIntParam)
            

        if self.ang_order is None:
            inds = random.sample(range(0, anchor.shape[0]), self.angle_num)
            inds.sort()
        else:
            inds = [x-1 for x in self.ang_order]
        anchor = angle_sampling(anchor,inds)
        positive = angle_sampling(positive,inds)
        negative = angle_sampling(negative,inds)
        anchor,idx,idy = crop_data(anchor,self.cropped_size,self.mask_flag)
        negative,idx,idy = crop_data(negative,self.cropped_size,self.mask_flag,idx,idy)
        positive,idx,idy = crop_data(positive,self.cropped_size,self.mask_flag)
        
       
        anchor = np.transpose(anchor, (1, 2, 0))
        positive = np.transpose(positive, (1, 2, 0))
        negative = np.transpose(negative, (1, 2, 0))
        
        anchor = self.transforms(anchor)
        anchor = anchor.float()
        positive = self.transforms(positive)
        positive = positive.float()
        negative = self.transforms(negative)
        negative = negative.float()


        ########################################################
        ################ get Zernike label #####################
        ########################################################
        if index < self.load_num:
            pos_label = self.gt[index]
        else:
            pos_label = load_gt(self.gt_path[index],self.zern_index)
        if neg_ind < self.load_num:
            neg_label = self.gt[neg_ind]
        else:
            neg_label = load_gt(self.gt_path[neg_ind],self.zern_index)

        return anchor,positive,negative,pos_label,neg_label
        
  
