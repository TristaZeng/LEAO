import os
import time
import shutil
import scipy.io as sio
import mat73

import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter

Tensor = torch.tensor

cmap = [[0 ,0 ,0],
 [0,0,255],
 [255,0,255],
 [255,0,0],
 [255,255,0],
 [255,255,255]]
cmap = torch.tensor(cmap,dtype=torch.float32).unsqueeze(0).unsqueeze(1)
cmap = torch.nn.functional.interpolate(cmap, size=[256,3],mode='bilinear',align_corners=True)
cmap = cmap.squeeze()
cmap = cmap.type(torch.uint8)

def zernike_poly_SH(c,device=None):
    if len(c.shape)==1:
        c = torch.unsqueeze(c,dim=0)
    bs = c.shape[0]
    load_path = './aberration_estimation/zernike_polynomials/'
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    phase = torch.zeros([bs,199,199]).to(device)
    for bs_ind in range(bs):
        for i in range(17):
            cur_polynomial = mat73.loadmat(load_path+'%02d'%(i+1)+'.mat')
            cur_polynomial = cur_polynomial['calcu_phase_k']
            cur_polynomial = torch.from_numpy(cur_polynomial).to(device)
            phase[bs_ind,:,:] = phase[bs_ind,:,:]+cur_polynomial*c[bs_ind,i]
    return phase
        
def zernike_poly_ANSI(c,device=None):
    if len(c.shape)==1:
        c = torch.unsqueeze(c,dim=0)
    bs = c.shape[0]
    load_path = './aberration_estimation/zernike_polynomials_ANSI/'
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    phase = torch.zeros([bs,199,199]).to(device)
    for bs_ind in range(bs):
        for i in range(18):
            cur_polynomial = sio.loadmat(load_path+'%02d'%(i+1)+'.mat')
            cur_polynomial = cur_polynomial['zern_poly']
            cur_polynomial = torch.from_numpy(cur_polynomial).to(device)
            phase[bs_ind,:,:] = phase[bs_ind,:,:]+cur_polynomial*c[bs_ind,i]
    return phase

def zernike2phasemap_loss_SH(predict,label,device=None):
    mse = nn.MSELoss()
    
    zernike_coeff = label[:,0:17]
    gt = zernike_poly_SH(zernike_coeff,device)
    
    zernike_coeff = predict
    predict = zernike_poly_SH(zernike_coeff,device)
    ps = predict.shape[-1]
    
    X,Y = torch.meshgrid(torch.arange(ps)-ps//2,torch.arange(ps)-ps//2)
    return mse(predict[:,X**2+Y**2<=(ps//2)**2],gt[:,X**2+Y**2<=(ps//2)**2])

def zernike2phasemap_loss_ANSI(predict,label,device=None):
    mse = nn.MSELoss()
    
    zernike_coeff = label[:,0:18]
    gt = zernike_poly_ANSI(zernike_coeff,device)
    
    bs = predict.shape[0]
    pred_mode1 = torch.unsqueeze(predict[:,0],dim=1)
    zernike_coeff = torch.concat([pred_mode1,torch.zeros([bs,1]).cuda(),predict[:,1:]],dim=1) # ANSI indexing has defocus as the 5th mode, we assert that it's 0 and do not estimate it by network
    predict = zernike_poly_ANSI(zernike_coeff,device)
    ps = predict.shape[-1]
    
    X,Y = torch.meshgrid(torch.arange(ps)-ps//2,torch.arange(ps)-ps//2)
    return mse(predict[:,X**2+Y**2<=(ps//2)**2],gt[:,X**2+Y**2<=(ps//2)**2])

def prctile_norm(x, min_prc=0.1, max_prc=100.0):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        if not np.isnan(v):
            self.v = (self.v * self.n + v * n) / (self.n + n)
            self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)
        
_log_path = None
def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_') and input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            print(1)
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def RMS(gt,pred):
    diff = gt-pred
    if len(diff.shape)<3:
        diff = torch.unsqueeze(diff,axis=0)
    rms_list = []
    for bs_ind in range(diff.shape[0]):
        w=[]
        for i in range(diff.shape[1]):
            for j in range(diff.shape[2]):
                if ((i-diff.shape[1]//2)**2+(j-diff.shape[2]//2)**2) <= (diff.shape[1]//2)**2:
                    w.append(diff[bs_ind,i,j])
        rms=torch.std(torch.tensor(w))
        rms_list.append(rms)
    return torch.tensor(rms_list)

def sigma(gt,pred):
    diff = gt-pred
    if len(diff.shape)<3:
        diff = torch.unsqueeze(diff,axis=0)
    if len(gt.shape)<3:
        gt = torch.unsqueeze(gt,axis=0)
    sigma_list = []
    for bs_ind in range(gt.shape[0]):
        w=[]
        for i in range(diff.shape[1]):
            for j in range(diff.shape[2]):
                if ((i-diff.shape[1]//2)**2+(j-diff.shape[2]//2)**2) <= (diff.shape[1]//2)**2:
                    w.append(diff[bs_ind,i,j])
        rms=torch.std(torch.tensor(w))
        
        peak = torch.max(gt[bs_ind,:,:])
        valley = torch.min(gt[bs_ind,:,:])
        sigma_list.append(rms/(peak-valley))
    return torch.tensor(sigma_list)