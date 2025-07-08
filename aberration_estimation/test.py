import argparse
import os
import gc

import yaml
import glob
import torch
from tqdm import tqdm
import tifffile as tiff
import numpy as np

from LoadData import MyDataset
import models
import utils
import xlwt
import xlrd
from xlutils.copy import copy
import matplotlib.pyplot as plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def write_matrix(res,sheet_name,epoch_detail,save_test_suffix,xls_path):
    xls_path = xls_path+'metrics_'+epoch_detail+save_test_suffix+'.xls'
    res = res.cpu().numpy()
    if os.path.isfile(xls_path):
        workbook_old = xlrd.open_workbook(xls_path)
        workbook = copy(workbook_old)
        workbook.save(xls_path[:-4]+'_backup.xls')
        if sheet_name in workbook_old.sheet_names():
            booksheet = workbook.get_sheet(sheet_name) 
        else:
            booksheet=workbook.add_sheet(sheet_name, cell_overwrite_ok=True)
    else:
        workbook = xlwt.Workbook(encoding='utf-8')
        booksheet=workbook.add_sheet(sheet_name, cell_overwrite_ok=True)
    booksheet.write(0,0,'row2 is avg, row 3 is std.')
    booksheet.write(1,0,str(np.mean(res)))
    booksheet.write(2,0,str(np.std(res)))
    for j,row in enumerate(res):
        if len(row.shape)==0:
            booksheet.write(j+4,0,str(row))
        else:
            for i,col in enumerate(row):
                booksheet.write(j+4,i,str(col))
    workbook.save(xls_path)

def test(loader, mymodel, epoch_detail, save_test_suffix, eval_type=None, save_path=None, 
              config=None):
    mymodel.eval()

    if eval_type is None:
        metric_fn = utils.sigma
    else:
        metric_fn = eval(eval_type)
    
    pbar = tqdm(loader, leave=False, desc='test')
    phase_list = None
    gt_phase_list = None
    for batch in pbar:
        if len(batch)>1:
            inp = batch[0].cuda()
        else:
            inp = batch.cuda()
        with torch.no_grad():
            _,zernike_coeff,_ = mymodel(inp)
            
        zernike_coeff = zernike_coeff.cpu()
        if 'SH' in config.get('phase_loss'):
            phase = utils.zernike_poly_SH(zernike_coeff)
        elif 'ANSI' in config.get('phase_loss'):
            bs = zernike_coeff.shape[0]
            zernike_coeff = torch.unsqueeze(zernike_coeff[0,:],dim=0)
            zernike_coeff_mode1 = torch.unsqueeze(zernike_coeff[:,0],dim=1)
            zernike_coeff = torch.concat([zernike_coeff_mode1,torch.zeros([bs,1]),zernike_coeff[:,1:]],dim=1) # ANSI indexing has defocus as the 5th mode, we assert that it's 0 and do not estimate it by network
            phase = utils.zernike_poly_ANSI(zernike_coeff)
            
        if len(batch)>1:
            zernike_coeff_gt = batch[1]
            if 'SH' in config.get('phase_loss'):
                gt_phase = utils.zernike_poly_SH(zernike_coeff_gt[:,0:17])
            elif 'ANSI' in config.get('phase_loss'):
                gt_phase = utils.zernike_poly_ANSI(zernike_coeff_gt[:,0:18])
        
        if phase_list is None:
            phase_list = phase
        else:
            phase_list = torch.concat([phase_list,phase],dim=0)
        if len(batch)>1:
            if gt_phase_list is None:
                gt_phase_list = gt_phase
            else:
                gt_phase_list = torch.concat([gt_phase_list,gt_phase],dim=0)

        torch.cuda.empty_cache()
        gc.collect()
    
    if len(batch)>1:
        res = metric_fn(phase_list, gt_phase_list)
        avg_res = torch.mean(res)
        write_matrix(res,eval_type,epoch_detail,save_test_suffix,save_path)
    else:
        avg_res = 0
    
    return phase_list,gt_phase_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./aberration_estimation/demo_model/test.yaml')
    parser.add_argument('--epoch_detail', default='epoch-best')
    parser.add_argument('--save_test_suffix', default='')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args_test = config.get('test_dataset')
    nw = 2  # number of workers
    test_dataset = MyDataset(args_test)
    bs = 1
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=bs, shuffle=False,
                                            num_workers=nw, pin_memory=True)
    print("using {} stacks for test.".format(len(test_dataset)))

    args_model = config.get('model')
    test_paths = args_test['inp_path']
    file_paths = []
    for i in range(len(test_paths)): 
        cur_path = glob.glob(test_paths[i]+'*tif')
        file_paths.extend(cur_path)
    
    
    config_name = args.config.split('/')[-1]
    load_mymodel_path = args.config[:-len(config_name)]
    model_spec = torch.load(load_mymodel_path+args.epoch_detail+'.pth')['model']
    mymodel = models.make(model_spec, load_sd=True, args=args_model['args']).cuda()

    save_path = load_mymodel_path+'/TestResult'+args.save_test_suffix+'/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    phase, gt_phase= test(test_loader, mymodel, args.epoch_detail, args.save_test_suffix,
        eval_type='utils.sigma', save_path=save_path, config=config)
    
    phase = phase.cpu().numpy()
    if gt_phase is not None:
        gt_phase = gt_phase.cpu().numpy()
    for file_id in range(phase.shape[0]):
        file_name = file_paths[file_id]
        file_name = file_name.split(os.sep)[-1]
        file_name = file_name[:-4]
        tiff.imwrite(save_path+file_name+'_pred_phase_'+args.epoch_detail+'.tif',phase[file_id,...])
        if gt_phase is not None:
            tiff.imwrite(save_path+file_name+'_gt_phase.tif',gt_phase[file_id,...])

        peak_val = np.max(np.abs(gt_phase[file_id,...]))
        fig, axs = plt.subplots(1, 2, figsize=[9, 6])
        ax1, ax2 = axs.flat
        im1 = ax1.imshow(gt_phase[file_id, ...], cmap='jet', vmin=-peak_val, vmax=peak_val)
        ax1.set_title('GT aberration wavefront')
        ax1.axis('off')  
        im2 = ax2.imshow(phase[file_id, ...], cmap='jet', vmin=-peak_val, vmax=peak_val)
        ax2.set_title('LEAO aberration wavefront')
        ax2.axis('off')  
        cbar = fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.9)
        tick_vals = [-peak_val, 0, peak_val]
        tick_labels = [f'{-peak_val/(2*np.pi):.2f}λ', '0', f'{peak_val/(2*np.pi):.2f}λ']
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('RMS')
        fig.savefig(save_path + file_name + '_pred_and_gt_comparison.png', dpi=300, bbox_inches='tight')

   