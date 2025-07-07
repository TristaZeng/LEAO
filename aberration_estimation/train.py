import argparse
import os
import datetime

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from LoadData import MyDataset_triplet, MyDataset
import models
import utils


def evaluate(loader, model, verbose=False, writer=None, 
              config=None, EPOCH=0, tensorboard_image_writing=True):
    model.eval()
    metric_fn = utils.RMS
    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    
    for batch in pbar:
        inp = batch[0].cuda()
        label = batch[1].cuda()

        with torch.no_grad():

            _,zernike_coeff,_ = model(inp)
            
            if 'SH' in config.get('phase_loss'):
                pred = utils.zernike_poly_SH(zernike_coeff)
                zernike_coeff = label[:,0:17]
                gt = utils.zernike_poly_SH(zernike_coeff)
            elif 'ANSI' in config.get('phase_loss'):
                bs = zernike_coeff.shape[0]
                pred_mode1 = torch.unsqueeze(pred[:,0],dim=1)
                zernike_coeff = torch.concat([pred_mode1,torch.zeros([bs,1]).cuda(),pred[:,1:]],dim=1) # ANSI indexing has defocus as the 5th mode, we assert that it's 0 and do not estimate it by network
                pred = utils.zernike_poly_ANSI(zernike_coeff)
                zernike_coeff = label[:,0:18]
                gt = utils.zernike_poly_ANSI(zernike_coeff)
            pred = torch.squeeze(pred)
            gt = torch.squeeze(gt)
            
        res = metric_fn(pred, gt)
        res = torch.mean(res)
        val_res.add(res.item(), gt.shape[0])

        if verbose:
            pbar.set_description('val phase {:.4f}'.format(val_res.item()))

        pred[torch.isnan(pred)] = 0
        pred[torch.isinf(pred)] = 0
        pred = torch.squeeze(pred)
        minval_gt = torch.quantile(gt,0.001)
        minval_pred = torch.quantile(pred,0.001)
        minval = torch.min(minval_gt,minval_pred)
        gt = gt-minval
        pred = pred-minval
        maxval_gt = torch.quantile(gt, 0.9999) + 1e-6
        maxval_pred = torch.quantile(pred, 0.9999) + 1e-6
        maxval = torch.max(maxval_gt,maxval_pred)
        gt = gt/maxval
        pred = pred/maxval
        gt = gt.clamp_(0, 1)
        pred = pred.clamp_(0, 1)
        
        if 'writer' in locals().keys() and tensorboard_image_writing == True:
            tensorboard_image_writing = False
            writer.add_image('val_phase_gt', utils.cmap[ (gt*255).long().cpu()] ,dataformats='HWC',global_step=EPOCH)          
            writer.add_image('val_phase_pred', utils.cmap[ (pred*255).long().cpu()] ,dataformats='HWC',global_step=EPOCH)

    return val_res.item()


def prepare_training():
    # resume training an identical model
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True,args=config['model']['args']).cuda()
        
        if config.get('latent_loss') is not None:
            latent_loss_fn = eval(config.get('latent_loss'))
            latent_loss_fn.load_state_dict(sv_file['latent_loss_sd'])
        else:
            latent_loss_fn = None
        
        params = list(model.parameters())
        if latent_loss_fn is not None:
            params.extend(list(latent_loss_fn.parameters()))

        optimizer = utils.make_optimizer(
            params, sv_file['optimizer'], load_sd=True)
        
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            for _ in range(epoch_start - 1):
                lr_scheduler.step()

        if sv_file.get('min_val') is not None:
            min_val = sv_file['min_val']
        else:
            min_val=1e18
        if sv_file.get('best_epoch') is not None:
            best_epoch = sv_file['best_epoch']
        else:
            best_epoch = 1000
        if sv_file.get('n_epochs_wo_going_better') is not None:
            n_epochs_wo_going_better = sv_file['n_epochs_wo_going_better']
        else:
            n_epochs_wo_going_better = 0
        if sv_file.get('latent_start') is not None:
            latent_start = sv_file['latent_start']
        else:
            latent_start = True
            

    # train a partially different model based on part of the pre-trained weights
    elif config.get('load_pretrain') is not None:
        sv_file = torch.load(config['load_pretrain'])
        model = models.make(config['model']).cuda()
        model_dict=model.state_dict()
        pretrained_dict = {k: v for k, v in sv_file['model']['sd'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if config.get('latent_loss') is not None:
            latent_loss_fn = eval(config.get('latent_loss'))
            if sv_file['latent_loss_sd'] is not None:
                latent_loss_fn.load_state_dict(sv_file['latent_loss_sd'])
        else:
            latent_loss_fn = None

        params = list(model.parameters())
        if latent_loss_fn is not None:
            params.extend(list(latent_loss_fn.parameters()))
        optimizer = utils.make_optimizer(
            params, config['optimizer'])
        
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        
        min_val = 1e18
        best_epoch = epoch_start
        n_epochs_wo_going_better = 0
        latent_start = False

    else:
        model = models.make(config['model']).cuda()
        if config.get('latent_loss') is not None:
            latent_loss_fn = eval(config.get('latent_loss'))
        else:
            latent_loss_fn = None
        params = list(model.parameters())
        if latent_loss_fn is not None:
            params.extend(list(latent_loss_fn.parameters()))
        optimizer = utils.make_optimizer(
            params, config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

        min_val=1e18
        best_epoch = 1000
        n_epochs_wo_going_better = 0
        latent_start = False

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    log('model details: ')
    log(model)
    return model, latent_loss_fn, optimizer, epoch_start, lr_scheduler, min_val, best_epoch, n_epochs_wo_going_better, latent_start


def train(train_loader, model, latent_loss_fn, optimizer, writer, write_img, EPOCH):
    model.train()
    phase_loss_fn = eval(config.get('phase_loss'))
    recon_loss_fn = eval(config.get('recon_loss'))
    phase_loss_record = utils.Averager()
    recon_loss_record = utils.Averager()
    latent_loss_record = utils.Averager()
    batch_id = 0

    for batch in tqdm(train_loader, leave=False, desc='train'):
        anchor,positive,negative,pos_label,neg_label = batch
        inp = torch.concat([anchor,positive,negative],dim=0).cuda()
        label = torch.concat([pos_label,pos_label,neg_label],dim=0).cuda()
        bs = anchor.shape[0]

        latent,pred,recon = model(inp)
        
        optimizer.zero_grad()
        if batch_id%2==0:
            loss = recon_loss_fn(recon, inp)
            recon_loss_record.add(loss.item())
        else:        
            loss = phase_loss_fn(pred, label)
            phase_loss_record.add(loss.item())

        # if latent_loss_fn is not None and EPOCH>=start_epoch:
        if latent_loss_fn is not None:
            latent_anc_aber = latent[0:bs,0:32,...]
            latent_pos_aber = latent[bs:2*bs,0:32,...]
            latent_neg_aber = latent[2*bs:3*bs,0:32,...]
            latent_anc_struc = latent[0:bs,32:,...]
            latent_pos_struc = latent[2*bs:3*bs,32:,...]
            latent_neg_struc = latent[bs:2*bs,32:,...]
            latent_loss = latent_loss_fn(latent_anc_aber,latent_pos_aber,latent_neg_aber)+latent_loss_fn(latent_anc_struc,latent_pos_struc,latent_neg_struc)
            if latent_loss>0:
                weight2 = 1
                loss += latent_loss*weight2
            latent_loss_record.add(latent_loss.item())
        loss.backward()
        optimizer.step()
        batch_id += 1

        if write_img:
            if 'SH' in config.get('phase_loss'):
                pred = utils.zernike_poly_SH(pred[0,:])
                gt = utils.zernike_poly_SH(label[0,0:17])
            elif 'ANSI' in config.get('phase_loss'):
                bs = 1
                pred = torch.unsqueeze(pred[0,:],dim=0)
                pred_mode1 = torch.unsqueeze(pred[:,0],dim=1)
                zernike_coeff = torch.concat([pred_mode1,torch.zeros([bs,1]).cuda(),pred[:,1:]],dim=1) # ANSI indexing has defocus as the 5th mode, we assert that it's 0 and do not estimate it by network
                pred = utils.zernike_poly_ANSI(zernike_coeff)
                gt = utils.zernike_poly_ANSI(label[0,0:18])
            pred[torch.isnan(pred)] = 0
            pred[torch.isinf(pred)] = 0
            pred = torch.squeeze(pred)
            gt = torch.squeeze(gt)
            minval_gt = torch.quantile(gt,0.001)
            minval_pred = torch.quantile(pred,0.001)
            minval = torch.min(minval_gt,minval_pred)
            gt = gt-minval
            pred = pred-minval
            maxval_gt = torch.quantile(gt, 0.9999) + 1e-6
            maxval_pred = torch.quantile(pred, 0.9999) + 1e-6
            maxval = torch.max(maxval_gt,maxval_pred)
            gt = gt/maxval
            pred = pred/maxval
            gt = gt.clamp_(0, 1)
            pred = pred.clamp_(0, 1)
            writer.add_image('train_phase_gt', utils.cmap[ (gt*255).long().cpu()],dataformats='HWC',global_step=EPOCH)          
            writer.add_image('train_phase_pred', utils.cmap[ (pred*255).long().cpu()],dataformats='HWC',global_step=EPOCH)

            write_img = 0

    return phase_loss_record.item(),recon_loss_record.item(),latent_loss_record.item()


def main(config_, save_path):
    global config, log, writer, epoch
    config = config_
    log, writer = utils.set_save_path(save_path)
    config_path = os.path.join(save_path, 'config.yaml')
    with open(config_path,'a') as f:
        f.write('\n')
        f.write(str(datetime.datetime.now())+'\n')
        yaml.dump(config, f, sort_keys=False)
    with open(os.path.join(save_path, 'config_latest.yaml'),'w') as f:
        yaml.dump(config, f, sort_keys=False)

    ## prepare dataloaders
    args = config.get('train_dataset')
    args_val = config.get('val_dataset')
    nw = 0  # number of workers
    
    train_dataset = MyDataset_triplet(args)
    train_num = len(train_dataset)
    print('Using {} dataloader workers every process'.format(nw))
    train_sampler = eval(args['sampler'])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=train_sampler(range(train_num)),
                                               batch_size=args['batch_size'], 
                                               num_workers=nw)

    validate_dataset = MyDataset(args_val)
    val_num = len(validate_dataset)
    val_loader = torch.utils.data.DataLoader(validate_dataset,
                                             batch_size=1, shuffle=False,
                                             num_workers=nw, pin_memory=True)
    print("using {} stacks for training, {} stacks for validation.".format(train_num,
                                                                           val_num))

    model, latent_loss_fn, optimizer, epoch_start, lr_scheduler, min_val, best_epoch, n_epochs_wo_going_better, latent_start = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1 and args['batch_size'] > n_gpus:
        print('using data parallelism with gpu :'+os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        if config.get('latent_loss') is None:
            latent_loss_sd = None
        else:
            latent_loss_sd = latent_loss_fn.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch,
            'latent_loss_sd': latent_loss_sd,
            'min_val': min_val,
            'best_epoch': best_epoch,
            'n_epochs_wo_going_better': n_epochs_wo_going_better,
            'latent_start': latent_start
        }
        
        if (epoch_val is not None) and (epoch % epoch_val == 0):

            write_img=1
            
            val_res = evaluate(val_loader, model,
                writer=writer,
                config=config,
                EPOCH=epoch)

            log_info.append('val: RMS of phase={:.4f}'.format(val_res))
            writer.add_scalars('RMS', {'val': val_res}, epoch)
            if val_res < min_val:
                n_epochs_wo_going_better = 0
                min_val = val_res
                best_epoch = epoch
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))
            else:
                n_epochs_wo_going_better = n_epochs_wo_going_better+1

        else:
            write_img=0

        n_epochs_thresh = 10
        if n_epochs_wo_going_better>=n_epochs_thresh or latent_start: # after [n_epochs_thresh] epochs, the validation RMS still does not go down
            latent_start = True
            phase_loss,recon_loss,latent_loss = train(train_loader, model, latent_loss_fn, optimizer, writer, write_img, epoch)
        else:
            phase_loss,recon_loss,latent_loss = train(train_loader, model, None, optimizer, writer, write_img, epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: phase loss={:.4f},recon loss={:.4f},latent loss={:.4f}'.format(phase_loss,recon_loss,latent_loss))
        writer.add_scalars('train loss', {'recon loss': recon_loss,'phase loss': phase_loss,
                                          'latent loss': latent_loss}, epoch)
        

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
 

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='train.yaml')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_details = config.get('save_details')
    model_args = config.get('model')
    save_name = save_details['subfolder']
    save_name += '/' + save_details['type']
    save_name += '_' + model_args['name']
    save_name += '_' + save_details['suffix']
    if config.get('latent_loss') is not None:
        latent_descript = '_LatWeight1_NoNewMin4_10ep'
    else:
        latent_descript = ''
    save_path = os.path.join('./saved_models/', save_name+'_TriInp'+latent_descript)

    main(config, save_path)
