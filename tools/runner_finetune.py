import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import os
import ipdb
import numpy as np
import cv2
from pointnet2_ops import pointnet2_utils
from utils.config import cfg_from_yaml_file
from tqdm import tqdm
import random
import shutil
from tools.data_augment import scale_translate, rotate, jitter, normalize, test_transforms
from tools.runner import Acc_Metric
from tqdm import tqdm
Gaussian_noise_number = 24
lidar_noise_number = 48


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    (_, test_dataloader) = builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    shutil.copy('tools/runner_finetune.py', args.experiment_path)
    shutil.copy('models/Point_MAE_cp.py', args.experiment_path)
    # pointr_config = cfg_from_yaml_file("./pointr_cfgs/ShapeNet55_models/PoinTr.yaml")
    # pointr_model = builder.model_builder(pointr_config.model)
    # builder.load_model(pointr_model, "./pretrained_bases/pointr_training_from_scratch_c55_best.pth", logger = logger)
    # pointr_model.to(args.local_rank)
    # pointr_model.eval()
    # pointr_model = nn.DataParallel(pointr_model).cuda()    

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    print_log("Require gradient parameters: ", logger = logger)
    # peft_list = ['cls_pos', 'cls_token', 'cls_head_finetune']
    # for name, param in base_model.named_parameters():
    #     if misc.peft_detect(name, peft_list) : 
    #         print_log(name, logger = logger)
    #         param.requires_grad_(True)
    #     else:
    #         param.requires_grad_(False)

    from utils.misc import summary_parameters
    summary_parameters(base_model, logger=logger)
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(base_model, (2048,3), as_strings=True)
    print(f"FLOPs: {flops}, Params: {params}")

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    metrics = validate(base_model, test_dataloader, 0, val_writer, args, config, logger=logger, pointr_model=None) # pointr_model        
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(train_dataloader)):
            num_iter += 1
            n_itr = epoch * n_batches + batch_idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            # online cropping 
            if config.noisy_train and args.incomplete_cropping:
                dataset_name = config.dataset.train._base_.NAME
                if dataset_name in ['ShapeNet','ModelNet']:
                    gt = points                                                                                                                #0.25
                    partial, _ = misc.seprate_point_cloud(gt, config.dataset.train._base_.N_POINTS, int(config.dataset.train._base_.N_POINTS * 0.5), sample_points=npoints, incomplete_shape=args.incomplete_shape)
                    visualize = False
                    if visualize == True:
                        task = 'cropping'
                        os.makedirs(f'./visualization/{task}', exist_ok=True)
                        np.save(f'./visualization/{task}/partial-{batch_idx}', partial.detach().cpu().numpy())
                        np.save(f'./visualization/{task}/crop-{batch_idx}', _.detach().cpu().numpy())
                        np.save(f'./visualization/{task}/gt-{batch_idx}', gt.detach().cpu().numpy())
                    partial = partial.cuda()
                    points = partial
                    del gt
                    torch.cuda.empty_cache()
                elif dataset_name in ['ScanObjectNN']:
                    gt = points
                    partial, _ = misc.seprate_point_cloud(gt, config.dataset.train._base_.N_POINTS, int(config.dataset.train._base_.N_POINTS * 0.25), sample_points=npoints, incomplete_shape=args.incomplete_shape)
                    partial = partial.cuda()
                    points = partial
                    del gt
                    torch.cuda.empty_cache()
            else:
                fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                del fps_idx
                torch.cuda.empty_cache()
            
            # Normalize the point clouds
            if args.normalize:
                points = normalize(points)

            if config.noisy_train and args.noise:
                if 'lidar_noise' in args.noise_type:
                    B, P, C = points.shape
                    lidar_noise = misc.lidar_noise(points, lidar_noise_number, low=1.2, scale=1.5)
                    points = torch.concat([points, lidar_noise], dim=1)
                if 'gaussian_noise' in args.noise_type:
                    B, P, C = points.shape
                    Gaussian_noise = misc.gaussian_noise([B, Gaussian_noise_number, C], loc=0., scale=0.1, shell_radius=0.9)
                    Gaussian_noise = torch.tensor(Gaussian_noise, dtype=torch.float32).cuda()
                    points  = torch.concat([points, Gaussian_noise], dim=1)
                # Jitter noise
                # points = jitter(points)
                
                visualize = False
                if visualize == True:
                    task = 'cropping'
                    os.makedirs(f'./visualization/{task}', exist_ok=True)
                    np.save(f'./visualization/{task}/points-gaussian-{batch_idx}', points.detach().cpu().numpy())
            
            # from ptflops import get_model_complexity_info
            # flops, params = get_model_complexity_info(base_model, (2048,3), as_strings=True)
            # print(f"FLOPs: {flops}, Params: {params}")
            # return None
                    
            # PoinTr for point completion
            # coarse_points, dense_points = pointr_model(points)
            # dense_points = train_transforms(dense_points)
            
            if config.data_augmentation == 'rotate':
                points = rotate(points)
            elif config.data_augmentation == 'scale-translate':
                points = scale_translate(points)
            ret = base_model(points) # dense_points points

            loss, acc = base_model.module.get_loss_acc(ret, label)
            _loss = loss
            _loss.backward()
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger, pointr_model=None) # pointr_model

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, pointr_model=None):
    base_model.eval()
    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader)):
            points = data[0].cuda()
            label = data[1].cuda()
            # online cropping 
            if config.noisy_validate:
                gt = points
                partial, _ = misc.seprate_point_cloud(gt, config.dataset.val._base_.N_POINTS, int(config.dataset.val._base_.N_POINTS * 0.25), fixed_points = torch.Tensor([1,1,1]), sample_points=npoints, incomplete_shape=True)
                partial = partial.cuda()
                points = partial
                del gt
                torch.cuda.empty_cache()

                # Lidar noise
                B, P, C = points.shape
                lidar_noise = misc.lidar_noise(points, lidar_noise_number, low=1.3, scale=1.6,) #deterministic=True
                points = torch.concat([points, lidar_noise], dim=1)
                # Gaussian noise
                # B, P, C = points.shape
                # Gaussian_noise = misc.gaussian_noise([B, 5, C], loc=0., scale=0.1, shell_radius=0.8, ) #deterministic=True
                # Gaussian_noise = torch.tensor(Gaussian_noise, dtype=torch.float32).cuda()
                # points  = torch.concat([points, Gaussian_noise], dim=1)
                # Jitter noise
                # points = jitter(points)
                
            else:
                points, _ = misc.fps(points, npoints)
                torch.cuda.empty_cache()
            
            # Normalize the point clouds
            if args.normalize:
                points = normalize(points)

            if pointr_model is not None:
                coarse_points, dense_points = pointr_model(points)
                visualize = False
                if visualize == True:
                    task = 'complement'
                    os.makedirs(f'./visualization/{task}', exist_ok=True)

                    np.save(f'./visualization/{task}/coarse-{idx}', coarse_points.detach().cpu().numpy())
                    np.save(f'./visualization/{task}/dense-{idx}', dense_points.detach().cpu().numpy())
                    np.save(f'./visualization/{task}/points-{idx}', points.detach().cpu().numpy())
                points = dense_points

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)



def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    if args.use_gpu:
        base_model.to(args.local_rank)
     
    test(base_model, test_dataloader, args, config, logger=logger)
    
def test(base_model, test_dataloader, args, config, logger = None,cp_model=None):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    #npoints = 1024

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader)):
            points = data[0].cuda()
            label = data[1].cuda()
    
            points = misc.fps(points, npoints)
            points = points[0]
            logits = base_model(points)
            
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)

                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)
    
    return acc
