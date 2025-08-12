import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from tqdm import tqdm
import shutil
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


rotate = transforms.Compose(
    [   
        data_transforms.PointcloudRotate()
    ]
)

scale_translate = transforms.Compose(
    [   
        data_transforms.PointcloudScaleAndTranslate()
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

class CD_Metric:
    def __init__(self, chanfer_distance = 0.):
        if type(chanfer_distance).__name__ == 'dict':
            self.chanfer_distance = chanfer_distance['CD']
        else:
            self.chanfer_distance = chanfer_distance

    def better_than(self, other):
        if self.chanfer_distance < other.chanfer_distance:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['chanfer_distance'] = self.chanfer_distance
        return _dict


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    shutil.copy('tools/runner_pretask.py', args.experiment_path)
    shutil.copy('models/Point_MAE_pretask_dev.py', args.experiment_path)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = CD_Metric(1000.)
    metrics = CD_Metric(1000.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = None)
        best_metrics = CD_Metric(best_metric)
    if args.ckpts is not None:
        base_model.load_model_from_ckpt(args.ckpts, logger = logger)
    else:
        print_log('Training from scratch', logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.peft_model:
        print_log("Require gradient parameters: ", logger = logger)
        peft_list = ['rectify_adapter', 'downstream_adapter', 'pretask_adapter', 
                     'rectify_adapter1', 'downstream_adapter1', 'pretask_adapter1', 
                     'rectify_prompts', 'downstream_prompts', 'pretask_prompts',
                     'coarse_pred', 'increase_dim', 'mask_token', 'dense_pred',
                     'rectify_prompter', 'shape_pred', 'coarse_pred', 'predict_token_generator', 'increase_dim',
                     'mask_prompter', 'shape_pred', 'mask_token_generator']
        for name, param in base_model.named_parameters():
            if misc.peft_detect(name, peft_list) : 
                # print_log(name, logger = logger)
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    from utils.misc import summary_parameters
    summary_parameters(base_model, logger=logger)

    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(base_model, (2052, 3), as_strings=True)
    # print(f"FLOPs: {flops}, Params: {params}")

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    ChamferDisL1 = ChamferDistanceL1().cuda()
    ChamferDisL2 = ChamferDistanceL2().cuda()

    # train and val
    # metrics = validate(base_model, test_dataloader, 0, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger, in_detail=False)
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
        losses = AverageMeter(['CroppingCoarseLoss', 'CroppingDenseLoss' ,'DenseLoss', 'NoiseLoss', 'Recall']) # 'CenterCoarseLoss'

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(train_dataloader)):
            num_iter += 1
            n_itr = epoch * n_batches + batch_idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name in ['PCN', 'Completion3D', 'Projected_ShapeNet']:
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if batch_idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune
                    points = partial.cuda()
            elif dataset_name in ['ShapeNet', 'ScanObjectNN', 'ScanObjectNN_hardest', 'ModelNet', 'PartNormalDataset']:
                gt = data[0].cuda()
                if config.data_augmentation == 'rotate':
                    gt = rotate(gt)
                elif config.data_augmentation == 'scale-translate':
                    gt = scale_translate(gt)   
                # gt = train_transforms(gt) # rotation data augmentation
                partial, cropping = misc.seprate_point_cloud(gt, config.dataset.train._base_.N_POINTS, 
                                                             [int(config.dataset.train._base_.N_POINTS * 0.15) , int( config.dataset.train._base_.N_POINTS * 0.5)], 
                                                             fixed_points = None, 
                                                             sample_points = npoints, 
                                                             incomplete_shape=True)
                points = partial.cuda()
                cropping = cropping.cuda()

            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            # partial = train_transforms(partial)
            # if epoch < config.max_epoch//2:
            #     coarse_point_cloud, dense_point_cloud, noise_loss = base_model(partial, require_gaussion_noise=True, denoise=False)
            # else:
                # coarse_point_cloud, dense_point_cloud, noise_loss = base_model(partial, require_gaussion_noise=True, denoise=True)
            if args.noise:
                B, P, C = points.shape
                if 'gaussian_noise' in args.noise_type:
                    import random
                    Gaussian_noise = misc.gaussian_noise([B, 20, C], loc=0., scale=0.2, shell_radius=(random.random()+2.0)/3)
                    Gaussian_noise = torch.tensor(Gaussian_noise, dtype=torch.float32).cuda()
                    points = torch.concat([points, Gaussian_noise], dim=1)
                    visualize = False
                    if visualize == True:
                        task = 'cropping'
                        os.makedirs(f'./visualization/{task}', exist_ok=True)
                        np.save(f'./visualization/{task}/partial-{batch_idx}', partial.detach().cpu().numpy())
                        np.save(f'./visualization/{task}/crop-{batch_idx}', cropping.detach().cpu().numpy())
                        np.save(f'./visualization/{task}/gt-{batch_idx}', gt.detach().cpu().numpy())
                        np.save(f'./visualization/{task}/points-gaussian-{batch_idx}', points.detach().cpu().numpy())
                if 'lidar_noise' in args.noise_type:
                    lidar_noise = misc.lidar_noise(points, 32, low=1.2, scale=1.5)
                    points = torch.concat([points, lidar_noise], dim=1)
                predict_coarse_center, rebuild_points, noise_loss, recall = base_model(points, point_num=npoints, train_with_gaussian=True, predict_center_num=16)
            else:
                noise_loss = 0
                recall = 1
                predict_coarse_center, rebuild_points = base_model(points, point_num=npoints, train_with_gaussian=False, predict_center_num=16)
            gt_center, _ = misc.fps(gt, 128)
            partial_center, _ = misc.fps(partial, 128)
            # coarse_center_loss = ChamferDisL1(coarse_point_cloud, gt_center)
            cropping_coarse_loss = ChamferDisL1(predict_coarse_center, cropping)
            # coarse_loss = ChamferDisL1(torch.concat([partial_center, predict_coarse_center], dim=1), gt)
            cropping_dense_loss = ChamferDisL1(rebuild_points, cropping)
            dense_loss = ChamferDisL1(torch.concat([partial, rebuild_points], dim=1), gt)

            loss = cropping_coarse_loss + cropping_dense_loss + dense_loss + noise_loss# * (1.0-recall) #* (1-epoch/config.max_epoch) # coarse_center_loss +
            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([cropping_coarse_loss.item()*1000, cropping_dense_loss.item()*1000, dense_loss.item()*1000, noise_loss.item()*1000]) # coarse_center_loss.item()*1000, 
            else:
                if noise_loss != 0:
                    losses.update([cropping_coarse_loss.item()*1000, cropping_dense_loss.item()*1000, dense_loss.item()*1000, noise_loss.item()*1000, recall.item()*100]) # coarse_center_loss.item()*1000, 
                else:
                    losses.update([cropping_coarse_loss.item()*1000, cropping_dense_loss.item()*1000, dense_loss.item()*1000, 0, 0])

            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                # train_writer.add_scalar('Loss/Batch/CenterCoarseLoss', coarse_center_loss.item(), n_itr)
                # train_writer.add_scalar('Loss/Batch/CoarseLoss', coarse_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/DenseLoss', dense_loss.item(), n_itr)
                if noise_loss != 0:
                    train_writer.add_scalar('Loss/Batch/NoiseLoss', noise_loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/Recall', recall.item()*100, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if batch_idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                            [f'{losses.items[i]}: %.4f' % l for i, l in enumerate(losses.val())], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)
        if epoch == 20:
            print_log("Require gradient parameters: ", logger = logger)
            peft_list = ['downstream_adapter', 'pretask_adapter', 
                        'downstream_adapter1', 'pretask_adapter1', 
                        'downstream_prompts', 'pretask_prompts',
                        'coarse_pred', 'dense_pred', 'mask_token', 
                        'shape_pred', 'coarse_pred', 'predict_token_generator', 'increase_dim',
                        'mask_prompter', 'shape_pred', 'mask_token_generator']
            for name, param in base_model.named_parameters():
                if misc.peft_detect(name, peft_list) : 
                    print_log(name, logger = logger)
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        if epoch % args.val_freq == 0 : #and epoch != 0
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)
        
            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)    
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger, in_detail=False)


def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None, in_detail=False, mode='easy'):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    crop_ratio = {
        'easy': 1/4,
        'median' :1/2,
        'hard':3/4
        }

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader)):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name in ['PCN', 'Projected_ShapeNet']:
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[-1]

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                # test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name in ['ShapeNet', 'ScanObjectNN', 'ScanObjectNN_hardest', 'ModelNet', 'PartNormalDataset']:
                gt = data[0].cuda()
                if in_detail:
                    choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                                torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                else:
                    choice = [torch.Tensor([1,1,1])]
                num_crop = int(npoints * crop_ratio[mode])
                for item in choice:
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: downsample the input
                    partial, _ = misc.fps(partial, 1024)
                    partial_center, _ = misc.fps(partial, 128)
                    predict_coarse_center, rebuild_points = base_model(partial, train_with_gaussian=False, predict_center_num=16)
                    coarse_points = torch.concat([partial_center, predict_coarse_center], dim=1)
                    dense_points = torch.concat([partial, rebuild_points], dim=1)
                    
                    # coarse_points = misc.fps(coarse_points, 128)[0]
                    # dense_points = misc.fps(dense_points, 1536)[0]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    if in_detail:
                        _metrics = Metrics.get(dense_points ,gt)
                        if taxonomy_id not in category_metrics:
                            category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                        category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[-1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if in_detail:
            for _,v in category_metrics.items():
                test_metrics.update(v.avg())
            print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)
            if args.distributed:
                torch.cuda.synchronize()
    if in_detail:
        # Print testing results
        shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
        print_log('============================ TEST RESULTS ============================',logger=logger)
        msg = ''
        msg += 'Taxonomy\t'
        msg += '#Sample\t'
        for metric in test_metrics.items:
            msg += metric + '\t'
        msg += '#ModelName\t'
        print_log(msg, logger=logger)
        for taxonomy_id in category_metrics:
            msg = ''
            msg += (taxonomy_id + '\t')
            msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
            for value in category_metrics[taxonomy_id].avg():
                msg += '%.3f \t' % value
            msg += shapenet_dict[taxonomy_id] + '\t'
            print_log(msg, logger=logger)
        msg = ''
        msg += 'Overall\t\t'
        for value in test_metrics.avg():
            msg += '%.3f \t' % value
        print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/SparseCDL2', test_losses.avg(1), epoch)
        val_writer.add_scalar('Loss/Epoch/DenseCDL2', test_losses.avg(3), epoch)
        if in_detail:
            for i, metric in enumerate(test_metrics.items):
                val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)
    print_log('[Epoch %d] validate dense Chamfer Distance L2: %.5f' % (epoch, test_losses.avg(3)), logger=logger)
    return CD_Metric(test_losses.avg(3))
