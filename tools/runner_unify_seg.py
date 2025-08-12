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
from datasets import data_transforms
import cv2
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from utils.config import cfg_from_yaml_file
from tqdm import tqdm
import sys

rotate = transforms.Compose(
    [   
        data_transforms.PointcloudRotate()
    ]
)

scale_translate = transforms.Compose(
    [   
        data_transforms.PointcloudScaleAndTranslate(scale_low=0.9, scale_high=1.1, translate_range=0.1)
    ]
)

jitter = transforms.Compose(
    [   
        data_transforms.PointcloudJitter(std=0.01, clip=0.05)
    ]
)

add_noise = transforms.Compose(
    [   
        data_transforms.AddNoise(noise_std_min=0.005, noise_std_max=0.03)
    ]
)

normalize = transforms.Compose(
    [   
        data_transforms.NormalizeUnitSphere()
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

def random_scale_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def run_net(args, config, train_writer=None, val_writer=None):
    def log_string(str):
        logger.info(str)
        print(str)

    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    (_, test_dataloader) = builder.dataset_builder(args, config.dataset.val)
    num_classes = 16
    num_part = 50

    # build model
    classifier = builder.model_builder(config.model)
    criterion = classifier.get_loss
    classifier.apply(inplace_relu)
    # print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0
    

    if args.ckpts is not None:
        classifier.load_model_from_ckpt(args.ckpts)
    else:
        print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        classifier.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        classifier = nn.DataParallel(classifier).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(classifier, config)

    print_log("Require gradient parameters: ", logger = logger)
    peft_list = ['downstream_adapter', 
                 'downstream_prompts',
                 #'bnorm', #'pos_embed',
                 'label_conv', 'propagation_0', 'seg_head', 'propagation_1']
    # 'coarse_pred', 'increase_dim', 'mask_prompter'
    for name, param in classifier.named_parameters():
        if misc.peft_detect(name, peft_list) : 
            print_log(name, logger = logger)
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    from utils.misc import summary_parameters
    summary_parameters(classifier, logger=logger)

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    test_metrics = validate(logger, num_part, classifier, test_dataloader, num_classes, config)
    import shutil
    shutil.copy('models/Point_MAE_unify_segment.py', str(args.experiment_path))
    shutil.copy('tools/runner_unify_seg.py', str(args.experiment_path))
    # training
    classifier.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        classifier.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints

        mean_correct = []
        print_log('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, config.max_epoch), logger = logger)
        '''Adjust learning rate and BN momentum'''
        classifier = classifier.train()
        loss_batch = []
        num_iter = 0
        '''learning one epoch'''
        for batch_idx, (points, label, target) in enumerate(tqdm(train_dataloader)):
            num_iter += 1
            n_itr = epoch * n_batches + batch_idx
            data_time.update(time.time() - batch_start_time)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            cls_label = to_categorical(label, num_classes)
            
            # Normalize the point clouds
            if args.normalize:
                points = normalize(points)

            if config.data_augmentation == 'rotate':
                points = rotate(points)
            elif config.data_augmentation == 'scale-translate':
                # points = scale_translate(points)
                points = points.cpu().data.numpy()
                points[:, :, 0:3] = random_scale_pointcloud(points[:, :, 0:3])
                points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points).cuda()
            gt_points = points

            gt = points
            partial, _ = misc.seprate_point_cloud(gt, config.dataset.train._base_.N_POINTS, int(config.dataset.train._base_.N_POINTS * 0.25), sample_points=npoints, incomplete_shape=args.incomplete_shape)
            partial = partial.cuda()
            points = partial

            B, P, C = points.shape
            
            Gaussian_noise = misc.gaussian_noise([B, 24, C], loc=0., scale=args.deviation, shell_radius=args.noise_radius)
            Gaussian_noise = torch.tensor(Gaussian_noise, dtype=torch.float32).cuda()
            points  = torch.concat([points, Gaussian_noise], dim=1)
            lidar_noise = misc.lidar_noise(points, 64, low=1.2, scale=1.5)
            points = torch.concat([points, lidar_noise], dim=1)

            seg_pred = classifier(points, cls_label, label_points=gt_points, completion_prompt=config.noisy_train, denoise=config.noisy_train, point_num=P, denoise_after=False, batch_idx=batch_idx)
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            acc = pred_choice.eq(target.data).cpu().sum()/(config.total_bs * config.dataset.train._base_.N_POINTS)
            mean_correct.append(acc.item())
            loss = criterion(seg_pred, target)
            loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                loss_batch.append(loss.detach().cpu())
                classifier.zero_grad()

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
            # break

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        
        train_instance_acc = np.mean(mean_correct)
        loss1 = np.mean(loss_batch)
        print_log('Train accuracy is: %.5f' % (train_instance_acc*100), logger = logger)
        print_log('Train loss: %.5f' % loss1, logger = logger)
        print_log('lr: %.6f' % optimizer.param_groups[0]['lr'], logger = logger)

        if epoch % args.val_freq == 0:
            test_metrics = validate(logger, num_part, classifier, test_dataloader, num_classes, config)
            
            # Save ckeckpoints
            if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
                best_metrics = test_metrics
                builder.save_checkpoint(classifier, optimizer, epoch, test_metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
            if test_metrics['class_avg_iou'] > best_class_avg_iou:
                best_class_avg_iou = test_metrics['class_avg_iou']
            if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
                best_inctance_avg_iou = test_metrics['inctance_avg_iou']
            print_log('Best accuracy is: %.5f ' % (best_acc*100), logger = logger)
            print_log('Best class avg mIOU is: %.5f ' % (best_class_avg_iou*100), logger = logger)
            print_log('Best inctance avg mIOU is: %.5f ' % (best_inctance_avg_iou*100), logger = logger)

        builder.save_checkpoint(classifier, optimizer, epoch, test_metrics, best_metrics, 'ckpt-last', args, logger = logger) 

        print_log('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy']*100, test_metrics['class_avg_iou']*100, test_metrics['inctance_avg_iou']*100), logger = logger)
        global_epoch += 1



def validate(logger, num_part, classifier, testDataLoader, num_classes, config, noise_test=False):
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()

        for batch_idx, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().contiguous().cuda(), label.long().cuda(), target.long().cuda()
            torch.cuda.empty_cache()
            cls_label = to_categorical(label, num_classes)
            seg_pred = classifier(points, cls_label, completion_prompt=False, denoise=False, point_num=NUM_POINT, batch_idx=batch_idx)
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
        for cat in sorted(shape_ious.keys()):
            print_log('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]), logger = logger)
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
    return test_metrics