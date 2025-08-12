import os
os.environ['OMP_NUM_THREADS']='5'
os.environ['MKL_NUM_THREADS']='5'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune
from tools import finetune_test_run_net, module_tune_test_run_net
from tools import module_run_net as module_tune
from tools import pretask_run_net
from tools import finetune_seg_run_net as finetune_seg
from tools import unify_seg_run_net as unify_seg
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import torch
from tensorboardX import SummaryWriter

def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    log_file = os.path.join(args.experiment_path, f'result.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer

    train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
    val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
    # config
    config = get_config(args, logger = logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size 
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size 
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs 
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs 
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()
        
    # ensemble(args, config, train_writer, val_writer)
    # return None
    if args.test:
        if args.finetune_model:
            finetune_test_run_net(args, config)
        elif args.peft_model:
            module_tune_test_run_net(args, config)
    elif config.task == 'classification':
        if args.finetune_model:
            # full fine-tuning
            # ensemble(args, config, train_writer, val_writer)
            print('finetuning starts!')
            finetune(args, config, train_writer, val_writer)
        elif args.peft_model:
            # GAPrompt tuning
            print('module tuning starts!')
            module_tune(args, config, train_writer, val_writer)
    elif config.task == 'segmentation':
        if args.finetune_model:
            # full fine-tuning
            print('finetuning starts!')
            finetune_seg(args, config, train_writer, val_writer)
        elif args.peft_model:
            # GAPrompt tuning
            print('module tuning starts!')
            unify_seg(args, config, train_writer, val_writer)
    elif config.task == 'pretask':
        pretask_run_net(args, config, train_writer, val_writer)
    
    elif config.task == 'pretrain':
        pretrain(args, config, train_writer, val_writer)

if __name__ == '__main__':
    main()
