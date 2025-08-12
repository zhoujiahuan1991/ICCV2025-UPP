import os
import argparse
from pathlib import Path
import time
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        default= "cfgs/unify_modelnet_cls.yaml",  # pretask.yaml finetune_scan_objbg_cls.yaml finetune_modelnet_cls.yaml cfgs/unify_scan_objonly_cls.yaml
        # unify_modelnet_cls.yaml cfgs/unify_shapenetpart_seg.yaml unify_shapenet55_cls.yaml
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=False,
        help='whether to set deterministic options for CUDNN backend.')
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args 0.1noise0.9-24-1.2lidar1.5-64 segmentation-128res-nocls-nopromp-max3-nopropaga-48+128  mae-finetune-48+128 noisy-train-noisy-val randnoisy-train-clean-val
    parser.add_argument('--exp_name', type = str, default='retrain', help = 'experiment name') # mae-unify-peft-crop0.4-completion-denoise-2stage-fps
    parser.add_argument('--loss', type=str, default='cd2', help='loss name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    # prompter_bases/mae-shapenet-1.521.pth prompter_bases/recon-shapenet-1.578.pth prompter_bases/femae-shapenet-1.715.pth
    parser.add_argument('--ckpts', type = str, default="prompter_bases/recon-shapenet-1.578.pth", help = 'test used ckpt path') # pretrained_bases/ pretask-ckpt-cf0.983.pth mae_base.pth recon_base.pth femae-epoch-300.pth
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument('--incomplete_cropping', action='store_true', default=True, help = 'random cropping the point cloud to produce fragments')
    parser.add_argument('--incomplete_shape', action='store_true', default=True, help = 'cropping point clouds to incomplete shape or incomplete points')
    parser.add_argument('--shape_generate', action='store_true', default=True, help = 'cropping point clouds to incomplete shape or incomplete points')
    parser.add_argument('--cropping_rate', type = float, default=0.1, help = 'cropping rate for point cloud to produce fragments')
    parser.add_argument('--noise', action='store_true', default=True, help = 'add random Gaussion noise to the point cloud')
    parser.add_argument('--rectify', action='store_true', default=False, help = 'rectify point cloud with prompter')
    parser.add_argument('--noise_radius', type = float, default=0.8, help = 'radius of outlier noise points')
    parser.add_argument('--deviation', type = float, default=0.1, help = 'deviation of outlier noise points distribution')
    parser.add_argument('--noise_type', type=list, nargs='+', choices=['gaussian_noise', 'lidar_noise'], 
                        default=['gaussian_noise', 'lidar_noise'], help = 'adding additional noisy points or imitate a noisy distribution in original point cloud')
    parser.add_argument('--finetune_model', action='store_true', default=False, help = 'finetune modelnet with pretrained weight')
    parser.add_argument('--peft_model', action='store_true', default=True, help = 'parameter efficient finetune modelnet')
    parser.add_argument('--joint_optimization', type = int, default=250, help = 'starting epoch of joint optimization for prompters with downstream task modules')
    parser.add_argument('--normalize', action='store_true', default=False, help = 'normalize input point cloud into sphere')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help = 'vote acc')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')
    parser.add_argument(
        '--way', type=int, default=5)
    parser.add_argument(
        '--shot', type=int, default=10)
    parser.add_argument(
        '--fold', type=int, default=9)

    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')


    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.finetune_model:
        args.exp_name = 'finetune'+'-'+args.exp_name
    if args.peft_model:
        args.exp_name = 'peft'+'-'+args.exp_name
    if args.test:
        args.exp_name = 'test-' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '-' +args.mode
    if args.ckpts:
        args.experiment_path = os.path.join('./experiments', Path(args.config).stem, os.path.splitext(os.path.basename(args.ckpts))[0], args.exp_name)
    else:
        args.experiment_path = os.path.join('./experiments', Path(args.config).stem, 'plain-network', args.exp_name)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.experiment_path = os.path.join(args.experiment_path, timestamp)
    if args.ckpts:
        args.tfboard_path = os.path.join('./experiments', 'TFBoard', Path(args.config).stem, os.path.splitext(os.path.basename(args.ckpts))[0], args.exp_name)
    else:
        args.tfboard_path = os.path.join('./experiments', 'TFBoard', Path(args.config).stem, 'plain-network', args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

