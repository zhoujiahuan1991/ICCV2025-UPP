import subprocess
import os
from tqdm import tqdm
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "3" 

# datasets = ['cifar100']
def run_command(exp_name="hyper", config='',ckpts="",  PEFT=True):
        command=[
                "python",
                "main.py",
                "--config", config,
                "--exp_name", exp_name,
                "--ckpts", ckpts,
            ]
        if PEFT:
            command.append("--peft")
        else:
            command.append("--finetune_model")
        subprocess.run(command, env=env)

# run_command(exp_name="peft-femae-objonly-r-prompter", config="cfgs/pretask_scan_objonly.yaml", ckpts="pretrained_bases/femae-epoch-300.pth", PEFT=True)
# run_command(exp_name="peft-femae-shapenet-st-prompter", config="cfgs/pretask_shapenet.yaml", ckpts="pretrained_bases/femae-epoch-300.pth", PEFT=True)


# run_command(exp_name="peft-recon-objonly-r-prompter", config="cfgs/pretask_scan_objonly.yaml", ckpts="pretrained_bases/recon_base.pth", PEFT=True)
# run_command(exp_name="peft-recon-shapenet-st-prompter", config="cfgs/pretask_shapenet.yaml", ckpts="pretrained_bases/recon_base.pth", PEFT=True)

# run_command(exp_name="20lidar-train-20lidar-val", config="cfgs/finetune_modelnet_cls.yaml", ckpts="pretrained_bases/mae_base.pth", PEFT=False)

# run_command(exp_name="mae-shapenetpart-noaug-prompter", config="cfgs/pretask_shapenetpart.yaml", ckpts="pretrained_bases/mae_base.pth", PEFT=True)
# run_command(exp_name="recon-shapenetpart-noaug-prompter", config="cfgs/pretask_shapenetpart.yaml", ckpts="pretrained_bases/recon_base.pth", PEFT=True)
# run_command(exp_name="femae-shapenetpart-noaug-prompter", config="cfgs/pretask_shapenetpart.yaml", ckpts="pretrained_bases/femae-epoch-300.pth", PEFT=True)


# run_command(exp_name="20gaus-train-20lidar-val", config="cfgs/unify_modelnet_cls.yaml", ckpts="prompter_bases/mae-modelnet-2.349.pth", PEFT=True)
# run_command(exp_name="24gaus64lidar-train-clean-val", config="cfgs/unify_modelnet_cls.yaml", ckpts="prompter_bases/femae-modelnet-2.235.pth", PEFT=True)
# run_command(exp_name="20gaus-train-20lidar-val", config="cfgs/unify_modelnet_cls.yaml", ckpts="prompter_bases/recon-modelnet-2.168.pth", PEFT=True)
# run_command(exp_name="noisy-train-noisy-val", config="cfgs/finetune_modelnet_cls.yaml", ckpts="pretrained_bases/recon_base.pth", PEFT=False)
# run_command(exp_name="noisy-train-noisy-val", config="cfgs/finetune_modelnet_cls.yaml", ckpts="pretrained_bases/femae-epoch-300.pth", PEFT=False)

run_command(exp_name="0.1incompl-train-clean-val-r", config="cfgs/unify_scan_objonly_cls.yaml", ckpts="prompter_bases/femae-objonly-2.963.pth", PEFT=True)



