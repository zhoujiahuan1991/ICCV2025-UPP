# 环境
# conda activate cwy

# # UPP based on Point-MAE in ModelNet40
python main.py  --peft_model  --config  cfgs/unify_modelnet_cls.yaml  --ckpts  prompter_bases/mae-modelnet-2.349.pth

# UPP based on ReCon in ModelNet40
python main.py  --peft_model  --config  cfgs/unify_modelnet_cls.yaml  --ckpts  prompter_bases/recon-modelnet-2.168.pth

# UPP based on Point-FEMAE in ModelNet40
python main.py  --peft_model  --config  cfgs/unify_modelnet_cls.yaml  --ckpts  prompter_bases/femae-modelnet-2.235.pth



# UPP based on Point-MAE in ShapeNet55
python main.py  --peft_model  --config cfgs/unify_shapenet55_cls.yaml  --ckpts  prompter_bases/mae-shapenet-1.521.pth

# UPP based on ReCon in ShapeNet55
python main.py  --peft_model  --config  cfgs/unify_shapenet55_cls.yaml  --ckpts  prompter_bases/recon-shapenet-1.578.pth

# UPP based on Point-FEMAE in ShapeNet55
python main.py  --peft_model  --config  cfgs/unify_shapenet55_cls.yaml  --ckpts  prompter_bases/femae-shapenet-1.715.pth



# UPP based on Point-FEMAE in ScanObjectNN
python main.py  --peft_model  --config  experiments/unify_scan_objonly_cls/femae-objonly-2.963/peft-incompl-train-clean-val-r-91.39/20250306_012158/config.yaml  --ckpts  experiments/unify_scan_objonly_cls/femae-objonly-2.963/peft-incompl-train-clean-val-r-91.39/20250306_012158/ckpt-best.pth

