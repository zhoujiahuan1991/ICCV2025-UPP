# 环境
# conda activate cwy

# # UPP based on Point-MAE in ModelNet40
python main.py  --test  --config  experiments/unify_modelnet_cls/mae-modelnet-2.349/peft-noisy-train-clean-test/20250305_140504-92.99/config.yaml  --ckpts  experiments/unify_modelnet_cls/mae-modelnet-2.349/peft-noisy-train-clean-test/20250305_140504-92.99/ckpt-best.pth

# UPP based on ReCon in ModelNet40
python main.py  --test  --config  experiments/unify_modelnet_cls/recon-modelnet-2.168/peft-shape-noisy-train-clean-test/20250305_144505/config.yaml  --ckpts  experiments/unify_modelnet_cls/recon-modelnet-2.168/peft-shape-noisy-train-clean-test/20250305_144505/ckpt-best.pth

# UPP based on Point-FEMAE in ModelNet40
python main.py  --test  --config  experiments/unify_modelnet_cls/femae-modelnet-2.235/peft-24gaus64lidar-train-clean-val/20250305_140643/config.yaml  --ckpts  experiments/unify_modelnet_cls/femae-modelnet-2.235/peft-24gaus64lidar-train-clean-val/20250305_140643/ckpt-best.pth



# UPP based on Point-MAE in ShapeNet55
python main.py  --test  --config experiments/unify_shapenet55_cls/mae-shapenet-1.521/peft-noisy-train-clean-val-st-joint/20250718_233924-90.40/config.yaml  --ckpts  experiments/unify_shapenet55_cls/mae-shapenet-1.521/peft-noisy-train-clean-val-st-joint/20250718_233924-90.40/ckpt-best.pth

# UPP based on ReCon in ShapeNet55
python main.py  --test  --config  experiments/unify_shapenet55_cls/recon-shapenet-1.578/peft-noisy-train-clean-val-st-joint/20250722_170254/config.yaml  --ckpts  experiments/unify_shapenet55_cls/recon-shapenet-1.578/peft-noisy-train-clean-val-st-joint/20250722_170254/ckpt-best.pth

# UPP based on Point-FEMAE in ShapeNet55
python main.py  --test  --config  experiments/unify_shapenet55_cls/femae-shapenet-1.715/peft-noisy-train-clean-val-st-joint/20250722_001144/config.yaml  --ckpts  experiments/unify_shapenet55_cls/femae-shapenet-1.715/peft-noisy-train-clean-val-st-joint/20250722_001144/ckpt-best.pth



# UPP based on Point-FEMAE in ScanObjectNN
python main.py  --test  --config  experiments/unify_scan_objonly_cls/femae-objonly-2.963/peft-incompl-train-clean-val-r-91.39/20250306_012158/config.yaml  --ckpts  experiments/unify_scan_objonly_cls/femae-objonly-2.963/peft-incompl-train-clean-val-r-91.39/20250306_012158/ckpt-best.pth

