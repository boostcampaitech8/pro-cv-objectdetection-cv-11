import os
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

import wandb; wandb.login()

NUM_FOLDS = 5
DATASET_ROOT = "../../dataset/"
FOLD_ROOT = "../../dataset/sgk-fold/"

# custom setting
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

config_path = "configs/my_model/faster-rcnn_r101_fpn_1x_recycle.py"
save_model_name = "faster-rcnn_r101_fpn_1x_recycle"

# kfold training
for FOLD in range(NUM_FOLDS):
    print(f"##### Training FOLD {FOLD} #####")

    train_ann = f"sgk-fold/fold{FOLD}_train.json"
    val_ann  = f"sgk-fold/fold{FOLD}_val.json"

    cfg = Config.fromfile(config_path)
    register_all_modules(init_default_scope=True)
    cfg.default_scope = "mmdet"

    for ds_key, ann_file in zip(["train_dataloader", "val_dataloader"], [train_ann, val_ann]):
        if ds_key not in cfg:
            continue
        ds = cfg[ds_key]["dataset"] if "dataset" in cfg[ds_key] else cfg[ds_key]
        ds.metainfo = dict(classes=classes)
        ds.data_root = DATASET_ROOT
        ds.data_prefix = dict(img="")
        ds.ann_file = ann_file

    cfg.train_dataloader.batch_size = 4
    cfg.train_dataloader.num_workers = max(2, cfg.train_dataloader.get("num_workers", 2))

    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = max(2, cfg.val_dataloader.get("num_workers", 2))

    cfg.train_cfg = cfg.get("train_cfg", {})
    cfg.train_cfg.max_epochs = 12
    cfg.train_cfg["val_interval"] = 1

    if "val_evaluator" in cfg:
        cfg.val_evaluator.ann_file = os.path.join(DATASET_ROOT, val_ann)
        cfg.val_evaluator.metric = "bbox"
    else:
        cfg.val_evaluator = dict(
            type="CocoMetric",
            ann_file=os.path.join(DATASET_ROOT, val_ann),
            metric="bbox"
        )
    
    cfg.device = "cuda"
    cfg.gpu_ids = [0]
    cfg.randomness = dict(seed=2025, deterministic=False)
    cfg.work_dir = f"./work_dirs/{save_model_name}/fold_{FOLD}/"

    cfg.model.roi_head.bbox_head.num_classes = len(classes)
    cfg.optim_wrapper = {**cfg.get("optim_wrapper", {}), "clip_grad": dict(max_norm=35, norm_type=2)}

    cfg.default_hooks["checkpoint"]["max_keep_ckpts"] = 3
    cfg.default_hooks["checkpoint"]["interval"] = 1

    wandb.init(
        project='cv_11_OD',
        entity='cv_11',
        name=cfg.vis_backends[1].init_kwargs['name'],
    )

    ## train start
    runner = Runner.from_cfg(cfg)
    runner.train()

print("Finished!")
