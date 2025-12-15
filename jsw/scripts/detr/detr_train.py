# 모듈 import
import numpy as np
import pandas as pd
from collections import Counter
from IPython.display import display

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules

import wandb
wandb.login()

api = wandb.Api()
me = api.viewer

print("Username:", me.username)
print("Teams:", me.teams)

base_path = "/data/ephemeral/home/jsw_pro-cv-objectdetection-cv-11/base_codes/mmdetection"
home_path = "/data/ephemeral/home"
work_path = "/data/ephemeral/home/jsw_pro-cv-objectdetection-cv-11/jsw"
wandb_name = "aug_test"
# custom 설정
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

root = f"{home_path}/dataset"
train_ann = "train.json"
test_ann  = "test.json"

# config file 들고오기
cfg = Config.fromfile(f"{base_path}/configs/detr/detr_r50_8xb2-150e_coco.py")
register_all_modules(init_default_scope=True)
cfg.default_scope = "mmdet"

print("cfg file:", f"{base_path}/configs/detr/detr_r50_8xb2-150e_coco.py")
print("cfg.load_from:", cfg.get('load_from', None))


# dataset config 수정
for ds_key in ["train_dataloader", "test_dataloader"]:
    if ds_key not in cfg:
        continue
    ds = cfg[ds_key]["dataset"] if "dataset" in cfg[ds_key] else cfg[ds_key]
    ds.metainfo = dict(classes=classes)
    ds.data_root = root
    ds.ann_file = train_ann if ds_key == "train_dataloader" else test_ann
    ds.data_prefix = dict(img="")

cfg.train_dataloader.batch_size = 4
cfg.train_dataloader.num_workers = max(2, cfg.train_dataloader.get("num_workers", 2))
#cfg.train_dataloader.dataset.pipeline[3]["scale"] = (512, 512)

cfg.test_dataloader.batch_size = 1
cfg.test_dataloader.num_workers = max(2, cfg.test_dataloader.get("num_workers", 2))
#cfg.test_dataloader.dataset.pipeline[2]["scale"] = (512, 512)

## custom aug (승재님 aug 코드 mmdetection wrapping 버전)
# === Albumentations DETR pipeline override ===

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=20,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10
    ),
    dict(
        type='Resize',
        scale=(1024, 1024),  # 원본 크기
        keep_ratio=True
    ),
    # dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackDetInputs')
]
cfg.train_dataloader.dataset.pipeline = train_pipeline

cfg.test_dataloader.dataset.pipeline = [    
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        scale=(1024, 1024),  # 원본 크기
        keep_ratio=True
    ),
    dict(type='PackDetInputs'),
]

# validate 비활성화
# for k in ("val_dataloader", "val_evaluator", "val_cfg", "val_loop"):
#     cfg.pop(k, None)

cfg.val_cfg = dict(type="ValLoop")
cfg.test_cfg = dict(type="TestLoop")

cfg.train_cfg = cfg.get("train_cfg", {})
cfg.train_cfg["val_interval"] = 0

# 학습 config 수정
cfg.device = "cuda"
cfg.gpu_ids = [0]
cfg.randomness = dict(seed=2025, deterministic=False)
cfg.work_dir = f"{work_path}/work_dirs/detr_r50_trash"

cfg.model.bbox_head.num_classes = len(classes)

cfg.optim_wrapper['clip_grad'] = dict(max_norm=0.1, norm_type=2)
# optimizer
cfg.optim_wrapper.optimizer.lr = 2.5e-5

cfg.train_cfg.max_epochs = 50
cfg.train_cfg.val_interval = 1
cfg.default_hooks["checkpoint"]["max_keep_ckpts"] = 10
cfg.default_hooks["checkpoint"]["interval"] = 5

# dataset summarization 확인
train_ds_cfg = cfg.train_dataloader.dataset
train_ds = DATASETS.build(train_ds_cfg)

# def summarize_dataset(ds):
#     ds.full_init()
#     num_images = len(ds)
#     classes = list(ds.metainfo.get("classes", []))

#     counts = Counter()
#     for i in range(num_images):
#         info = ds.get_data_info(i)
#         for inst in info.get("instances", []):
#             lbl = inst.get("bbox_label", None)
#             if lbl is not None:
#                 counts[lbl] += 1

#     df = pd.DataFrame({
#         "category": [f"{i} [{c}]" for i, c in enumerate(classes)],
#         "count": [counts.get(i, 0) for i in range(len(classes))]
#     })

#     print(f"\n [Info] CocoDataset Train dataset with number of images {num_images}, and instance counts:")
#     display(df)

# summarize_dataset(train_ds)

train_ds_cfg = cfg.train_dataloader.dataset
train_ds = DATASETS.build(train_ds_cfg)

# # dataset 크기 확인
# print(f"[DEBUG] Train dataset length: {len(train_ds)}")

# # batch_size 확인
# batch_size = cfg.train_dataloader.batch_size
# print(f"[DEBUG] train_dataloader.batch_size: {batch_size}")

# # drop_last, shuffle 확인
# drop_last = cfg.train_dataloader.get('drop_last', False)
# shuffle = cfg.train_dataloader.get('shuffle', True)
# print(f"[DEBUG] train_dataloader.drop_last: {drop_last}")
# print(f"[DEBUG] train_dataloader.shuffle: {shuffle}")

# # iteration 수 계산 (epoch 당)
# num_samples = len(train_ds)
# if drop_last:
#     num_iters = num_samples // batch_size
# else:
#     num_iters = (num_samples + batch_size - 1) // batch_size
# print(f"[DEBUG] Expected iterations per epoch: {num_iters}")

# # test dataset도 확인
test_ds_cfg = cfg.test_dataloader.dataset
test_ds = DATASETS.build(test_ds_cfg)
# print(f"[DEBUG] Test dataset length: {len(test_ds)}")
# print(f"[DEBUG] test_dataloader.batch_size: {cfg.test_dataloader.batch_size}")


# visualize
vis_backends = [
    dict(type='LocalVisBackend'),
        dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='cv_11_OD',     # 너가 보고 싶은 프로젝트
            entity='cv_11',      # 팀 이름
            name=wandb_name
        )
    )
]

cfg.visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

cfg.log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)


# 모델 학습
# runner = Runner.from_cfg(cfg)
# runner.train()
# for fold in range(5):
#     # train dataset ann_file만 바꾸면 됨
#     cfg.train_dataloader.dataset.ann_file = f"{root}/folds/train_fold{fold}.json"

#     # fold별 val_dataloader 새로 생성
#     cfg.val_dataloader = dict(
#         batch_size=1,
#         num_workers=2,
#         dataset=cfg.test_dataloader.dataset
#     )
#     cfg.val_dataloader.dataset.ann_file = f"{root}/folds/val_fold{fold}.json"

#     # fold별 val_evaluator
#     cfg.val_evaluator = dict(
#         type="CocoMetric",
#         ann_file=cfg.val_dataloader.dataset.ann_file,
#         metric=["bbox"],
#         classwise=True
#     )

#     # fold별 work_dir, wandb 이름
#     cfg.work_dir = f"{work_path}/work_dirs/detr_r50_trash_fold{fold}"
#     cfg.visualizer.vis_backends[1]['init_kwargs']['name'] = f"{wandb_name}_fold{fold}"

#     runner = Runner.from_cfg(cfg)
#     runner.train()

# train dataset ann_file만 바꾸면 됨
cfg.train_dataloader.dataset.ann_file = f"{root}/folds/train_fold0.json"

# fold별 val_dataloader 새로 생성
cfg.val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=cfg.test_dataloader.dataset
)
cfg.val_dataloader.dataset.ann_file = f"{root}/folds/val_fold0.json"

# fold별 val_evaluator
cfg.val_evaluator = dict(
    type="CocoMetric",
    ann_file=cfg.val_dataloader.dataset.ann_file,
    metric=["bbox"],
    classwise=True
)

# fold별 work_dir, wandb 이름
cfg.work_dir = f"{work_path}/work_dirs/detr_r50_trash_251211"
cfg.visualizer.vis_backends[1]['init_kwargs']['name'] = f"{wandb_name}_251211"

runner = Runner.from_cfg(cfg)
runner.train()
