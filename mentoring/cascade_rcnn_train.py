import wandb
# 모듈 import
import pandas as pd
from collections import Counter
from IPython.display import display

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules

wandb.login()

# custom 설정
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

root = "../../dataset/"
train_ann = "folds/train_fold0.json"
val_ann   = "folds/val_fold0.json"
test_ann  = "test.json"

# config file 들고오기
cfg = Config.fromfile("configs/my_model/cascade_rcnn_swin_tiny.py")
# cfg = Config.fromfile("configs/my_model/cascade_rcnn_swin_large.py")

register_all_modules(init_default_scope=True)
cfg.default_scope = "mmdet"


########################################################
# 1) Train / Val / Test dataset 구성
########################################################
for ds_key, ann_path in [
    ("train_dataloader", train_ann),
    ("val_dataloader",   val_ann),
    ("test_dataloader",  test_ann),
]:
    if ds_key not in cfg:
        continue

    ds = cfg[ds_key]["dataset"] if "dataset" in cfg[ds_key] else cfg[ds_key]
    ds.metainfo = dict(classes=classes)
    ds.data_root = root
    ds.ann_file = ann_path
    ds.data_prefix = dict(img="")


# dataloader batch 설정
cfg.train_dataloader.batch_size = 1
cfg.train_dataloader.num_workers = 4
cfg.val_dataloader.batch_size = 2
cfg.val_dataloader.num_workers = 2
cfg.test_dataloader.batch_size = 1
cfg.test_dataloader.num_workers = 2


########################################################
# 2) Augmentation pipeline
########################################################
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=[0.5, 0.15], direction=['horizontal', 'vertical']),
    dict(type='MinIoURandomCrop', min_ious=[0.4, 0.5, 0.6, 0.7], min_crop_size=0.3),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(type='RandomBrightnessContrast', p=1.0),
                    dict(type='HueSaturationValue', p=1.0),
                    dict(type='CLAHE', p=1.0),
                    dict(type='RGBShift', p=1.0),
                ],
                p=0.5
            ),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', p=1.0),
                    dict(type='GaussianBlur', p=1.0),
                    dict(type='Blur', p=1.0),
                ],
                p=0.2
            ),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='ShiftScaleRotate', rotate_limit=10, p=1.0),
                    dict(type='RandomRotate90', p=1.0),
                ],
                p=0.2
            ),
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True
        ),
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
        skip_img_without_anno=True
    ),
    dict(
    type='RandomChoiceResize',
    scales=[(800, 800), (1024, 1024), (1200, 1200)],
    keep_ratio=True
    ),
    dict(type='PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # --- Multi-Scale TTA ---
        scales=[
            (1024, 1024),  # baseline
            (800, 800),   # COCO 공식 cascade scale
            (1200, 1200),   # 더 큰 객체 대응
        ],
        flip=True,  # 좌우 flip TTA
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='PackDetInputs')
        ]
    ),
]

cfg.test_dataloader.dataset.pipeline = tta_pipeline
cfg.train_dataloader.dataset.pipeline = train_pipeline
cfg.val_dataloader.dataset.pipeline = val_pipeline


########################################################
# 3) Validation 활성화
########################################################
cfg.val_evaluator = dict(
    type="CocoMetric",
    ann_file=root + val_ann,
    metric=["bbox"],
    classwise=True     # 클래스별 mAP도 출력 가능
)

cfg.val_cfg = dict(type="ValLoop")
cfg.test_cfg = dict(type="TestLoop")

# 매 epoch마다 validation
cfg.train_cfg.max_epochs = 18
cfg.train_cfg.val_interval = 1



########################################################
# 4) Backbone 클래스 수 수정
########################################################
for i in range(3):
    cfg.model.roi_head.bbox_head[i].num_classes = len(classes)


########################################################
# 5) Optimizer (AdamW)
########################################################
cfg.optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=35, norm_type=2),
)


########################################################
# 6) LR Scheduler
########################################################
cfg.param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=cfg.train_cfg["max_epochs"],
        eta_min=1e-6,
    )
]


########################################################
# 7) Checkpoint 저장 방식 → mAP 기반 Top3 자동 저장
########################################################
cfg.default_hooks["checkpoint"] = dict(
    type="CheckpointHook",
    interval=1,
    max_keep_ckpts=3,
    save_best="coco/bbox_mAP",  # mAP 기준 best 저장
    rule="greater",             # 값이 클수록 좋음
)

# 저장 파일명 포맷 (epoch 포함)
cfg.work_dir = "./work_dirs/cascade_rcnn_swin_tiny_last"


########################################################
# 8) W&B 설정
########################################################
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='cv_11_OD',
            entity='cv_11',
            name='cascade_swin_tiny_last'
        )
    )
]

cfg.visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

cfg.log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
cfg.device = "cuda"

# 모델 학습
runner = Runner.from_cfg(cfg)
runner.train()