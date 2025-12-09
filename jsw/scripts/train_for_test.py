from mmengine.config import Config
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules

register_all_modules(init_default_scope=True)

cfg = Config.fromfile("configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py")

train_ds_cfg = cfg.train_dataloader.dataset
train_ds = DATASETS.build(train_ds_cfg)

print("Dataset length:", len(train_ds))
print("Batch size:", cfg.train_dataloader.batch_size)