import os
import os.path as osp

import numpy as np
import pandas as pd

from mmengine.config import Config
from mmdet.apis import DetInferencer
from pycocotools.coco import COCO

base_path = "/data/ephemeral/home/jsw_pro-cv-objectdetection-cv-11/base_codes/mmdetection"
home_path = "/data/ephemeral/home"
work_path = "/data/ephemeral/home/jsw_pro-cv-objectdetection-cv-11/jsw"
submission_name = "251209_detr_1_epoch.csv"

# custom 설정
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

root = f"{home_path}/dataset"
test_ann = "test.json"

# saved_epoch = 1
saved_epoch = 1
batch_size = 4
score_thr = 0.05
device = "cuda:0"

work_dir = f"{work_path}/work_dirs/detr_r50_trash"
checkpoint_path = osp.join(work_dir, f"epoch_{saved_epoch}.pth")

# config file 들고오기
cfg = Config.fromfile(f"{base_path}/configs/detr/detr_r50_8xb2-150e_coco.py")
cfg.default_scope = "mmdet"

# dataset config 수정
ds = cfg.test_dataloader["dataset"] if "dataset" in cfg.test_dataloader else cfg.test_dataloader
ds.metainfo = dict(classes=classes)
ds.data_root = root
ds.ann_file = test_ann
ds.data_prefix = dict(img="")
# cfg.test_dataloader.dataset.pipeline[1]["scale"] = (512, 512)

cfg.randomness = dict(seed=2025, deterministic=False)
cfg.model.bbox_head.num_classes = len(classes)

# test image info
coco = COCO(osp.join(root, test_ann))
img_ids = coco.getImgIds()
imgs = coco.loadImgs(img_ids)

def _img_path(im):
    return osp.join(root, "", im["file_name"]) if "" \
        else osp.join(root, im["file_name"])

img_paths = [_img_path(im) for im in imgs]
file_names = [im["file_name"] for im in imgs]


# DetInferencer를 이용한 batch inference
inferencer = DetInferencer(
    model=cfg,
    weights=checkpoint_path,
    device=device
)

results = inferencer(
    img_paths,
    batch_size=batch_size,
    return_datasamples=True,
    no_save_vis=True,
)


# submission 양식에 맞게 output 후처리
prediction_strings = []

for data_sample in results["predictions"]:
    inst = data_sample.pred_instances
    bboxes = inst.bboxes.cpu().numpy() if hasattr(inst.bboxes, "cpu") else np.asarray(inst.bboxes)
    scores = inst.scores.cpu().numpy() if hasattr(inst.scores, "cpu") else np.asarray(inst.scores)
    labels = inst.labels.cpu().numpy() if hasattr(inst.labels, "cpu") else np.asarray(inst.labels)

    keep = scores >= score_thr
    bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]

    ps = []
    for (x1, y1, x2, y2), sc, lb in zip(bboxes, scores, labels):
        ps.extend([str(int(lb)), f"{float(sc):.6f}",
                   f"{float(x1):.1f}", f"{float(y1):.1f}",
                   f"{float(x2):.1f}", f"{float(y2):.1f}"])
    prediction_strings.append(" ".join(ps))

submission = pd.DataFrame({
    "PredictionString": prediction_strings,
    "image_id": file_names,
})
save_path = osp.join(work_dir, submission_name)
submission.to_csv(save_path, index=False)
print(f"Saved: {save_path}")
print(submission.head())