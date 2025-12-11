#!/usr/bin/env python
import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse

from pycocotools.coco import COCO
from validate import validate  # rwightman EfficientDet validate.py

# ===========================
# 환경 설정
# ===========================
base_path = "/data/ephemeral/home/jsw_pro-cv-objectdetection-cv-11/jsw"
root = "/data/ephemeral/home/dataset"
test_ann = "test.json"
work_dir = f"{base_path}/work_dirs/efficientdet_d4_trash"
submission_name = "efficientdet_d4_submission.csv"
checkpoint_path = osp.join(work_dir, "epoch_1.pth")
device = "cuda:0"

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
num_classes = len(classes)
img_size = 1024
batch_size = 4
score_thr = 0.05
num_gpu = 1

# ===========================
# COCO test 이미지 로드
# ===========================
coco = COCO(osp.join(root, test_ann))
img_ids = coco.getImgIds()
imgs = coco.loadImgs(img_ids)

def _img_path(im):
    return osp.join(root, im["file_name"])

img_paths = [_img_path(im) for im in imgs]
file_names = [im["file_name"] for im in imgs]

# ===========================
# validate.py Args 생성
# ===========================
args = argparse.Namespace(
    root=root,
    model='tf_efficientdet_d4',
    num_classes=num_classes,
    pretrained=True,
    img_size=img_size,
    num_gpu=num_gpu,
    results=osp.join(work_dir, "results.json"),
    checkpoint=checkpoint_path,
    amp=False,
    apex_amp=False,
    native_amp=False,
    redundant_bias=None,
    soft_nms=None,
    no_prefetcher=False,
    use_ema=False,
    torchscript=False,
    torchcompile=None
)

# ===========================
# 추론 실행
# ===========================
mean_ap = validate(args)
print(f"Mean AP (dummy for submission): {mean_ap}")

# ===========================
# 결과 JSON 로드 후 submission CSV 생성
# ===========================
import json
with open(args.results, "r") as f:
    results_json = json.load(f)

prediction_strings = []

for res in results_json:
    # res 예: {'image_id': xxx, 'bbox': [[x1,y1,w,h],...], 'score': [...], 'category_id': [...]}
    bboxes = np.array(res['bbox'])
    scores = np.array(res['score'])
    labels = np.array(res['category_id'])
    
    keep = scores >= score_thr
    bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]
    
    ps = []
    for (x, y, w, h), sc, lb in zip(bboxes, scores, labels):
        ps.extend([str(lb), f"{sc:.6f}", f"{x:.1f}", f"{y:.1f}", f"{x+w:.1f}", f"{y+h:.1f}"])
    prediction_strings.append(" ".join(ps))

submission = pd.DataFrame({
    "PredictionString": prediction_strings,
    "image_id": file_names
})

save_path = osp.join(work_dir, submission_name)
submission.to_csv(save_path, index=False)
print(f"Saved submission CSV: {save_path}")
print(submission.head())
