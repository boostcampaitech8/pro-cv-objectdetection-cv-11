#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from effdet import create_model, create_evaluator
import timm.utils as utils

from pycocotools.coco import COCO

# ===========================
# 사용자 환경 설정
# ===========================
home_path = "/data/ephemeral/home"
work_path = "/data/ephemeral/home/jsw_pro-cv-objectdetection-cv-11/jsw"
dataset_root = f"{home_path}/dataset"
train_ann = "train.json"
test_ann = "test.json"
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
num_classes = len(classes)
checkpoint_dir = osp.join(work_path, "work_dirs/efficientdet_d4_trash")
os.makedirs(checkpoint_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===========================
# 파서 설정
# ===========================
parser = argparse.ArgumentParser(description="EfficientDet D4 Custom Training")
parser.add_argument('--root', default=dataset_root, type=str, help='Dataset root dir')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--img-size', default=1024, type=int)
parser.add_argument('--lr', default=0.002, type=float)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--pretrained', action='store_true', default=True)
parser.add_argument('--checkpoint-dir', default=checkpoint_dir, type=str)
args = parser.parse_args()

# ===========================
# 랜덤 시드, device
# ===========================
utils.random_seed(2025, 0)
device = torch.device(args.device)

# ===========================
# COCO Dataset 정의
# ===========================
class CocoDataset(Dataset):
    def __init__(self, root, ann_file, img_size=1024):
        self.coco = COCO(osp.join(root, ann_file))
        self.ids = self.coco.getImgIds()
        self.root = root
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.ids[idx])[0]
        path = osp.join(self.root, img_info['file_name'])
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for ann in anns:
            boxes.append(ann['bbox'])  # [x, y, w, h]
            labels.append(ann['category_id'])
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
        }
        return img, target

# ===========================
# DataLoader 생성
# ===========================
train_dataset = CocoDataset(dataset_root, train_ann, img_size=args.img_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

val_dataset = CocoDataset(dataset_root, test_ann, img_size=args.img_size)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# Evaluator 생성
evaluator = create_evaluator('coco', val_dataset, pred_yxyx=False)

# ===========================
# 모델 생성
# ===========================
model = create_model(
    'tf_efficientdet_d4',
    bench_task='train',
    num_classes=num_classes,
    pretrained_backbone=args.pretrained,  # backbone만 pretrained
)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# ===========================
# 학습 루프
# ===========================
for epoch in range(args.epochs):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = model(inputs, targets)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")

    # ===========================
    # 체크포인트 저장
    # ===========================
    ckpt_path = osp.join(args.checkpoint_dir, f"epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), ckpt_path)
    logging.info(f"Saved checkpoint: {ckpt_path}")

# ===========================
# 최종 평가
# ===========================
model.eval()
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(inputs, targets)
        evaluator.add_predictions(outputs['detections'], targets)

map_score = evaluator.evaluate()
logging.info(f"Validation mAP: {map_score:.6f}")
