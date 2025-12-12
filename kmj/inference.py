import os
import numpy as np
import pandas as pd
import os.path as osp

from pycocotools.coco import COCO
from mmengine.config import Config
from mmdet.apis import DetInferencer

NUM_FOLDS = 5
DATASET_ROOT = "../../dataset/"
TEST_JSON = "test.json"

saved_epoch = 10
batch_size = 4
score_thr = 0.05
device = "cuda:0"

# custom setting
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

config_path = "configs/my_model/faster-rcnn_r101_fpn_1x_recycle.py"
save_model_name = "faster-rcnn_r101_fpn_1x_recycle"

coco = COCO(osp.join(DATASET_ROOT, TEST_JSON))
img_ids = coco.getImgIds()
imgs = coco.loadImgs(img_ids)

def _img_path(im):
    return osp.join(DATASET_ROOT, im["file_name"])

img_paths = [_img_path(im) for im in imgs]
file_names = [im["file_name"] for im in imgs]

# kfold inference
for FOLD in range(NUM_FOLDS):
    print(f"##### Inference: FOLD {FOLD} #####")

    work_dir = f"./work_dirs/{save_model_name}/fold_{FOLD}/"
    checkpoint_path = osp.join(work_dir, f"epoch_{saved_epoch}.pth")

    if not osp.exists(checkpoint_path):
        continue

    cfg = Config.fromfile(config_path)
    cfg.default_scope = "mmdet"

    ds = cfg.test_dataloader["dataset"]
    ds.metainfo = dict(classes=classes)
    ds.data_root = DATASET_ROOT
    ds.ann_file = TEST_JSON
    ds.data_prefix = dict(img="")
    cfg.test_dataloader.dataset.pipeline[1]["scale"] = (512, 512)

    cfg.randomness = dict(seed=2025, deterministic=False)
    cfg.model.roi_head.bbox_head.num_classes = len(classes)

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
    save_path = osp.join(work_dir, f"test_output_fold{FOLD}.csv")
    submission.to_csv(save_path, index=False)
    
    print(f"Saved FOLD {FOLD}: {save_path}")

print("Finished!")
