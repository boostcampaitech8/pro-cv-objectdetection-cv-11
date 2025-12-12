import os
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

n_splits = 5

# Load json
train_ann = "./dataset/train.json"
save_dir = "./dataset/sgk-fold/"
os.makedirs(save_dir, exist_ok=True)

with open(train_ann) as f:
    data = json.load(f)

var = [(ann["image_id"], ann["category_id"]) for ann in data["annotations"]]  ## img_ids, labels
X = np.ones((len(data["annotations"]),1))
y = np.array([v[1] for v in var])  ## labels
groups = np.array([v[0] for v in var])  ## img_ids

cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=411)

fold = 0
for train_idx, val_idx in cv.split(X, y, groups):
    print("TRAIN:", groups[train_idx])
    print(" ", y[train_idx])
    print(" TEST:", groups[val_idx])
    print(" ", y[val_idx])

    print(f"### Fold {fold} ###")
    train_img_ids = set(groups[train_idx])
    val_img_ids = set(groups[val_idx])

    # Create new json dict
    train_json = {
        "images": [img for img in data["images"] if img["id"] in train_img_ids],
        "annotations": [ann for ann in data["annotations"] if ann["image_id"] in train_img_ids],
        "categories": data["categories"]
    }
    val_json = {
        "images": [img for img in data["images"] if img["id"] in val_img_ids],
        "annotations": [ann for ann in data["annotations"] if ann["image_id"] in val_img_ids],
        "categories": data["categories"]
    }

    # Save json
    train_path = os.path.join(save_dir, f"fold{fold}_train.json")
    val_path = os.path.join(save_dir, f"fold{fold}_val.json")

    with open(train_path, "w") as f:
        json.dump(train_json, f, indent=4)
    with open(val_path, "w") as f:
        json.dump(val_json, f, indent=4)
    
    fold += 1

    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")

print("Finished!")
