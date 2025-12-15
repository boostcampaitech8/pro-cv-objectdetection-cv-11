# base from ai stages, edited by 형석님
import json
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# load json annotation = {dataset file 경로}
data_path = "/data/ephemeral/home/dataset"
save_dir = f'{data_path}/folds'

annotation = f"{data_path}/train.json"
with open(annotation) as f: 
    data = json.load(f)

var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones((len(data['annotations']),1)) # dummy
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

for train_idx, val_idx in cv.split(X, y, groups):
    print("TRAIN:", groups[train_idx])
    print(" ", y[train_idx])
    print(" TEST:", groups[val_idx])
    print(" ", y[val_idx])


# check distribution

def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())
    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) + 1)]

distrs = [get_distribution(y)]
index = ['training set']

for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
    train_y, val_y = y[train_idx], y[val_idx]
    train_gr, val_gr = groups[train_idx], groups[val_idx]

    # 그룹 겹침 확인
    assert len(set(train_gr) & set(val_gr)) == 0

    distrs.append(get_distribution(train_y))
    distrs.append(get_distribution(val_y))
    index.append(f'train - fold{fold_ind}')
    index.append(f'val - fold{fold_ind}')

categories = [d['name'] for d in data['categories']]
df = pd.DataFrame(distrs, index=index, columns=[categories[i] for i in range(np.max(y) + 1)])

print(df)


import os

os.makedirs(save_dir, exist_ok=True)

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
    print(f"\n===== Fold {fold_idx} =====")

    # 이번 fold에서 사용하는 image_id 목록 추출
    train_img_ids = set(groups[train_idx])
    val_img_ids = set(groups[val_idx])

    # 이미지 중복 없음 확인
    assert len(train_img_ids & val_img_ids) == 0

    # 이미지 정보 가져오기
    images = data['images']
    annotations = data['annotations']

    # fold별 이미지/annotation 분할
    train_images = [img for img in images if img['id'] in train_img_ids]
    val_images   = [img for img in images if img['id'] in val_img_ids]

    train_annotations = [ann for ann in annotations if ann['image_id'] in train_img_ids]
    val_annotations   = [ann for ann in annotations if ann['image_id'] in val_img_ids]

    # json 형태로 저장
    train_json = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": data['categories']
    }

    val_json = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": data['categories']
    }

    # 파일 저장
    with open(f"{save_dir}/train_fold{fold_idx}.json", "w") as f:
        json.dump(train_json, f, indent=2)

    with open(f"{save_dir}/val_fold{fold_idx}.json", "w") as f:
        json.dump(val_json, f, indent=2)

    print(f"Saved train_fold{fold_idx}.json  ({len(train_images)} images)")
    print(f"Saved val_fold{fold_idx}.json    ({len(val_images)} images)")
