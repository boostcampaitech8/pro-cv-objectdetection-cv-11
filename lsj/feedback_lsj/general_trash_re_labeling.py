import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 경로 설정
DATASET_DIR = Path("/data/ephemeral/home/pro-cv-objectdetection-cv-11-sj/dataset")          # dataset 폴더
ANNOT_PATH = DATASET_DIR / "train.json"
IMG_ROOT   = DATASET_DIR               # file_name 안에 이미 'train/0000.jpg' 형식이 들어있다
OUT_JSON   = DATASET_DIR / "train_cleaned_general_trash.json"

# 1) COCO json 로드
with open(ANNOT_PATH, "r") as f:
    coco = json.load(f)

images      = {img["id"]: img for img in coco["images"]}
annotations = {ann["id"]: ann for ann in coco["annotations"]}

# 2) "General trash" category_id 찾기
gt_category_id = None
for cat in coco["categories"]:
    if cat["name"] == "General trash":
        gt_category_id = cat["id"]
        break

if gt_category_id is None:
    raise ValueError("'General trash' 카테고리를 찾을 수 없습니다. categories를 확인하세요.")

print("General trash category_id:", gt_category_id)

# 3) image_id별 General trash annotation 모으기
anns_by_image = defaultdict(list)
for ann in coco["annotations"]:
    if ann["category_id"] == gt_category_id:
        anns_by_image[ann["image_id"]].append(ann)

# 4) General trash가 있는 image만 filename 순서대로 정렬
image_ids_with_gt = sorted(
    anns_by_image.keys(),
    key=lambda img_id: images[img_id]["file_name"]
)

print("General trash가 포함된 이미지 수:", len(image_ids_with_gt))


def save_cleaned_json(out_path=OUT_JSON, deleted_ids=None, edited_anns=None):
    """
    deleted_ids: 삭제된 annotation id들의 set
    edited_anns: 수정된 annotation dict들 {ann_id: ann_dict}
    """
    deleted_ids = deleted_ids or set()
    edited_anns = edited_anns or {}

    new_annotations = []
    for ann_id, ann in annotations.items():
        if ann_id in deleted_ids:
            continue
        if ann_id in edited_anns:
            new_annotations.append(edited_anns[ann_id])
        else:
            new_annotations.append(ann)

    cleaned_coco = {
        "images": list(images.values()),
        "categories": coco["categories"],
        "annotations": new_annotations,
    }

    with open(out_path, "w") as f:
        json.dump(cleaned_coco, f)
    print(f"[저장] {out_path} 에 현재까지의 수정 내용을 저장했습니다. (총 ann: {len(new_annotations)})")

def review_general_trash(start_index=0, auto_save_every=10):
    """
    start_index: image_ids_with_gt 중 몇 번째 이미지부터 시작할지 (처음은 0)
    auto_save_every: N장마다 자동으로 json 저장
    """
    deleted_ids = set()        # 삭제된 ann.id
    edited_anns = {}           # 수정된 ann: {ann_id: ann_dict}

    total = len(image_ids_with_gt)

    for idx_in_list in range(start_index, total):
        img_id   = image_ids_with_gt[idx_in_list]
        img_info = images[img_id]
        file_name = img_info["file_name"]          # 예: train/0000.jpg
        img_path  = IMG_ROOT / file_name

        # 이미지 로드
        image = np.array(Image.open(img_path).convert("RGB"))

        # 현재 이미지의 General trash ann들 (삭제되지 않은 것만)
        gt_anns_all = anns_by_image[img_id]
        gt_anns = [ann for ann in gt_anns_all if ann["id"] not in deleted_ids]

        if len(gt_anns) == 0:
            # 모두 삭제해버린 경우: 그냥 다음으로
            print(f"[{idx_in_list+1}/{total}] {file_name} (모든 General trash bbox가 삭제되었습니다. 건너뜀)")
            continue

        # ====== 여기부터: 좌(원본), 우( bbox 포함 ) 2개 subplot ======
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        # 왼쪽: 원본 이미지
        axes[0].imshow(image)
        axes[0].set_title(f"[{idx_in_list+1}/{total}] {file_name} - Original")
        axes[0].axis("off")

        # 오른쪽: bbox 포함 이미지
        axes[1].imshow(image)
        for i, ann in enumerate(gt_anns):
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            axes[1].add_patch(rect)
            # 번호 + ann_id 같이 출력
            axes[1].text(
                x, y - 5,
                f"{i} (id={ann['id']})",
                fontsize=8,
                color="yellow",
                bbox=dict(facecolor="black", alpha=0.5, pad=1)
            )
        axes[1].set_title("General trash with BBoxes")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        # ====== 여기까지 그림 부분만 이전 버전과 변경됨 ======

        # 사용자 입력
        print("명령을 입력하세요:")
        print("  Enter : 변경 없이 다음 이미지")
        print("  d 0,2 : 인덱스 0과 2 bbox 삭제")
        print("  e 1   : 인덱스 1 bbox 좌표 수정")
        print("  q     : 종료 후 저장")
        cmd = input(">> ").strip()

        if cmd == "":
            # 아무 변화 없이 다음
            pass

        elif cmd.startswith("d"):
            # 삭제
            # 예) "d 0,2" 또는 "d 1"
            try:
                parts = cmd[1:].strip()
                if parts:
                    idx_strs = parts.split(",")
                    deleted_list = []
                    for s in idx_strs:
                        s = s.strip()
                        if s == "":
                            continue
                        i = int(s)
                        ann = gt_anns[i]
                        deleted_ids.add(ann["id"])
                        deleted_list.append(ann["id"])
                    print("삭제된 ann_ids:", deleted_list)
                else:
                    print("삭제할 인덱스를 입력하지 않았습니다. (예: d 0,2)")
            except Exception as e:
                print("삭제 명령 파싱 중 에러:", e)

        elif cmd.startswith("e"):
            # 수정 (bbox 좌표 직접 입력)
            # 예) "e 1" -> 인덱스 1 수정
            try:
                parts = cmd[1:].strip()
                if not parts:
                    print("수정할 인덱스를 입력하세요. (예: e 1)")
                else:
                    i = int(parts)
                    ann = gt_anns[i].copy()
                    print(f"현재 bbox (x, y, w, h): {ann['bbox']}")
                    new_str = input("새 bbox 입력 (x y w h, 공백 구분, Enter로 취소): ").strip()
                    if new_str:
                        xs = new_str.split()
                        if len(xs) != 4:
                            print("4개의 숫자를 입력해야 합니다.")
                        else:
                            new_bbox = [float(v) for v in xs]
                            ann["bbox"] = new_bbox
                            edited_anns[ann["id"]] = ann
                            print(f"bbox 업데이트 완료: {new_bbox}")
                    else:
                        print("수정 취소")
            except Exception as e:
                print("수정 명령 처리 중 에러:", e)

        elif cmd == "q":
            # 종료 + 저장
            save_cleaned_json(OUT_JSON, deleted_ids, edited_anns)
            print("리뷰를 중단합니다.")
            return

        else:
            print("알 수 없는 명령입니다. 그대로 다음 이미지로 넘어갑니다.")

        # 자동 저장
        if (idx_in_list + 1) % auto_save_every == 0:
            save_cleaned_json(OUT_JSON, deleted_ids, edited_anns)

    # 모든 이미지 처리 완료 후 최종 저장
    save_cleaned_json(OUT_JSON, deleted_ids, edited_anns)
    print("모든 General trash 이미지 검토 완료!")

review_general_trash(start_index=0, auto_save_every=10)
