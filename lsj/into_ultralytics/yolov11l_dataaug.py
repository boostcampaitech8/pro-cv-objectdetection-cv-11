import wandb
wandb.login()

import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch, gc

BASE_DIR = Path("/data/ephemeral/home/pro-cv-objectdetection-cv-11-sj")
data_yaml_path = BASE_DIR / "dataset" / "yolo_dataset2" / "trash10.yaml"

WANDB_ENTITY = "cv_11"
WANDB_PROJECT = "cv_11_OD"

PROJECT_NAME = "trash10-yolo11l-dataaug"

def save_submission(model, stage, base_dir, img_test_dir):
    """
    model          : YOLO model after training
    stage          : 'baseline' / 'tune' / 'final' / 'dataaug' ...
    base_dir       : BASE_DIR
    img_test_dir   : path to test image directory
    """

    print(f"\n========== [{stage}] Running validation & submission save ==========\n")

    # -------------------------
    # 1) Validation metrics
    # -------------------------
    metrics = model.val()  # Ultralytics가 한번 더 val 돌려줌

    # 전체 mAP
    map50_95 = float(metrics.box.map)
    map50    = float(metrics.box.map50)
    map75    = float(metrics.box.map75)

    print(f"[{stage}] mAP50-95: {map50_95}")
    print(f"[{stage}] mAP50: {map50}")
    print(f"[{stage}] mAP75: {map75}")

    # Save metrics CSV locally
    metrics_path = Path(base_dir) / f"metrics_{stage}.csv"
    pd.DataFrame({
        "mAP50-95": [map50_95],
        "mAP50": [map50],
        "mAP75": [map75]
    }).to_csv(metrics_path, index=False)
    print(f"[{stage}] Metrics saved → {metrics_path}")

    # -------------------------
    # 1-1) 클래스별 mAP (mAP50-95) 계산 & wandb 로깅
    # -------------------------
    # metrics.box.maps: per-class mAP50-95 (list/array 길이 = num_classes)
    per_class_maps = metrics.box.maps  # e.g. array([0.1, 0.3, ...])

    # class 이름: model.names 혹은 yaml에서 쓰던 names
    try:
        class_names = model.names  # dict or list
        # dict일 경우 index 순 정렬
        if isinstance(class_names, dict):
            class_names = [class_names[i] for i in range(len(class_names))]
    except Exception:
        # 혹시 model.names를 못 읽으면, 대회 class 목록 직접 사용
        class_names = [
            "General trash",
            "Paper",
            "Paper pack",
            "Metal",
            "Glass",
            "Plastic",
            "Styrofoam",
            "Plastic bag",
            "Battery",
            "Clothing",
        ]

    # per-class 스칼라 로그 + 바차트용 테이블 준비
    import wandb
    per_class_log = {}
    table_data = []

    for i, m in enumerate(per_class_maps):
        # m이 numpy.float32 같은 경우도 있으니 float()로 변환
        m_val = float(m)
        cls_name = class_names[i] if i < len(class_names) else f"class_{i}"
        key = f"{stage}/mAP50-95/{cls_name}"
        per_class_log[key] = m_val
        table_data.append([cls_name, m_val])

    # wandb.Table + bar plot
    table = wandb.Table(data=table_data, columns=["class", "mAP50-95"])
    bar_plot = wandb.plot.bar(
        table,
        "class",
        "mAP50-95",
        title=f"{stage} per-class mAP50-95"
    )

    # -------------------------
    # 1-2) metrics.results_dict 에서 로스/지표들까지 전부 로깅
    # -------------------------
    extra_logs = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for k, v in results_dict.items():
            # v 가 tensor, numpy, float 등일 수 있으니 float 변환 시도
            try:
                extra_logs[f"{stage}/{k}"] = float(v)
            except Exception:
                # 스칼라가 아니면 무시
                pass

    # 🔥 wandb에 최종 validation 지표 + per-class + bar chart + results_dict 모두 로깅
    log_dict = {
        f"{stage}/mAP50-95": map50_95,
        f"{stage}/mAP50":    map50,
        f"{stage}/mAP75":    map75,
        f"{stage}/mAP50-95_per_class_bar": bar_plot,
    }
    log_dict.update(per_class_log)
    log_dict.update(extra_logs)

    wandb.log(log_dict)

    # -------------------------
    # 2) Prediction
    # -------------------------
    pred_results = model.predict(
        source=str(img_test_dir),
        conf=0.25,
        imgsz=1024,
        save=False,
        verbose=False,
    )

    rows = []
    for r in pred_results:
        filename = Path(r.path).name
        image_id = f"test/{filename}"

        preds = []
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            preds.append(f"{int(cls)} {float(conf):.8f} {x1:.5f} {y1:.5f} {x2:.5f} {y2:.5f}")

        rows.append({
            "PredictionString": " ".join(preds),
            "image_id": image_id
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("image_id").reset_index(drop=True)

    # -------------------------
    # 3) Sort PredictionString by label
    # -------------------------
    def sort_pred(s):
        if not isinstance(s, str) or not s.strip():
            return ""
        p = s.split()
        boxes = [p[i:i+6] for i in range(0, len(p), 6)]
        boxes = sorted(boxes, key=lambda b: int(b[0]))
        return " ".join(" ".join(b) for b in boxes)

    df["PredictionString"] = df["PredictionString"].apply(sort_pred)

    # -------------------------
    # 4) Save submission
    # -------------------------
    save_path = Path(base_dir) / f"submission_yolov11l_{stage}.csv"
    df.to_csv(save_path, index=False)

    print(f"[{stage}] Submission saved → {save_path}")

    # ✅ wandb artifact로 submission 파일 업로드 (선택 사항)
    artifact = wandb.Artifact(
        name=f"submission_yolov11l_{stage}",
        type="submission"
    )
    artifact.add_file(str(save_path))
    wandb.log_artifact(artifact)
    print(f"[{stage}] Submission artifact logged to wandb")


# ====================== 1) BASELINE TRAIN ======================
wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name="yolo11l-dataaug"
)

model = YOLO("yolo11l.pt")

results_baseline = model.train(
    data=str(data_yaml_path),
    epochs=50,
    imgsz=1024,
    batch=16,
    project=PROJECT_NAME,
    name="yolo11l-dataaug",
)

# === baseline submission 저장 ===
save_submission(
    model=model,
    stage="dataaug",
    base_dir=BASE_DIR,
    img_test_dir=BASE_DIR / "dataset" / "yolo_dataset2" / "images" / "test"
)


wandb.finish()
print("Baseline complete\n")


del model
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()