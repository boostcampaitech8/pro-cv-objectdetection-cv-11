import wandb
wandb.login()

import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch, gc

BASE_DIR = Path("/data/ephemeral/home/pro-cv-objectdetection-cv-11-sj")
data_yaml_path = BASE_DIR / "dataset" / "yolo_dataset" / "trash10.yaml"

WANDB_ENTITY = "cv_11"
WANDB_PROJECT = "cv_11_OD"

PROJECT_NAME = "trash10-yolo11l"

def save_submission(model, stage, base_dir, img_test_dir):
    """
    model          : YOLO model after training
    stage          : 'baseline' / 'tune' / 'final' ...
    base_dir       : BASE_DIR
    img_test_dir   : path to test image directory
    """

    print(f"\n========== [{stage}] Running validation & submission save ==========\n")

    # -------------------------
    # 1) Validation metrics
    # -------------------------
    metrics = model.val()

    print(f"[{stage}] mAP50-95: {metrics.box.map}")
    print(f"[{stage}] mAP50: {metrics.box.map50}")
    print(f"[{stage}] mAP75: {metrics.box.map75}")

    # Save metrics CSV
    metrics_path = Path(base_dir) / f"metrics_{stage}.csv"
    pd.DataFrame({
        "mAP50-95": [metrics.box.map],
        "mAP50": [metrics.box.map50],
        "mAP75": [metrics.box.map75]
    }).to_csv(metrics_path, index=False)
    print(f"[{stage}] Metrics saved → {metrics_path}")

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


# ====================== 1) BASELINE TRAIN ======================
wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name="yolo11l-baseline"
)

model = YOLO("yolo11l.pt")

results_baseline = model.train(
    data=str(data_yaml_path),
    epochs=50,
    imgsz=1024,
    batch=16,
    project=PROJECT_NAME,
    name="yolo11l-baseline",
)

# === baseline submission 저장 ===
save_submission(
    model=model,
    stage="baseline",
    base_dir=BASE_DIR,
    img_test_dir=BASE_DIR / "dataset" / "yolo_dataset" / "images" / "test"
)


wandb.finish()
print("Baseline complete\n")


del model
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# ====================== 2) TUNING ======================
wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name="yolo11l-tune"
)

model = YOLO("yolo11l.pt")    # ← 다시 모델 선언 필요

search_space = {
    "lr0": (1e-5, 1e-2),
    "lrf": (0.01, 0.5),
    "momentum": (0.7, 0.98),
    "weight_decay": (0.0, 1e-3),
    "warmup_epochs": (0.0, 3.0),

    "box": (0.5, 15.0),
    "cls": (0.2, 4.0),

    "hsv_s": (0.0, 0.8),
    "hsv_v": (0.0, 0.8),
    "scale": (0.0, 0.9),
    "translate": (0.0, 0.3),
    "fliplr": (0.0, 1.0),
    "flipud": (0.0, 0.5),
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 0.5),
}

tune_results = model.tune(
    data=str(data_yaml_path),
    epochs=20,
    iterations=30,
    imgsz=1024,
    batch=16,
    space=search_space,
    project=PROJECT_NAME,
    name="yolo11l-tune",
    optimizer="AdamW",
    plots=False,
    save=False,
    val=False,
)

# === tuning submission 저장 ===
save_submission(
    model=model,   # tune도 동일 model 객체 업데이트됨
    stage="tune",
    base_dir=BASE_DIR,
    img_test_dir=BASE_DIR / "dataset" / "yolo_dataset" / "images" / "test"
)

wandb.finish()
print("Tuning complete\n")



del model
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# ====================== 3) FINAL TRAIN WITH BEST HYP ======================
import yaml

wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name="yolo11l-final-best-hyp"
)

tune_dir = Path("runs") / "detect" / "tune"
best_hyp_path = tune_dir / "best_hyperparameters.yaml"

assert best_hyp_path.exists(), f"{best_hyp_path} not found. Run tuning first."

with open(best_hyp_path, "r") as f:
    best_hyp = yaml.safe_load(f)

final_model = YOLO("yolo11l.pt")

final_results = final_model.train(
    data=str(data_yaml_path),
    epochs=100,
    imgsz=1024,
    batch=16,
    project=PROJECT_NAME,
    name="yolo11l-final-best-hyp",
    **best_hyp,
)

# === final submission 저장 ===
save_submission(
    model=final_model,
    stage="final",
    base_dir=BASE_DIR,
    img_test_dir=BASE_DIR / "dataset" / "yolo_dataset" / "images" / "test"
)

wandb.finish()
print("Final training complete\n")



del final_model
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()