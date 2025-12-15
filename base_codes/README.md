# Baseline Object Detection Models

이 프로젝트는 Faster R-CNN 기반 객체 탐지 모델의 훈련 및 추론을 위한 baseline 코드입니다.

##  프로젝트 구조

```
baseline/
├── mmdetection/          # MMDetection 3.x 기반 구현
│   ├── faster_rcnn_train.ipynb
│   ├── faster_rcnn_inference.ipynb
│   └── CHANGELOG.md
├── detectron2/           # Detectron2 기반 구현
│   ├── faster_rcnn_train.ipynb
│   ├── faster_rcnn_inference.ipynb
│   └── CHANGELOG.md
├── faster_rcnn/          # PyTorch Torchvision 기반 구현
└── requirements.txt
```

##  주요 변경사항

### MMDetection: 버전 업그레이드 (2.x → 3.x)

MMDetection은 **MMDetection 2.x에서 3.x로 대규모 버전 업그레이드**를 진행했습니다:

- **API 완전 변경**: `mmcv.Config` → `mmengine.config.Config`
- **훈련 방식 혁신**: `train_detector()` → `Runner.from_cfg(cfg).train()`
- **추론 API 재설계**: `single_gpu_test` → `DetInferencer`
- **설정 구조 개선**: `cfg.data.train` → `cfg.train_dataloader.dataset`
- **새로운 기능**: 데이터셋 분석, 배치 추론, 안전한 텐서 처리

자세한 변경사항은 [mmdetection/CHANGELOG.md](mmdetection/CHANGELOG.md)를 참조하세요.

### Detectron2: 코드 리팩토링

Detectron2는 **동일한 버전 내에서 코드 품질 개선**을 위한 리팩토링을 진행했습니다:

- **Import 최적화**: 불필요한 import 제거 및 구조화
- **변수명 개선**: `ROOT`, `TRAIN_NAME` 등 명확한 네이밍
- **코드 구조화**: 설정 방식 체계화 및 가독성 향상
- **오타 수정**: `NUM_WOREKRS` → `NUM_WORKERS`
- **효율성 개선**: 더 효율적인 데이터 처리 방식

자세한 변경사항은 [detectron2/CHANGELOG.md](detectron2/CHANGELOG.md)를 참조하세요.

##  환경 요구사항

### 이전 버전 (old_baseline)
- **PyTorch**: 기본 버전
- **CUDA**: 기본 버전
- **MMDetection**: 2.x
- **MMCV**: 1.7.0

### 현재 버전 (baseline)
- **PyTorch**: 2.1.0
- **CUDA**: 11.8
- **MMDetection**: 3.x
- **MMEngine**: 최신 버전
- **Detectron2**: 최신 버전

### 주요 의존성 패키지

```txt
# 핵심 패키지
torch==2.1.0
torchvision
mmdet==3.x
mmengine
detectron2

# 데이터 처리
opencv-python==4.7.0.72
pycocotools==2.0.7
pandas
numpy==1.26.0

# 시각화 및 유틸리티
tqdm==4.65.0
jupyter==1.0.0
tensorboard
seaborn==0.12.2

# 데이터 증강
albumentations==0.4.6
imgaug==0.4.0
```

##  사용 방법

### 1. 환경 설정
```bash
pip install .
```

### 2. MMDetection 사용
```bash
cd mmdetection
jupyter notebook faster_rcnn_train.ipynb
jupyter notebook faster_rcnn_inference.ipynb
```

### 3. Detectron2 사용
```bash
cd detectron2
jupyter notebook faster_rcnn_train.ipynb
jupyter notebook faster_rcnn_inference.ipynb
```

## 콘텐츠 라이선스
<font color='red'><b>**WARNING**</b></font> : **본 교육 콘텐츠의 지식재산권은 재단법인 네이버커넥트에 귀속됩니다. 본 콘텐츠를 어떠한 경로로든 외부로 유출 및 수정하는 행위를 엄격히 금합니다.** 다만, 비영리적 교육 및 연구활동에 한정되어 사용할 수 있으나 재단의 허락을 받아야 합니다. 이를 위반하는 경우, 관련 법률에 따라 책임을 질 수 있습니다.
