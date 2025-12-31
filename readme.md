<div align='center'> 
  <h2>♻️ 재활용 품목 분류를 위한 Object Detection</h2>
  <img width="605" height="428" alt="Image" src="https://github.com/user-attachments/assets/5aa18669-83b9-495e-b5fd-c193890b8784" />
</div>



## Project Overview
대량 생산과 소비가 일상화된 현대 사회에서는 쓰레기 배출량 증가로 인해 매립지 부족과 환경 오염 등 다양한 사회 문제가 발생하고 있다.

분리수거는 이러한 환경 부담을 줄일 수 있는 중요한 방법으로, 정확한 분리배출 여부에 따라 재활용 가능 자원의 가치가 크게 달라진다.

본 프로젝트에서는 **이미지 기반 Object Detection 모델**을 활용하여 사진 속 재활용 쓰레기를 자동으로 탐지하고 분류하는 시스템을 구축하는 것을 목표로 한다.

- **Competition Period** : 2025.12.03 ~ 2025.12.11
- **Input**
  - 쓰레기 객체가 담긴 이미지, bbox 정보(좌표, 카테고리)
  - bbox annotation은 COCO format
- **Output**
  - bbox 좌표, 카테고리, score 값을 리턴.
  - submission 양식에 맞게 csv 파일을 만들어 제출
  - COCO format이 아닌 Pascal VOC format


## Dataset

- Total images: 9,754 (train: 4883, test: 4871)
- Image size: 1024 × 1024  
- Classes: 10 (General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing)  
- Annotation: COCO format (train only)

## Project Overview

본 프로젝트에서는 단일 모델의 성능 향상보다는  
**데이터 특성에 대한 이해와 모델 조합 전략**이 최종 성능에 더 큰 영향을 미친다는 점에 주목하였다.

EDA를 통해 데이터 분포와 특성을 분석하고, 이를 바탕으로 학습 및 전처리 전략을 설계하였다.  
단일 YOLO 기반 객체 탐지 모델은 제한적인 성능을 보였으나,  
서로 다른 오류 패턴을 가진 여러 모델을 결합함으로써 객체 탐지 성능을 효과적으로 개선할 수 있었다.

또한 WandB를 활용하여 실험 과정을 체계적으로 기록하고,  
모델 간 성능 비교와 학습 과정 분석을 효율적으로 수행하였다.

## Team Members

<table align="center">
    <tr align="center">
        <td><img src="https://github.com/user-attachments/assets/63c982d2-cc44-474c-9b73-c142627df75e" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/5c459428-9ffa-4506-b59d-a880a63413b9" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/ffd16ff0-3c70-4cd1-9f29-f9ce3beda107" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/9f4be4be-083c-4ce7-948b-6c1e57ed3ed9" width="140"></td>
        <td><img src="https://github.com/user-attachments/assets/a5fc0ec6-1645-4e2e-a4bd-a249b0f9c87a" width="140"></td>
    </tr>
    <tr align="center">
        <td><a href="https://github.com/hyeongseokgo" target="_blank">고형석</a></td>
        <td><a href="https://github.com/M1niJ" target="_blank">김민진</a></td>
        <td><a href="https://github.com/uss0302-cmd" target="_blank">류제윤</a></td>
        <td><a href="https://github.com/Ea3124" target="_blank">이승재</a></td>
        <td><a href="https://github.com/cuffyluv" target="_blank">주상우</a></td>
    </tr>
    <tr align="center">
        <td>T8012</td>
        <td>T8028</td>
        <td>T8065</td>
        <td>T8155</td>
        <td>T8199</td>
    </tr>
</table>


## Role

| Member | Roles |
|--------|-------|
| **고형석** | WandB 세팅, 데이터 EDA, 가설 검증, 전처리, 증강, cascade-rcnn, 백본 swin 구조 모델 실험 |
| **김민진** | 데이터 EDA, K-Fold, 데이터 증강, MMdetection 모델 실험 |
| **류제윤** | 데이터 EDA 및 전처리, Faster-rcnn, cascade-rcnn 모델 실험 |
| **이승재** | data re-labeling, 데이터 EDA, Albu를 통한 증강, yolov11n, yolov11l 모델 실험, Ensemble(NMS, WBF) 실험 |
| **주상우** | 데이터 EDA, MMdetection Faster-rcnn, DETR 모델 실험, EfficientDet-pytorch 모델 시도 |


## Repository Structure

```bash
.
├── base_codes
│   ├── README.md
│   ├── detectron2
│   ├── mmdetection
│   ├── requirements.txt
│   └── setup.py
│
├── ghs        # (고형석)EDA, Cascade R-CNN, Swin-Tiny/Large, Stratified K-Fold
├── jsw        # (주상우)EDA, DETR, EfficientDet
├── kmj        # (김민진)K-Fold, Faster R-CNN, Cascade R-CNN
├── lsj        # (이승재)Data Augmentation, Re-labeling, Ensemble, YOLOv11
├── rjw        # (류제윤)Cascade R-CNN vs Faster R-CNN comparison
│
├── README.md
└── ssh.md
