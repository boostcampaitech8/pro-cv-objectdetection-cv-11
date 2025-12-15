
# GHS folder

## 키워드
- EDA
- datapreprocessing(전처리 + train/val)
- Augmentation
- cascade_rcnn
- swin_tiny, swin_large
- ensemble


## 폴더구조

```
|-- faster_rcnn_inference.ipynb                       -> faster_rcnn 추론 베이스라인
|-- faster_rcnn_train.ipynb                           -> faster_rcnn + wandb 베이스라인
|-- ang.ipynb                                         -> 간단하게 WBF 앙상블
|-- kfold_val.ipynb                                   -> kfold를 통해 train과 validation json 파일을 나눠주는 코드
|-- cascade
|   |-- cascade_rcnn_train.ipynb                      -> cascade_rcnn 베이스라인
|   |-- cascade_rcnn_inference copy.ipynb             -> cascade_swin 추론
|   |-- cascade_rcnn_swin_train.ipynb                 -> cascade_swin 베이스라인
|   |-- cascade_rcnn_swin_trainval.ipynb              -> 주로 사용한 cascade_swin_large 혹은 tiny 학습 파일

|-- folds
|   |-- train_fold0~4.json                            -> kfold_val.ipynb을 통해 나온 train_fold
|   |-- val_fold0~4.json                              -> kfold_val.ipynb을 통해 나온 val_fold
|-- my_model
|   |-- cascade_rcnn_swin_large.py                    -> 기존 mmdetection에 있는 cascade_rcnn config를 기반으로 backbone을 swin_large로 교체한 모델 컨피그
|   |-- cascade_rcnn_swin_tiny.py                     -> 기존 mmdetection에 있는 cascade_rcnn config를 기반으로 backbone을 swin_tiny로 교체한 모델 컨피그
```


### 추가설명
- **전체적인 설명** : 올린 모든 코드들은 mmdetection 3.xx 버전안에서 진행된 코드들입니다. my_model은 mmdetection/config안에 넣고, folds는 해당 dataset폴더 안에 넣고 사용하면 됩니다.
  -   컨피그 파일 주소를 통해 tiny,large를 변경할 수 있고 train, val, test의 주소만 변경해주면 학습가능 할 것입니다.
- **cascade** : cascade 기본 베이스, swin을 추가한 모델을 사용하는 버전 모두 들어가있습니다. 추가로 추론부분에서도 config 주소를 수정하면 됩니다.
  -   large 모델의 경우는 pretrained가 약 1기가로, 계속 다운 받으면서 사용하기 어렵기에 따로 다운받아서 새로 pretrained에 넣어서 사용했습니다.

## Cascade Result

| Model | mAP |
| - | - |
| Cascade-rcnn-baseline | 0.3998 |
| Cascade-rcnn-swin-tiny | 0.4874 |
| Cascade-rcnn-swin-large | 0.5487 |
