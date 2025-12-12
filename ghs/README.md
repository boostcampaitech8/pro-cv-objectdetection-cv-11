
# ghs folder

## 키워드
- EDA
- datapreprocessing(전처리 + train/val)
- Augmentation
- cascade_rcnn
- swin_tiny, swin_large
- ensemble


## 폴더구조

```
.          
|-- faster_rcnn_inference.ipynb

|-- faster_rcnn_train.ipynb
|-- ang.ipynb                            -> 간단하게 WBF 앙상블
|-- cascade
|   |-- cascade_rcnn_train.ipynb
|   |--
|   |--
|   |--
|   |--


|-- folds
|   |-- train_fold0~4.json
|   |-- val_fold0~4.json
|-- my_model
|   |-- cascade_rcnn_swin_large.py
|   |-- cascade_rcnn_swin_tiny.py
```


### 추가설명
- **전체적인 설명** : 올린 모든 코드들은 mmdetection 3.xx 버전안에서 진행된 코드들입니다. my_model은 mmdetection/config안에 넣고, folds는 해당 dataset폴더 안에 넣고 사용하면 됩니다.
  -   컨피그 파일 주소를 통해 tiny,large를 변경할 수 있고 train, val, test의 주소만 변경해주면 학습가능 할 것입니다.
- **cascade** : ultralytics레포 git clone후 해당 폴더 안으로 넣어야 하는 3가지 파일들(그래야 해당 코드들이 경로 문제없이 잘 동작합니다) (you should put in .../lsj/ultralytics/(in here) after git clone)
   -  해당 폴더 안에 있는 python파일들은, 데이터 전처리 후 모델을 돌릴때 사용하는 코드입니다. (이때 tmux를 활용하여 백그라운드에서 돌리면 ssh server나가있어도 돌아갑니다.)

## Cascade Result

| Model | mAP |
| - | - |
| Cascade-rcnn-baseline | 0.3998 |
| Cascade-rcnn-swin-tiny | 0.4874 |
| Cascade-rcnn-swin-large | 0.5487 |
