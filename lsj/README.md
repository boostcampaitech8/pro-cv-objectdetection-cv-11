
# Welcome to lsj folder (이해를 돕기위한 설명들을 적어놓았습니다)

## 키워드
- EDA
- datapreprocessing(전처리 + train/val)
- Augmentation
- General trash re-labeling
- yolov11l&result
- ensemble
정도가 코드에 녹아있습니다.

## 폴더구조

```
.
|-- Ensemble.ipynb                                  -> 앙상블 기본 4가지 방법론 실행코드
|-- README.md                                       
|-- data_preprocesing_baseline.ipynb          -> baseline 전처리
|-- data_preprocesing_strong_aug(dataaug).ipynb     -> 강한 Augmentation & 전처리
|-- data_preprocesing_weak_aug(rev3).ipynb          -> 약한 Augmentation & 전처리
|-- general_trash_re_ann.ipynb                      -> 일반쓰레기 data re-labeling 코드
|-- into_ultralytics                                -> 아래에서 따로 설명
|   |-- yolov11l.py                                 -> yolov11l base+tuning 순차 작업 코드
|   |-- yolov11l_dataaug.py                         -> yolov11l dataaug용 학습코드
|   `-- yolov11l_dataaug3.py                        -> yolov11l rev3용 학습코드
|-- metrics_baseline.csv                            -> metrics 결과(baseline)
|-- metrics_dataaug-rev3.csv                        -> metrics 결과(rev3)
|-- metrics_dataaug.csv                             -> metrics 결과(dataaug)
|-- submission_yolov11l_baseline.csv                -> 실제 csv제출 파일(baseline)
|-- submission_yolov11l_dataaug-rev3.csv            -> 실제 csv제출 파일(rev3)
|-- submission_yolov11l_dataaug.csv                 -> 실제 csv제출 파일(dataaug)
|-- train_cleaned_general_trash copy.json           -> 아래에서 설명
|-- ultralytics                                     -> ultralytics레포 git clone
... 
|   |-- trash10-yolo11l                             -> 세부 결과 자료들(결과사진, 그래프)
|   |-- trash10-yolo11l-dataaug
|   |-- trash10-yolo11l-dataaug-rev3
...
`-- waste_dataset_eda.ipynb                         -> eda 자료들
```

### 추가설명
- **into_ultralytics** : ultralytics레포 git clone후 해당 폴더 안으로 넣어야 하는 3가지 파일들(그래야 해당 코드들이 경로 문제없이 잘 동작합니다) (you should put in .../lsj/ultralytics/(in here) after git clone)
   -  해당 폴더 안에 있는 python파일들은, 데이터 전처리 후 모델을 돌릴때 사용하는 코드입니다. (이때 tmux를 활용하여 백그라운드에서 돌리면 ssh server나가있어도 돌아갑니다.)
- **trash10-yolo11l 등 3가지 폴더** : 이것들은 원래 코드 실행되면 ultralytics 폴더 안에서 생성되게끔 해놓았고, 실제로 폴더 설명에도 ultralytics폴더 안에 있지만, git ignore가 되어있어 일부러 밖에 빼놓았습니다. 참고로 weight는 git ignore되어있어, 가중치를 확인하고 싶으면 실제로 다른 코드를 돌려야 합니다.
- **train_cleaned_general_trash copy.json** : general_trash_re_ann.ipynb 해당 코드를 통해 2105개의 general trash가 있는 image중 211개(10%)를 re-labeling한 json파일, 이 파일은 사용하고 싶다면 dataset안에 있어야 합니다. 

## QuickStart

1. git clone in **lsj folder's terminal** 
```
$ git clone https://github.com/ultralytics/ultralytics.git
```

2. Put in into_ultralytics's files to ultralytics

3. If you have dataset, do first data_preprocessing and yolov11l.py code
```
# if you want to do baseline :
data_preprocesing_baseline.ipynb -> yolov11l.py

# if you want to do strong aug :
data_preprocesing_strong_aug(dataaug).ipynb -> yolov11l_dataaug.py

# if you want to do weak aug :
data_preprocesing_weak_aug(rev3).ipynb -> yolov11l_dataaug3.py
```

4. additional codes

```
# if you want to do EDA, do it "waste_dataset_eda.ipynb"

# if you want to do re-labeling about General trash, do it "general_trash_re_ann.ipynb"

# if you want to do ensemble, do it "Ensemble.ipynb"
```

## Yolov11l Result

| Model | mAP |
| - | - |
| YOLO11l-baseline | 0.581 |
| YOLO11l-weak_aug | 0.567 |
| YOLO11l-strong_aug | 0.548 |