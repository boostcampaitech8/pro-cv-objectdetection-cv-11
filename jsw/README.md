# jsw 개인 작업 디렉토리
.gitignore에 등록된 디렉토리를 제외하고, 실제 코드들은 scripts 폴더에 전부 들어가 있습니다.

# 디렉토리 구조
```bash
scripts
|-- EDA
|   |-- EDA_251204_jsw.ipynb  => 기본 EDA 노트북.
|-- detr
|   |-- README.md             => detr 작업한 환경 설명.
|   |-- detr_inference.py     => detr 추론 코드.
|   |-- detr_train.py         => detr 최종 학습 코드 -> albu 없이 기본적인 데이터 증강만 사용함.
|   |-- detr_train_no_val.py  => validation set 없이 train과 test set으로만 구성된 학습 코드.
|   `-- detr_train_try_arg.py => 바로 위 코드에서 albu 데이터 증강을 시도한 코드 -> 적용 실패했음.
|-- efficientDet 
|   |-- train.py              => efficientDet-Pytorch repo에 있는 공식 train.py.
|   |-- train_jsw.py          => 위 코드를 우리 프로젝트에 적용하고자 시도한 코드 -> 적용 실패했음.
|   |-- validate.py           => efficientDet-Pytorch repo에 있는 공식 validate.py.
|   `-- validate_jsw.py       => 위 코드를 우리 프로젝트에 적용하고자 시도한 코드 -> 적용 실패했음.
|-- faster_rcnn
|   |-- faster_rcnn_inference.ipynb    => 기본 베이스라인에서 일부 코드를 수정한 faster rcnn 추론 코드.
|   `-- faster_rcnn_train.ipynb        => 기본 베이스라인에서 일부 코드를 수정한 faster rcnn 학습 코드.
|-- k-fold
|   |-- kfold_val_251208_jsw.ipynb     => ai-stages에서 제공된 기본 kfold_val 노트북에 BBOX 시각화를 추가한 노트북.
|   `-- kfold_val_251210.py            => 위 노트북을 기반으로 실제 .json 파일을 생성해주는 코드.
|-- utils
|   |-- errors_jsw.md                  => 초기 py310 가상환경 세팅 과정에 생긴 에러들 기록한 파일.
|   |-- gpu_check.sh                   => gpu 현 상황 체크에 쓰이는 리눅스 명령어들.
|   |-- gradient_accumulation_이란.md  => gradient_accumulation 사용 필요성에 대해 조사한 파일.
|   |-- train.sh                       => train_with_sh.py에 인자를 전달해 학습을 돌리는 쉘스크립트 파일.
|   |-- train_for_test.py              => train_with_sh.py를 작성하는 과정에서 테스트 용도로 사용한 파일.
|   |-- train_with_sh.py               => mmdetection repo의 공식 train.py를 변형해, 우리 프로젝트 구조에 맞게 적용한 파일. -> 실제 학습에 사용하진 않았음.
|   `-- 현재환경.md                     => 위에 gpu_check.sh 결과와 데이터 셋 크기 등을 조합해 현 상황에 적절한 하이퍼파라미터 탐구해 정리해준 파일. 
`-- yolo
    `-- yolov3_train.ipynb             => yolov3으로 학습을 돌리는 파일. 정상 작동함.
                                       => 이 외에도 yolov3_inference.ipynb가 있으나, boostcamp 과제 코드이므로 공개 x.
```

# 추가
- mmdetection 디렉토리 밖에서 모듈을 부르고 있기에, mmdetection에 대한 pip install . 이 필요합니다.