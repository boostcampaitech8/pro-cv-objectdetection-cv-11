
# PR #1 : 데이터 리라벨링 관련


1. 설치 라이브러리
```
pip install numpy matplotlib pillow
```

## 파일 내용
general_trash_re_labeling.py 
- COCO 포맷 annotation에서 General trash만 필터링하여 검토한다
- 이미지와 bounding box를 시각화하고, bbox 식제(d)·수정(e)을 입력으로 직접 수행 가능함
- 일정 주기마다 자동으로 JSON을 저장한다
- 마지막엔 라벨이 정제된 COCO annotation 파일(json) 생성

## PR 내용

Object Detection 학습을 위한 Data Preprocessing 기능 중, re-labeling관련 질문입니다.

현재의 코드 구성은, LLM의 도움을 받아 General trash에 대한 라벨링을 다시하는 코드를 구현했습니다. 코드 내용은 파일내용과 같습니다.

1. 현재 코드보다 라벨링을 할 수 있는 더 효율적인 방법이 있는지
2. 실무에서도 이런 re-labeling작업을 자주하는지, 그리고 이렇게 한다면 데이터에서 집중하는 부분이 어디고, 효율적으로 하기위해선 어떻게 진행하는지 궁금합니다.