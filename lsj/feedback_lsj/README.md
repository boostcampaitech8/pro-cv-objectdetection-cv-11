
# PR #1 : 데이터 리라벨링 관련


1. 설치 라이브러리
```
pip install numpy matplotlib pillow
```



## PR 내용

Object Detection 학습을 위한 Data Preprocessing 기능 중, re-labeling관련 질문입니다.

현재의 코드 구성은, LLM의 도움을 받아 General trash에 대한 라벨링을 다시하는 코드를 구현했습니다.

첫번째로 현재의 re-labeling 구성과 그 다음으로 현재 코드보다 라벨링을 할 수 있는 더 효율적인 방법이 있는지에 관한 피드백을 받고 싶습니다.