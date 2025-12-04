
## 11조용 readme 초안

base_codes가 baseline이라고 보시면 됩니다.
내부에 train&inference코드들은 주석을 달아놓고 첫 커밋 시작했습니다.

gitignore에 

₩₩₩
base_codes/detectron2/*
base_codes/faster_rcnn/*
base_codes/mmdetection/*
base_codes/object_detection.egg-info

dataset/* 

sample_submission/*
₩₩₩
라고 되어있습니다. 

따라서 detectron2, faster_rcnn, mmdetection 내부 파일 같은경우에는 따로 다시 넣으셔서 setup 진행하시면 될 것 같습니다.

dataset과 sample_submission 내부도 ignore처리해놓았으나, sample_submission은 경우에 따라 풀어도 될 것 같습니다. 

5명 폴더 안에 각각 baseline을 넣기엔 용량이 문제가 될까봐 base_codes에서 연결해와서 쓰는 방향으로 설계했습니다..