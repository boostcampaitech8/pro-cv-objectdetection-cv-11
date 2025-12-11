# 1. pip install . 에러
pip install . 하니까 torch 없다고 에러 남.
근데 torch 실제론 (py310) 가상환경에 존재함!! 왜 없다고 하지??
llm 물어 보니 setup.py는 기본적으로 빌드를 위한 가상 환경 새로 만들어서 거기서 빌드하려고 한다는데,
그 가상 환경에서는 torch가 없어서 오류 나는 거라고 함.
--no-build-isolation 붙여서 하면 내 가상 환경(py310)에서 그대로 빌드할 수 있어서 오류 안 난다고 함.
pip install --no-build-isolation .

그러나 이젠 다른 오류가 발생:
error: metadata-generation-failed

시도1 (참고:https://alstn59v.tistory.com/73)
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel
해결 안 됨. 

시도2 (참고:https://jheaon.tistory.com/entry/Error-Error-metadata-generation-failed-Encountered-error-while-generating-package-metadata-%EC%98%A4%EB%A5%98)
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
**해결됨!!!**
README.MD에 있는 torch 및 CUDA 버전으로 맞춰서 재설치를 해주니 됐다.

이제 test용으로 faster_rcnn_train.ipynb 및 faster_rcnn_test.ipynb를 실행해 본 결과, output.csv가 잘 실행되는 것을 확인했음(리더보드 제출 결과 0.4177)

# 2. root 계정으로 mmdetection 수행 에러
mmdetection에 대해, jupyter notebook faster_rcnn_train.ipynb 시도하니까
Running as root is not recommended. Use --allow-root to bypass.
에러 나옴.
root 계정으로 시도해서 그럼. 사용자 계정으로 뭐 로그인할 수는 없으니까,
llm 도움 받아 --allow-root 붙여서 함.
jupyter notebook faster_rcnn_train.ipynb --allow-root

음 근데 이러니까 그냥 주피터 노트북 웹브라우저가 열리네.
생각해보니 우린 지금 vscode로 ssh 연결해서 있으니까 사실 그냥 주피터 노트북 그대로 실행하면 되는 거였음.