1️⃣ Gradient Accumulation 개념
🔹 문제 상황

GPU 메모리가 제한되어 있어 batch size를 크게 할 수 없음

예: batch size = 16으로 학습하고 싶은데 GPU 32GB에서는 batch size 4 이상 못 올림

🔹 Gradient Accumulation 해결책

작은 batch를 여러 번 forward/backward한 뒤, 한 번에 optimizer step 수행

이렇게 하면 GPU 메모리는 적게 쓰면서 effective batch size를 늘릴 수 있음

예시

GPU batch size = 4

원하는 effective batch size = 16

accumulation_steps = effective batch size ÷ GPU batch size = 16 ÷ 4 = 4

학습 과정:

batch 1 (4개 이미지) forward → backward → gradient 저장

batch 2 (4개 이미지) forward → backward → gradient 더해짐

batch 3 (4개 이미지) forward → backward → gradient 더해짐

batch 4 (4개 이미지) forward → backward → optimizer step 수행, gradient 초기화

이렇게 하면 메모리는 GPU batch size 기준만 사용, optimizer update는 effective batch size 기준으로 수행