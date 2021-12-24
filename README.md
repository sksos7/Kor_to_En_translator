<div align=center> </div>
 
**한영 번역기**
---

![image](https://user-images.githubusercontent.com/41771983/147334822-aca972ad-a513-4779-aea8-c940e94da313.png)

 
<img src="https://img.shields.io/badge/Python-green?style=plastic&logo=Python&logoColor=#3776AB"/> <img src="https://img.shields.io/badge/Jupyter-inactive?style=plastic&logo=Jupyter&logoColor=#F37626"/>
<img src="https://img.shields.io/badge/NumPy-important?style=plastic&logo=NumPy&logoColor=#013243"/>
<img src="https://img.shields.io/badge/pandas-ff69b4?style=plastic&logo=pandas&logoColor=#150458"/>
<img src="https://img.shields.io/badge/Tensorflow-blue?style=plastic&logo=TensorFlow&logoColor=#FF6F00"/>
<img src="https://img.shields.io/badge/Flask-9cf?style=plastic&logo=Flask&logoColor=#000000"/>


딥러닝(Sequence to Sequence with Attention)을 이용한 한영 번역기
---

<!-- ![01](https://user-images.githubusercontent.com/41771983/147331484-76ec4c8e-495c-4bbe-972c-b1e4abb531aa.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331484-76ec4c8e-495c-4bbe-972c-b1e4abb531aa.png" width="768" height="432"/>

---
<!-- ![02](https://user-images.githubusercontent.com/41771983/147331547-a34227d8-9ba5-40af-b17a-cf0f834edc39.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331547-a34227d8-9ba5-40af-b17a-cf0f834edc39.png" width="768" height="432">

데이터셋은 Manythings 와 AI 허브의 데이터셋을 이용했습니다.
1. http://www.manythings.org/anki/
2. https://aihub.or.kr/aidata/87

---
<!-- ![03](https://user-images.githubusercontent.com/41771983/147331551-508ba050-d3af-4658-95d3-5cb94b0d59b4.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331551-508ba050-d3af-4658-95d3-5cb94b0d59b4.png" width="768" height="432">


한글 형태소 분석기 Mecab을 사용했습니다.

https://github.com/Pusnow/mecab-ko-dic-msvc

---

<!-- ![04](https://user-images.githubusercontent.com/41771983/147331552-454dca36-2436-4cc3-b0af-c3add9fd8f3b.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331552-454dca36-2436-4cc3-b0af-c3add9fd8f3b.png" width="768" height="432">


---
<!-- ![05](https://user-images.githubusercontent.com/41771983/147331554-38512110-2511-4cbb-a02c-b97d9c37bf81.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331554-38512110-2511-4cbb-a02c-b97d9c37bf81.png" width="768" height="432">


처음에는 RNN 기반의 Sequence to Sequence 의 모델을 사용을 했으나
1. 고정된 크기의 context vector 에서 발생하는 정보 손실 문제
2. 문장이 길어졌을 때, 계산이 앞에서 부터 되다보니 앞쪽 단어의 정보가 손실되는 기울기 소실 문제

---
<!-- ![06](https://user-images.githubusercontent.com/41771983/147331556-cc94c65d-34b2-4c61-bac4-cee18671c434.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331556-cc94c65d-34b2-4c61-bac4-cee18671c434.png" width="768" height="432">


- 이를 극복하기 위해 Attention 매커니즘을 추가
 
- 이전 모델과는 다르게 인코더들이 정보들을 가지고 디코더에 넘겨줌으로써
고정된 길이의 context vector 에서 발생하는 정보 손실 문제와 기울기 소실 문제를 어느정도 해결

---

웹에서 구현한 번역기
---
<!-- ![08-1](https://user-images.githubusercontent.com/41771983/147331557-ebf1d8ce-2ec1-4d8d-b542-3cdd0bd1f014.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331557-ebf1d8ce-2ec1-4d8d-b542-3cdd0bd1f014.png" width="420" height="416">

<!-- ![08-2](https://user-images.githubusercontent.com/41771983/147331558-9d14cd88-d08b-4d6b-b07d-57fda9d686e2.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331558-9d14cd88-d08b-4d6b-b07d-57fda9d686e2.png" width="420" height="416">

<!-- ![08-3](https://user-images.githubusercontent.com/41771983/147331560-6a461e5c-5712-45b2-9567-55d05c1cf6a6.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331560-6a461e5c-5712-45b2-9567-55d05c1cf6a6.png" width="420" height="416">


---

<!-- ![09](https://user-images.githubusercontent.com/41771983/147331563-d2594632-1b6d-40bc-8035-3b7bb7c2b7fa.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331563-d2594632-1b6d-40bc-8035-3b7bb7c2b7fa.png" width="768" height="432">


RNN 기반의 문제
- 무조건 앞에서 부터 계산이 되기 때문에 병렬 처리 불가능
- 문장이 짧을 때는 체감이 안되지만, 문장이 길어질 경우 학습 속도가 현저하게 떨어짐.

---
<!-- ![10](https://user-images.githubusercontent.com/41771983/147331564-0f7535b1-7dde-41de-aaed-3d3750b502b4.png) -->
<img src="https://user-images.githubusercontent.com/41771983/147331564-0f7535b1-7dde-41de-aaed-3d3750b502b4.png" width="768" height="432">


이를 해결하기 위해
- Transformer 모델을 사용
- 하나씩 계산되던걸 행렬곱으로 한번에 처리
- 병렬 처리가 가능해지므로 학습 속도가 빨라짐
- 더 많은 데이터셋을 이용함으로서 더 높은 번역 성능을 낼거라고 예상

---

참조
---
1. http://www.manythings.org/anki/
2. https://aihub.or.kr/aidata/87
3. https://github.com/Pusnow/mecab-ko-dic-msvc
4. https://www.tensorflow.org/tutorials/text/nmt_with_attention
5. https://blog.promedius.ai/transformer/
