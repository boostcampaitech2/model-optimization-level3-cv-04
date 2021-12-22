# ♻️ Model Optimization for Recycling Trash

</br>

Boostcourse AI Competition from [https://stages.ai/](https://stages.ai/)

</br>

## 👨‍🌾 Team

- Level 2 CV Team 4 - 무럭무럭 감자밭 🥔
- 팀 구성원: 김세영, 박성진, 신승혁, 이상원, 이윤영, 이채윤, 조성욱

</br>

## 🏆 LB Score

- Public LB: 1.1007 score (10등/39팀)
- Private LB: 1.0396 score (8등/38팀)
</br>

## 🎈 Main Subject
- 최근들어 분야를 막론하고 인공지능 기술은 사람을 뛰어넘은 엄청난 성능을 보여주고 있고, 때문에 여러 산업에서 인공지능을 이용해 그동안 해결하지 못 했던 문제들을 풀려는 노력을 하고 있음
- 대표적인 예로는 인공지능 분리수거 기계인 수퍼빈의 수퍼큐브가 있는데 이를 만들기 위해서는 즉각적인 쓰레기 분류 모델 필요
- 분리수거 로봇에 가장 기초 기술인 쓰레기 분류기를 만들면서 실제로 로봇에 탑재될 만큼 **작고 계산량이 적은** 모델을 만들어야 함
- 재활용 쓰레기 데이터셋에 대해서 이미지 분류를 수행하는 모델 설계

</br>

## ⚙ Development Environment
- GPU : Nvidia Tesla V100
- OS : Linux Ubuntu 18.04
- Runtime : Python 3.8.5
<br>

## 📥 Install Dependencies
```
pip install -r requirements.txt
```

<br>

## 🔑 Project Summary

### Dataset

- 전체 이미지 개수 : 26,068장
- 10 class : Metal, Paper, Paper pack, Plastic, Plastic bag, Styrofoam
- 학습데이터는 20,851장, 평가데이터는 5,217장으로
    - 평가데이터: Public 50%, Private 50%

### Metrics
<img width="500" alt="681be142-fe4c-4c7f-9586-f73eec436d8b" src="https://user-images.githubusercontent.com/58019653/147069393-a7cde571-fc60-49d1-88c9-88f387e2aca4.png">

- 기준모델 Score : 1.1950  
- 기준모델보다 적은 score를 내는 것이 목표  

<br>

## 🎢Run
### Train
```
python train.py --model_config ${path_to_model_config} --data_config ${path_to_data_config}
```
### Inference(submission.csv)
```
python inference.py --model_config configs/model/mobilenetv3.yaml --weight exp/2021-05-13_16-41-57/best.pt --img_root /opt/ml/data/test --data_config configs/data/taco.yaml3
```

## Reference
- Our basic structure is based on [Kindle](https://github.com/JeiKeiLim/kindle)(by [JeiKeiLim](https://github.com/JeiKeiLim))
