# KLUE - Relation Extraction

[ Overview 요약 ]  

문장의 단어(Entity)에 대한 속성과 관계를 예측하여 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에 사용됩니다.


### Training examples

[기본 설정]
- Optimizer: Adam
- Loss: LabelSmoothing
- Stratified K-fold (K=8)
- Seed: 42

[실행]

R-Roberta 모델을 사용하고, Smoothing rate는 0.5로 두어 Test 이름으로 저장 (이때, wandb에 baseline이라는 project로 log를 저장)

```console
python ./train.py --project_name baseline --model r_roberta --smoothing 0.5 --model_name Test

```  


- Train Arguments 정리
    - project_name: wandb에 저장할 Project 이름
    - seed: random seed
    - epochs: model 훈련 횟수
    - batch_size: training에서 input image의 batch size
    - model: 사용할 모델 (kobert, multi, koelectra, roberta, r_roberta)
    - lr: learning rate (학습 step size)
    - smoothing: label smoothing loss의 smoothing 정도
    - dp: Classifier 전에 Dropout을 적용할 정도
    - train_file: 훈련에 사용할 train file 이름 (train, gold_train, pororo_train, gold_pororo_train, ner_train)
    - train_dir: train file이 저장된 경로
    - isAug: 많은 외부 데이터를 사용하여 augmentation할 경우 True, 사용하지 않으면 False
    - model_name: 저장하거나 불러올 모델의 이름  

- Test Arguments 정리
    - seed: random seed
    - model: 평가 모델 (kobert, multi, koelectra, roberta, r_roberta)
    - test_dir: test file이 저장된 경로
    - test_file: 평가할 test file 이름 (test, ner_test)
    - model_name: 저장하거나 불러올 모델의 이름  


### Test examples
Test 이름으로 저장한 R-Roberta를 평가

```console
python ./evaluation.py --model r_roberta --model_name Test
```  



### Using Shell script

훈련과 테스트를 한번에 수행할 수 있습니다.
command_file.txt파일에 훈련에 사용할 Arguments를 저장한 뒤 run.sh파일을 실행시킵니다.

주의! command_file.txt파일 마지막에 공백이 필요합니다.

command_file.txt파일
```text
--model_name kfold --model roberta
--model_name r_kfold_gold --train_file gold_train --model r_roberta

```

run.sh파일
```bash
#!/bin/bash

while  read line
do
    python train.py $line
    python evaluation.py $line

done < command_file.txt
```



### Results
- 3개의 모델을 Hard voting을 사용한 앙상블 결과로 제출하였을 때, 81.8% 성능을 보였습니다.

1. original train 데이터를 사용한 R-Roberta 모델
  - Batch size: 64
  - epoch: 15 (valid set의 Acc가 best인 모델만 저장)

R-Robeta는 [R-BERT 모델](https://github.com/monologg/R-BERT)을 reference하여 수정한 모델입니다. 구조는 아래와 같이 사용하였습니다.

![R-Roberta Framework](https://user-images.githubusercontent.com/46676700/115904175-72efb700-a49f-11eb-94a7-c67988807953.png)

2. NER Tag를 추가한 train 데이터를 사용한 Roberta 모델
  - Batch size: 16 (Colab 환경에서 수행하여 작은 batch를 사용했습니다.)
  - epoch: 30 (valid set의 Acc가 낮으면 저장하지 않음)
    - 사용 chkpt_list: [27, 22, 28, 11]

3. Gold-standard-v1 데이터를 추가한 train 데이터를 사용한 KoElectra 모델
  - Batch size: 32
  - epoch: 15 (valid set의 Acc가 낮으면 저장하지 않음)
    - 사용 chkpt_list: [8, 11, 9, 4, 6, 4]
