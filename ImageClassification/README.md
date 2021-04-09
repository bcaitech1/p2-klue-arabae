# Mask Image Classification


[ Overview 요약 ]  
COVID-19의 확산으로 공공 장소에 있는 사람들은 반드시 마스크를 착용해야할 필요성이 있으며, 무엇 보다도 코와 입을 완전히 카릴 수 있도록 올바르게 착용하는 것이 중요합니다.
따라서, 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템을 구축합니다.  



### Dependencies

모델을 훈련하고 테스트하는데 필요한 패키지를 다운받습니다.

```console
pip install -r requirements.txt
```  



### Training examples

- ResNet-50 model, Adam optimizer, F1 loss 사용시

```console
python ./train.py --model resnet --optim adam --loss f1 --model_name Test

```  


- Arguments 정리

    - seed: random seed
    - epochs: model 훈련 횟수
    - batch_size: training에서 input image의 batch size
    - model: 사용할 모델
    - optim: 사용할 최적화 함수
    - loss: 사용할 손실 함수
    - lr: learning rate (학습 step size)
    - val_ratio: train과 valid로 나눌 때 valid로 사용할 비율
    - isTrain: 훈련 할 경우 True이며, 테스트할 경우 False
    - model_name: 저장하거나 불러올 모델의 이름
    - chpkt_idx: 제출 파일을 만들기 위해 사용할 저장된 모델의 인덱스  



### Test examples

- Efficientnet-b4로 훈련한 모델 중 effi-5.pt를 불러와 제출 파일을 만들 경우

```console
python ./train.py --model efficient --model_name effi --chkpt_idx 5 --isTrain false
```  



### Implemented models, optimizer and loss functions

- 아래의 model과 optimzer 그리고 loss function을 사용하실 수 있습니다. 사용시 괄호안에 있는 것과 같이 매개변수를 넣어주세요.

```text
model: ResNet-50(resnet), Efficientnet-b4(efficient)
optimizer: SGD(sgd), Adam(adam), AdamP(adamp)
loss: CrossEntropy Loss(cross_entropy), F1 Loss(f1), Focal Loss(focal), Label Smoothing(label_smoothing)
```

<br/>

폴더 안 "./models/baseline_ver2/ckpht-10.pt"를 사용하여 submission 파일을 만들어 제출한 경우 Public data 기준 Accuracy 80.4127%, F1 score 0.7643 입니다.
