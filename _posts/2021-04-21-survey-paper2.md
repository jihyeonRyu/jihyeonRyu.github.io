---
layout: post  
title: (Federated Learning in Mobile Edge Networks) A Comprehensive Survey   
subtitle: AI Survey     
tags: [ai, ml, computer vision, federated learning, mobile edge networks, data privacy]    
comments: true  
---  

```
Proceeding: 2020    
Authors: Wei Yang Bryan Lim, Nguyen Cong Luong, Dinh Thai Hoang, Yutao Jiao, Ying-Chang Liang, Qiang Yang, Dusit Niyato, Chunyan Miao     
```

[Paper Link](https://arxiv.org/pdf/1909.11875.pdf)  

## 1. Introduction
요새는 7억만의 IoT 디바이스와 3억만의 스마트폰이 존재한다. 
이런 디바이스들은 진화된 센서와 컴퓨팅, 그리고 소통 능력을 가지고 있다. 
따라서 의료 목적 및 대기 질 모니터링을 위해 다양한 크라우드 센싱 작업에 잠재적으로 배포될 수 있다. 
딥러닝의 발전에 따라, 의미있는 연구와 application을 위해서 수 많은 날씨 데이터를 end device로 부터 수집할 수 있다. 

잔통적인 cloud-centric 접근 방식은, 모바일 디바이스로부터 데이터를 수집하고 업로드 한뒤, 중앙 서버에서 데이터를 처리하는 과정을 거쳐야 했다. 
특히, IoT 디바이스나, 스마트폰에서 수집한 측정값이나, 사진, 비디오, 위치 정보를 data center에 수집하였다. 
이런 데이터들은 inference model을 위해서 효과적인 인사이트를 주지만, 이런 접근 방법은 아래와 같은 이유로 더이상 적합하지 않다. 
* 데이터 소유자의 프라이버시 민감도가 증가함에 따라, 데이터 프라이버시 법규가 제정되었다. 
* cloud-centric 접근 방식은 불필요한 긴 지연시간이 걸림
* 클라우드로 데이터를 전송하면 비정형 데이터와 관련된 작업이 백본 네트워크에 큰 부담을 줌  

요즘에는 데이터 소스는 클라우드의 밖에 위치하므로, Mobile Edge Computing(MEC)은 자연스럽게 최종 디바이스 및 에지 서버의 컴퓨팅 및 스토리지 기능을 활용하여
모델 교육을 데이터가 생성되는 위치에 가깝게 제공하는 솔루션으로 제안되었다. 

기존 MEC 접근 방식의 모델 훈련의 경우, 컴퓨팅 집약적인 작업이 클라우드로 오프로드되기 전에, 훈련 데이터가 low-level의 DNN까지 training을 먼저 하기 위해, 에지 서버로 먼저 전송하는 협업 패러다임이 제안되었다.
![](./../assets/resource/survey/paper2/1.png)  
하지만 이 방식도 계산 비용이 많이 들고 지속적인 학습이 필욯한 application에는 적합하지 않다. 그리고 여전히 민감한 개인 정보의 이동을 요구한다. 
이는 사용자가 모델 학습에 참여하는 것을 꺼리게 만든다. 
비록 다양한 프라이버시 보호 방법이 있음에도 불구하고 많은 사용자들은 그들의 개인정보를 외부의 서버에 노출되지 않기를 바란다.

학습 데이터가 개인 디바이스에 보존 되는 것을 보증하기 위해 분산된 디바이스에서 학습하는 decentralized ML 방식인 Federated Learning(FL)이 등장하였다. 
FL 에서는 모바일 디바이스는 그들의 local data를 사용하여 FL 서버와 협동적으로 학습을 수행한다. 그리고 오로지 학습된 모델의 weight만을 FL 서버로 전송한다.
이 과정은 원하는 정확도에 도달할때 까지 여러번 수행한다. 
이로인해 ML 모델의 학습은 모바일 edge network에서 수행될 수 있다.
전통적인 cloud-centric training 접근방식과 비교하여 FL은 아래와 같은 강점을 지닌다. 
* 네트워크 대역폭의 매우 효울적인 사용: 데이터 전체를 전송할 필요 없이 오로지 모델의 파라미터만 전송하므로 communication 비용이 줄어든다. 
* Privacy: 클라우드로 사용자의 데이터가 전송될 필요가 없음. 이는 사용자로 하여금 모델 학습에 참여할 의지를 키울 수 있고 더 나은 모델을 학습시킬 수 있음
* Low latency: 모델이 지속적으로 학습되고 업데이트 될 수 있음.

FL은 최근 몇가지 application에서 성공을 거두고 있다. 
예를 들어 Federated Averaging 알고리즘 (FedAvg)는 Google의 Gboard에 적용되어 다음 단어를 예측 하는 모델의 발전을 가져왔다. 
또한 FL은 health AI의 진단 모델 개발에 사용되고 다영한 병원들과 정부 기관들간의 협동을 가능하게 했다. 

점점더 복잡해지는 모바일 edge network의 계산과 저장 제약을 감안할 때, 정적 모델을 기반으로 구축한 기존 네트워크 최적화 접근 방식은 동적 네트워크 모델링에 비해 상대적으로 열악하다. 
따라서 리소스 할당의 최적화를 위한 data-driven 딥러닝 접근 방식이 점점 인기를 얻고 있다. 
예를들어, 딥러닝은 네트워크 조건의 representation learning에 사용될 수 있는 반면, 강화학습은 동적 환경과 상호작용하면서 의사결정을 최적화 할 수 있다. 
하지만 앞서 말한 접근은 사용자 데이터가 input으로 들어가고 이런 데이터는 매우 민감하거나 자연적으로 접근 불가능할 수 있다.

FL을 수행하기 전에 몇가지 해결해야할 도전 과제 들이 있다. 
* 클라우드 서버로 raw 데이터가 전송될 필요가 없다 하도라도 고차원의 몯ㄹ을 업데이트 시키기 위해서는 여전히 통신 비용과 참가하는 모바일 디바이스의 통신 대역의 한계가 존재한다.
* 많고 복잡한 모바일 edbge network에서 참가하는 디바이스간의 다양성(데이터 질, computation power, 의지)이 잘 관리 되어야 한다.
* 악의적인 참가자 또는 서버가 있는 경우 개인정보를 보장하지 않는다. 최근 연구 결과에 따르면 악성 참여자가 FL에 존재할 수 있으며, 공유된 매개 변수 만으로 다른 참여자의 정보를 유추할 수 있음이 분명해졌다.
 그래서 프라이버시와 보안 이슈는 FL에서 여전히 고려되어야 한다. 
 
독자들의 편리함을 위해 저자는 이 서베이에서 논의한 관련 연구들을 아래와 같이 분류하였다.  
![](./../assets/resource/survey/paper2/2.png)  

* FL at mobile edge network: 도전 과제를 해결하고 collaborated training 적용 관련 이슈 정리
* FL for mobile edge network: FL 최적화를 위한 application 탐색 

## 2. Background and Fundamentals of Federated Learning

### Federated Learning
FL 시스템에느 두개의 main entity인 data owners(participants)와 model owner(FL server)가 있다. 
*N* = {1, 2, ..., N}을 owner들의 개인적인 dataset D라고 하자. 
각 owner는 자신의 데이터인 D_i를 local model w_i의 학습에 사용한다. 그리고 오직 local model의 파라미터만을 FL 서버로 보낸다.
그리고 모아진 모든 local model 파라미터들을 global model w_G를 만들기 위해 더한다.
이것은 전통적인 중앙적인 학습 방식과 다르다.
FL의 일반적 구조와 학습 프로세스는 아래와 같다.   
![](./../assets/resource/survey/paper2/3.png)  

FL 학습 과정은 다음과 같은 세 단계를 거친다.
* Step1 (Task Initialization): 서버에서 테스크와 목표 application 그리고, 필요한 데이터를 정의한다. 
또한 global model의 hyper parameter와 학습 과정을 정의한다. 그리고 서버는 initialized global model과 task를 선택된 participants에 배포한다.
* Step2 (Local model training and update): global model에 기반하여 각 참가자는 local data를 사용하여 local model parameter를 업데이트한다. 
각 참가자들의 목적은 interation t 동안 loss를 minimize하는 parameter w_i(t)를 찾는 것이다. 그리고 이 local model parameter를 server로 보낸다.
* Step3 (Global model aggregation and update): local model parameter를 누적하고 업데이트된 글로벌 모델 파라미터를 다시 data owner에게 보낸다. 
이때 서버도 모델의 global loss function을 원하는 만큼 minizie 하거나 원하는 정확도에 도달할때 까지 이를 반복한다.
classical한 누적 방식은 Algorithm1 처럼 FedAvg 방식이었다.   
![](./../assets/resource/survey/paper2/4.png)  

### Statistical Challenge of FL
전통적인 분산된 ML은 center server가 모든 학습 데이터셋에 접근할 수 있었다. 
서버는 학습 데이터셋을 비극한 분포를 가진 subset으로 분리할 수 있었다. 그리고 이 subset은 참가자들의 노드에 분산 학습을 위해 분배되었다. 
하지만 FL은 local dataset은 오직 데이터의 소유자만 볼 수 있기 때문에 이런 접근 방식은 불가능하다.

FL 셋팅에서는 local dataset은 서로 다른 분포인 non-IID이다. 
그럼에도 FedAvg 알고리즘의 저자는 이런 non-IID에서도 원하는 정확도를 달성할 수 있음을 보여줬지만, 다른 저자는 그렇지 않은 것을 발견했다. 
예를들어 FedAvg로 학습한 CNN 모델은 centrally 학습된 모델 대비 51% 더 낮은 정확도를 보였다고 한다.
이 발견은 Earth mover's distance (EMD)와 비교하여 FL 참가자의 데이터 분포 차이로 정량화 되는 것을 볼 수 있다.   
만약에 데이터가 매우 non-IID이고 치우쳐저 있을 경우, 모든 클래스에 대해서 uniform한 분포를 가지도록 data-sharing 할 경우
각 참가자가 local model을 자신의 privcate data와 공유 받은 5%의 데이터로만 학습해도 EMD가 줄어들기 때문에 global model이 30%의 정확도 향상을 가져왔다. 

하지만 공통의 dataset은 항상 FL 서버를 통해서 공유 가능하지 않을 것이다. 대체 솔루션은 공통 데이터 셋 구축에 대한 기여도를 수집하는 것이다.
어떤 저자는 모든 FL 참가자로부터 수집한 data를 사용할 경우 class imbalance를 초례하고 이는 결국 모델의 정확도를 떨어뜨리는 문제를 발생시킴을 발견했다. 
그래서 Astraea framework가 제안되었다.   

initialization에서 FL 참가자들은 그들의 data distribution을 서버에 보낸다.
그리고 training step 전에 rebalancing step을 추가하였는데, 각 참가자들은 minority class에 대해서 data augmentation을 수행한다. 
(random rotation, shift) 이런 augmented data 로 학습이 끝난 후에, FL server로 파라미터를 보내기 전에, 중재자는 참가자들의 data distribution(uniform distribution에 가까 운 것을 찾음)을 통해서 최고의 기여자를 선택한다. 
이는 greedy 알고리즘을 통해 찾고 local data와 uniform distribution간의 KL Divergence가 최소가 되는 것을 찾는다. 

각 디바이스마다 서로 다른 데이터 분포를 갖을 경우, multi-task learning의 개념을 가져오는 많은 연구들이 생긴다. 
일반적으로 사용해오던 loss를 그대로 사용하는 것이 아니라 task 간의 관계를 고려하여 loss function을 수정한다.  
MOCHA(Matroid Optimization: Combinatorial Heuristics and Algorithms, [Source Link](https://github.com/coin-or/MOCHA)) 알고리즘은 최소화 문제를 approximate 문제로 푸는 대체 최적화 알고리즘으로 제안되었다. 
흥미롭게 MOCHA는 참가자들의 리소스 제한에 맞게 조정되게끔 설계되었다.
예를들어 approximate 정도는 네트워크 컨디션과 디바이스의 cpu 상태에 맞게 adaptively 하게 조정된다. 
하지만 MOCHA는 non-convex DL model에는 사용할 수 없다. 
FEDPER(Federated Learning with Personalization Layers) 접근 방식은 통계적인 이질성을 처리하는 multi-task learning 개념을 사용한다.
참가자는 FedAvg 알고리즘을 이용해서 base layer를 학습하고 이를 공유한다.
그 다음, 각 참가자는 각자의 데이터를 가지고 personalization layer를 학습한다. 
특히 이 방식은 참가자의 다양한 선호가 존재하는 추천 시스템에 적합하다. 
그러나 각 참가자는 개인화된 모델을 학습시키기 위한 로컬 데이터 샘플이 충분하지 않기 때문에 base layer의 공동 훈련의 높은 정확도가 중요하다는 점에 주목해야한다. 

데이터간의 이질성때문에, 분산 학습 알고리즘의 수렴은 항상 고민거리였다.
높은 수렴율은 각 참가자들에게 많은 시간과 리소스를 줄일 수 있게 해주고, 또한 적은 수의 커뮤니케이션 라운드가 참가자의 dropout을 줄이고
연합 교육의 성공률을 크게 높인다. 

FedProx 알고리즘은 training loss가 증가할 때, 모델 업데이트가 현재의 파라미터에 영향을 적게 받도록 adaptive하게 조정한다. 
비슷하게 LoAdaBoostFedAvg 알고리즘은 의학 데이터와 같은 data-sharing 접근 방식의 보안 방식이다. 
참가자들은 그들의 로컬 데이터를 이용해서 모델을 학습하고 현재의 CE를 이전 훈련 라운드의 손실 중앙값과 비교한다. 
만약 현재의 CE가 더 높으면 모델은 global aggregation 하기 전에 학습을 다시 진행함으로써 학습의 효율성을 높일 수 있다. 

또한 Communication cost는 이런 fast convergence로 어느정도 감소 시킬 수 있다.

### FL Protocols and Frameworks
확장성을 개선하기 위해 아래와 같은 FL 프로토콜이 제안되었다.  
![](./../assets/resource/survey/paper2/5.png)  
이 프로토콜은 디바이스와의 불안정한 연결과 통신상의 안정성 이슈를 해결한다. 
프로토콜은 아래와 같이 3개의 phase로 구성되어 있다. 
1) Selection: FL 서버는 training round에 참가할 디바이스들의 subset을 선택한다. 
이 기준은 후에도 서버의 필요성에 따라 보정될 수 있다. 
2) Configuration: 서버는 선호하는 aggregation 매커니즘에 따라 구성된다. 그 다음 서버는 각 참가자에게 학습 일정과 global model을 보낸다.
3) Reporting: 서버는 각 참가자로부터 업데이트를 받는다. 그 다음 업데이트는 FedAvg 알고리즘을 통해 집계된다. 

FL 참가자의 규모에 따라 장치 연결을 관리하려면 속도 조정도 권장된다. 
속도 조정은 참가자가 FL 서버에 다시 연결할 수 있는 최적의 시간을 adaptively 관리한다. 
예를들어, FL 참가 규모가 작을 경우엔 충분히 참가 디바이스가 서버에 동시에 연결될 수 있다. 
반면에 참가 규모가 클 경우에는, 랜덤하게 디바이스를 선택하여 너무 많은 참가 디바이스가 동시에 연결되지 않도록 조정해야한다. 

통신 효율과는 별개로 local update가 일어날때 통신 안정성도 또 다른 해결해야할 문제이다. 여기에는 주로 두가지 관점이 있다. 
* Secure Aggregation: 로컬 업데이트가 추적되고 FL 참가자의 신원을 추론하는데 사용되지 않게 하기 위해 신뢰 가능한 third party server가 local model aggregation을 위해 배포된다. 
또한 비밀 공유 매커니즘으로 인증된 암호화를 사용하여 local update를 전송하는데 사용한다. 
* Differential Privacy: FL 서버가 local update의 owner의 신원 정보를 파악하는 것을 받지한다. 이를 위해 original local update에 모델 정확도를 해치지 않는 수준의 일정한 noise를 추가한다. 

최근의 몇몇 open-source 프레임워크를 통해 FL을 배포할 수 있다. 
* __Tensorflow Federated(TFF)__: TFF는 두 가지 layer로 구성되어 있다. 
    * FL: 존재하는 tf model에 FL 알고리즘을 개인적으로 추가할 필요 없이 FL을 적용할 수 있게 해주는 high-level interface
    * Federated Core(FC): TF layer와 통신 operator를 통해 결합하여, 사용자가 커스텀화 되고 새로 디자인된 FL 알고리즘을 사용할 수 있게 한다. 
* __PySyft__: PyTorch 프레임워크에 기반하여서 암호화되고 프라이버시가 보존되는 DL을 수행할 수 있게 한다. 
PySyft는 native Torch interface를 유지하도록 개발되었다. 즉. 모든 텐서 작업을 실행하는 방법은 PyTorch의 방법과 변경되지 않는다. 
SyftTensor가 생성되면 LocalTensor는 자동적으로 생성되어 기본 PyTorch 텐서에도 입력 명령을 적용한다. 
FL을 시뮬레이션 하기 위해, 참가자는 Virtual Workers로 생성된다. 그리고 data는 각 virtual worker에 분산되어 들어간다. 
그리고 data 주인과 저장 위치를 특정하기 위해 PointerTensor가 생성된다. 또한 global aggregation을 위해 virtual workers에서 모델 업데이트를 가져올 수 있다. 
* __LEAF__: FL에서 벤치마크로 사용할 수 있는 오픈소스 프레임워크의 데이터셋 이다. 예를들어 Federated Extended MNIST는 작성자를 기준으로 분할된 데이터세트를 만든다.
각 데이터세트의 작성자는 FL의 참여자로 간주되며 데이터는 로컬 데이터로 간주 된다. 이런 데이터 세트에 새로 설계된 알고리즘을 구현하면 연구간 신뢰할 수 있는 비교가 가능하다. 
* __FATE__: Federated AI Technology Enabler의 줄임말로 WeBank에서 개발한 오픈소스 프레임워크이다. 

### Unique Characteristics and Issue of FL
FL은 분산 ML에 비해 독특한 특징을 가진다. 
1) Slow and Unstable Communication
2) Heterogeneous Devices
3) Privacy and Security Concerns

## 3. Communication Cost
high dimensionality의 업데이트로 인해 많은 통신 비용이 발생하고 학습의 bottleneck이 생길 수 있다. 
또한 이 bottleneck이 더 심해지는 이유는 참가 device의 불안정한 네트워크 상태와 업로드 속도가 다운로드 속도보다 빠른 속도 불균형에 있다.
그래서 통신 효율성을 증가시키는게 필요하다. 아래와 같은 접근 방식으로 통신 비용을 줄이는 것이 고려되고 있다. 
* __Edge and End Computation__: FL 셋업에서 종종 통신 바용이 계산 비용을 뛰어 넘을 때가 있다. 왜냐하면 디바이스 내의 데이터셋은 상대적으로 작고 점점 참가자들의 모바일 디바이스의 프로세서는 빨라지고 있기 때문이다.
반면에 참가자들은 모델 학습을 오직 Wi-Fi에 연결되어 있을 때만 수행하길 원한다. 
따라서 모델 학습에 필요한 통신 라운드 수를 줄이기 위해 각 global aggregation 전에 edge node 또는 최종 디바이스에서 더 많은 계산을 수행할 수 있다. 
또한 알고리즘의 빠른 수렴이 보장되면 edge server 및 최종 장치에서 더 많은 계산을 수행하는 대신 관련된 라운드 수를 줄일 수 있다. 
* __Model Compression__: 분산 학습에 공통적으로 쓰이는 기법이다. 모델 또는 gradient 압축은 업데이트 관련 통신을 간결하게 할 수 있다. 
완전한 업데이트 통신 보다는 sparsification, quantization, subsampling 등으로 압축하여 업데이트한다. 
하지만 이런 압축으로 인해 노이즈가 발생할 수 있으므로 각 라운드 동안 전송되는 업데이트의 사이즈를 줄이면서도 학습 모델의 품질을 유지하는 것이 목표이다. 
* __Importance-based Updating__: 각 라운드에서 오직 중요하거나 관련 있는 업데이트만을 선택하여 통신하는 전략이다. 실제로 통신 비용을 절약하는 것 이외에도 참가자 일부 업데이트를 생략하면 global 모델의 성능을 향상시킬 수도 있다. 

### Edge and End Computation
![](./../assets/resource/survey/paper2/6.png)  
global aggregation을 수행하기 전에 참가자의 end device에서 더 많은 computation을 수행하여야 communication round의 수를 줄일 수 있다.
참가자 디바이스의 계산량을 증가시키는 방법은 두가지 이다.  
(1) 각 라운드마다 참가자 수를 늘려 병렬화를 늘린다.  
(2) 각 참가차가 global aggregation을 위한 업데이트 전에 더 많은 계산을 수행한다.   
FederatedSGD 알고리즘과 FedAvg 알고리즘으로 이 두가지를 비교한다. 
FedSGD 알고리즘은 모든 참가자가 참여를 하며, 각 트레이닝 라운드마다 오직 하나의 pass만을 수행한다. 이는 마치 minibatch 사이즈가 각 참가자의 데이터셋크기와 같은 것을 의미한다.
이것은 centeralized DL에서 full-batch 학습과 비슷하다.
FedAvg 알고리즘을 위해서는 참가자가 더 많은 로컬 계산을 수행하도록 hyperparameter가 조정된다. 예를들어 각 참가자들은 각 통신 라운드에서 dataset epoch를 늘리거나 minibatch 사이즈를 더 작게 해서 훈련하게끔 한다. 
시뮬레이션 결과에서 (1)과 같은 병렬화는 특정한 threshold를 도달하는 데까지의 통신 비용을 줄이는데 효과가 없음을 보여줬다.
(2)처럼 선택된 참가자의 수를 유지하면서 참가자의 계산을 늘리는 것이 더 효과적이었다.
Federated Stochastic Block Coordinate Descent(FedBCD) 알고리즘은 각 참가자들은 global aggregation을 위한 통신 전에 multiple한 local update를 수행한다.
게다가 각 통신마다 approximate 보정이 적용되면서 convergence가 보증한다. 

또다른 통신 비용을 줄이는 방법은 수렴 속도를 증가시키게끔 학습 알고리즘을 바꾸는 것이다.
transfer learning과 domain adaptation에서 공통적으로 사용하는 two-stream 모델을 사용한다. 
각 학습 라운드에서 참가자는 global model을 수신하고 훈련 과정에서 참조하기 위해 이를 고정한다. 그리고 학습 중에 참가자는 이 고정된 global model을 참조하여 자기 자신의 local data 뿐만 아니라 다른 참가자들도 고려하여 학습할 수 있다. 
이는 Maxmimum Mean Discrepancy(MMD)를 loss function에 사용함으로써 달성할 수 있다.
MMD는 두 데이터의 분포 평균 차이를 알려준다. 로컬 모델과 글로벌 모델간의 MMD loss를 최소화함으로써, 참가자들은 더욱 global model로부터 일반화된 특징을 추출할 수 있고
결국 수렴 속도를 가속화 함으로써 통신 비용을 줄일 수 있다. 

edge server와 참여자 간의 전파 지연시간이 더 짧고 edge 서버는 중간 매개 변수 집계자의 역할을 할 수 있기 때문에 Cloud와 참여자간의 통신 비용을 줄일 수 있는 방법으로 고안되었다. 
HierachicalFL(HierFaVG) 알고리즘은 edge server가 local model의 파라미터들을 집계한다. 
미리 정의된 수의 edge server 집계 수가 만족되면 ede server는 cloud와 통신하여 global model aggregation을 수행한다.
따라서 참가자와 클라우드간의 통신은 여러 로컬 업데이트 간격 후에 한번만 발생한다. 
edge aggregation을 더 많이 사용할 수록 통신 비용을 더 줄일 수 있다. 
하지만 non-IID 데이터에 대해서는 edge와 cloud간의 차이가 커서 수렴이 더 잘 안될 수 있다. 하지만 여전히 통신 비용을 줄일 수 있고, remote cloud의 부담을 줄일 수 있기 때문에
좋은 접근 방법이다.

### Model Compression
참가자가 FL 서버로 보내는 모델 업데이트의 사이즈를 줄이기 위해 사용되는 방법이다.
구조화된 업데이트는 참가자 업데이트가 미리 지정된 구조를 갖도록 제한한다. 
low-rank 구조의 경우 각 업데이트는 두 행렬의 곱으로 표현되는 low-rank 행렬이되도록 적용됩니다.(W = W1 * W2 => nxn = nx1 * 1xn) 
여기서 하나의 행렬(W1)은 무작위로 생성되고 각 통신 라운드 동안 일정하게 유지되는 반면 다른 하나(W2)는 최적화됩니다. 
따라서 최적화 된 매트릭스(W2) 만 서버로 보내면됩니다.
random mask 구조에서는 각 라운드마다 독립적으로 설정한 sparsity pattern 만을 업데이트 한다. 그래서 0이 아닌 entry만 서버에 보내진다.
반면에 sketched update는 디바이스에서 업데이트 하기전에 압축된 형태로 encode하고 server에서는 aggregation 하기 전에 decode를 수행한다. 
sketched update의 한 예시는 디바이스에서 오직 업데이트할 matrix의 random subset 만을 통신하고, 서버에서 실제 평균에서 편향되지 않는 결과를 얻기 위해 subsampled update들을 평균하여 사용한다. 
또 다른 sketched update는 확률적 quantization 기법이다. 업데이트할 matrix는 벡터화 되고 각 scalar에 대해서 quantize를 수행한다. 
quantization으로 인한 오차를 줄이기 위해, Walsh-Hadamard 행렬과 이진 대각 행렬의 곱의 구조화 된 임의 회전을 양자화 전에 적용 할 수 있습니다.
시뮬레이션 결과에서는 low-rank보다 random mask가 더 좋은 정확도를 보였고, 뿐만 아니라 sketched 알고리즘 보다도 더 좋은 성능을 보였다. 
하지만 세 가지의 (subsampling, quantization, rotation) sketching tool 모두를 함께 사용하는 것이 모델의 빠른 수렴과 압축률을 보여줬다. 


### Importance-Based Updating


 ## 4. Resource Allocation
 


