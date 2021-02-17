---
layout: post    
title: (Paper Review) Robustness  
subtitle: Maximum-Entropy Adversarial Data Augmentation for Improved Generalization and Robustness      
tags: [ai, ml, robustness]    
comments: true  
--- 
적대적 data augmentation은 보지 못했던 데이터의 shift와 corruption에 대해 딥러닝 네트워크를 더욱 강건하게 학습하도록 만든다.  
하지만 source 분포와는 크게 다른 "hard" adverserial perturbation을 포함하는 허구의 target 분포를 생성하는 것은 매우 어렵다.  
이 논문에서는 adversarial data augmentation을 위한 새롭고 효과적인 regularization term을 제안한다.  
직관적으로 regularization term은 기본 source 분포를 교란하여 현재 모델의 예측 불확실성을 확대하도록 장려함으로써 더욱 강건하게 모델을 학습하도록 도와준다.  
 
```
Proceeding: NIPS 2020
Authors: Long Zhao, Ting Liu, Xi Peng, Dimitris Metaxas
```
[Source Code Link](https://github.com/garyzhao/ME-ADA)  
[Paper Link](https://proceedings.neurips.cc/paper/2020/file/a5bfc9e07964f8dddeb95fc584cd965d-Paper.pdf)  

## Introduction
딥러닝 네트워크는 같은 분포에서 온 train과 test 데이터 셋에 대해서는 매우 좋은 퍼포먼스를 보인다.  
하지만 실제로는 train과 test 분포는 다른 경우가 대부분이다.  
최근에는 model robustness를 위해서 보지 못한 data shift를 resemble 하기 위해 adversarial loss를 이용하여 허구의 target distribution을 생성한다.  
하지만 이런 heuristic loss function은 큰 dataset shift를 합성하기에는 역부족이다.  
이 문제를 완화하기 위해 정보 병목 현상(IB) 원칙을 사용하여 정보 이론 관점에서 adversarial data augmentation을 위한 정규화 기법을 제안한다.  
IB 원칙은 input variable에서 예측에 도움이 되지 않는 부분을 줄이고  optimal한 representation을 학습하도록 도와준다.  
최근에는 이런 IB를 이용한 방식이 많이 나오고 있지만 그럼에도 adversarial data augmentation의 효율성은 아직 불명확하다.  

