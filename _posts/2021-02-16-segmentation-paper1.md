---
layout: post  
title: (Paper Review) Segmentation  
subtitle: Video Object Segmentation with Adaptive Feature Bank and Uncertain-Region Refinement  
gh-repo: xmlyqing00/AFB-URR
gh-badge: [star, fork, follow]  
tags: [ai, ml, segmentation]  
comments: true
--- 

이 논문에서는 Semi-supervised video object segmentation을 위한 새로운 Matching-based framework를 제안한다.  
기존의 matching-based 방식에 region matching과 classification을 위해서 feature bank를 사용했지만, 지속적으로 커지는 feature bank에 정보를 효과적으로 구성하는 방법을 찾진 못했다.  
이를 해결하기 위해 저자는 동적으로 새로운 feature을 흡수하고 구식의 feature는 무시하는 adaptive한 feature bank를 소개하고 있다.  
또한 불확실한 region에 대한 segmentation accuracy를 높이기 위해 새로운 confidence loss와 fine-grained segmentation module을 제안다.  

```
Proceeding: NIPS 2020
Authors: Yongqing Liang, Xin Li, Navid Jafari, Jim Chen
```
[Source Code Link](https://github.com/xmlyqing00/AFB-URR)

## Introduction
VOS(Video Object Segmentation)의 semi-supervised setting에서 성능에 가장 영향을 주는 두 스텝은  
(1) object의 region과 배경과 구별  
(2) segmentation의 boundary를 깔끔하게 구별  
하는 것이다.  

최근의 연구를 implicit learning과 explicit learing으로 카테고리화 할 수 있다.  
implicit 방식은 detection-based와 propagation-based learning을 포함한다. FCN 파이프라인을 사용하고 종종 비디오에서 새로운 물체에 대해 학습하기 위해 online fine-tune을 요구한다.  
explicit 방식은 물체의 appearance를 명시적으로 학습한다. 이 embedding 공간에서 pixel-wise classification로 공식화한다.  
object의 appearance를 기억하기 위해 embedding 공간을 설계하고 similarity 계산 이용해서 subsequent frame들을 segment 한다.  
그래서 이방식을 matching-based라고 칭하기도 한다. 최근에 이 방식은 VOS 벤치마크에서 SOTA를 달성하였다.  

matching-based VOS의 근본적인 이슈는 어떻게 이전 프레임의 정보를 이용하여 새로운 프레임을 명시적으로 segment 하느냐에 있다.  
메모리 크기는 제한적이고, 모든 프레임의 정보들을 기억하는 것은 불가능하다.  
대부분의 방식은 처음과 최근의 프레임을 이용하거나 프레임 uniformly sampling하여 사용한다.  
하지만 비디오가 점차 길어지면서 out-of-memory crash 문제를 일으키거나 key-frame을 샘플링 하지 못하는 문제를 발생시킨다.  

저자는 이를 해결하기 위해 Adaptive Feature Bank(AFB)를 도입하였다.  
weighted averageing과 least frequently used(LFU) index를 이용하여 구식의 feature를 무시하는 매커니즘을 사용한다.  
이 방법으로 다양한 object의 특징을 기억할 수 있고, 긴 비디오에서도 적은 메모리 만으로 동시에 segmentation을 수행할 수 있다.  

또한 uncertain-region refinement(URR)을 이용하여 segmentation의 퀄리티를 높였다. 
이는 segmentation의 모호함을 평가하는 새로운 confidence loss와 이 모호한 영역을 refine하기 위한 local fine-grained segmentation을 이용한다.  

## Approach
![Figure 1](../assets/_images/segmentation/paper1/1.png)

### Matching-based segmentation

첫번째 프레임과 gt annotation을 인코딩하여 feature bank를 설계하는데 사용한다.  
그다음 두번째 프레임부터 target object를 segmentation 하기 위해 feature bank를 이용한다.  
decoder는 matching 결과를 이용하여 segmentation을 수행한다. 

#### Encoders
* query Encoder: 현재의 frame을 인코딩하며, named query frame을 이용한다.  
    * ResNet-50의 layer3 feature map을 이용 (Size: H/8 X W/8 X 1024)
* Reference Encoder: Target object의 특징을 기억하기 위해 설계 
    * t-th frame을 segmentation 하기 위해서 past frame으로 1부터 t-1번째 까지를 사용
    * 각 프레임의 reference frame: L개의 물체를 구분하기 위해서 L 개의 feature map을 사용 (Size: L X H/8 X W/8 X 1024) 
    * Pj = {p1, p2, ..., pL}, j ∈ [1, t − 1].
    
#### Feature map embedding
기존의 matching-based 방식에서는 query feature map과 reference feature map간에 direct하게 비교를 수행했다.  
classification에서는 좋은 성능을 내지만 segmantic information에는 부족한 방식이다.  
이를 보완하기 위해 두 convolutional module을 이용하여 두 개의 key와 value embedding space로 인코딩을 수행한다.  
key를 이용하여 matching을 수행하고 value는 풍부한 sementic 정보를 보존하기 위해 다른 값을 갖을 수 있도록 허용한다.

#### Matcher
query frame과 feature bank 간의 similarity는 object by object 간에 수행된다.
![Figure2](../assets/_images/segmentation/paper1/2.png)  
![Figure2-1](../assets/_images/segmentation/paper1/4.png)  
i ∈ [1, L]  
query value와 가장 비슷한 value map과 concatenate를 수행 한다.
![Figure2-2](../assets/_images/segmentation/paper1/5.png)

#### Decoder
matching 결과인 y를 이용하여 각 object의 mask를 독립적으로 예측한다.  
refinement module을 이용하여 feature map을 점진적으로 upscale한다.  
각 stage에서 previous stage와 query encoder로부터 온 feature map간 skip connection을 이용한다.  
최종적으로 object i에 대한 마스크인 Mi를 구한다.   
![Figure3](../assets/_images/segmentation/paper1/3.png)

### Adaptive Feature Bank
AFB의 주요 특징은 새로운 feature를 흡수하고 구식은 제거하는 것이다.  

#### Absorbing new features
대부분 최근의 프레임이 더 중요한 정보를 가지고 있지만 초기의 프레임도 필요한 정보를 담고 있을 수 있다.  
그래서 초기 프레임을 단순히 무시하기 보다 이를 저장하고 weighted averageing을 이용하여 관리하는 방식을 취한다. 
새로운 특징이 추출되고, 이전 특징과 충분히 가깝다면 이들을 merge 하므로써 중복된 정보를 저장하는 것을 피할 수 있다.   
![Figure4](../assets/_images/segmentation/paper1/6.png)

new feature embedding: a(i) = (Kp, Vp)  
old feature embeddings: b(j) = (Kfb, Vfb)  
Similarity Function:
![Figure5](../assets/_images/segmentation/paper1/7.png)

a(i)에 대해서 가장 비슷한 b(j`)을 구한 후 merging을 수행  
![Figure6](../assets/_images/segmentation/paper1/8.png)  
비슷하지 않으면 feature bank에 추가한다.  

#### Removing obsolete features
cache 정책과 비슷하게 오래된 feature에 대해서 최근 사용된 빈도를 계산한다.(LFU)  
만약 각 matching에서 similarity가 10^-4보다 크다면 count를 증가시킨다  
![Figure7](../assets/_images/segmentation/paper1/9.png)  
feature bank의 크기가 정해진 크기를 초과하면 삭제를 진행한다. 

### Uncertain-region Refinement
#### Confidence loss
decoding과 softmax normalization을 수행한 initial segmentation Mi는 [0, 1]의 값을 가지고  
합은 1이다. 이는 해당 object에 대한 likelihood를 뜻한다.  
pixel-wise uncertainity map U를 구하기 위해서 마스크에서 가장 큰 값과 두번 째로 가장 큰 값의 비율을 이용한다.  
![Figure8](../assets/_images/segmentation/paper1/10.png)
0에서 1의 값을 갖게 하기 위해 L2 norm을 수행한다.  

#### Local Refinement mechanism
이웃한 픽셀들을 이용하여 refinement를 수행한다.  
![Figure9](../assets/_images/segmentation/paper1/11.png)
residual network module f는 local similarity를 예측하도록 학습된다. 
e: local refinement mask  
c: confidence score  
S: final segmentation   
![Figure9](../assets/_images/segmentation/paper1/12.png)
![Figure9](../assets/_images/segmentation/paper1/13.png)
![Figure9](../assets/_images/segmentation/paper1/14.png)
![Figure9](../assets/_images/segmentation/paper1/15.png)

