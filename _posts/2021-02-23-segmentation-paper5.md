---
layout: post  
title: (Paper Review) Segmentation  
subtitle: Fast Video Object Segmentation with Temporal Aggregation Network and Dynamic Template Matching   
tags: [ai, ml, segmentation]  
comments: true
--- 

VOS task는 image instance segmentation과 video object tracking 문제로 나눌 수 있다.
이 논문에서는 새로운 temporal aggregation network와 dynamic time-evolving template matching mechanism을 통해 segmentation을 tracking으로 일관되게 통합하는 tracking-by-detection을 소개한다.
이는 online 방식으로 one-shot learning에 적합하며, 한번의 forward pass 만으로 다양한 object segmetation이 가능하게 한다. 

```
Proceeding: 2020  
Authors: Xuhua Huang1, Jiarui Xu1, Yu-Wing Tai, Chi-Keung Tang
```

[Paper Link](https://arxiv.org/pdf/2007.05687.pdf)  

## Introduction
현재 존재하는 SOTA 방법들은 매우 무겁고, semantic segmentation에 더 치중된 디자인을 가지고 있어서 tracking solution에 큰 이점을 주지 못한다. 
VOS task는 자연스럽게 semantic segmentation과 object tracking 문제로 분할 된다. 
Semantic Segmentation에서 대부분은 Fully Convoutional Networks를 기본으로 한다. 
하지만 semantic segmentation map은 
