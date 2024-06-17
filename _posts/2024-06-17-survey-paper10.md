---
layout: post   
title: A Survey of Resource-Efficient LLM and Multimodal Foundation Models    
subtitle: AI Survey     
tags: [ai, ml, Large Language Model, Vision Language Model, LLM, LVLM, VLM]    
comments: true  
---

이 survey는 LLM, ViTs, Diffusion Model 및 LLM 기반다중 모델과 같은 대형 기초 모델들이 기계학습 전 과정에서 혁신을 이루고 있음을 다룹니다.
이러한 모델이 제공하는 다재다능성과 성능의 발전은 하드우ㅐ어 자원 측면에서 상당한 비용을 초래합니다.
이러한 대형 모델의 성장과 환경적으로 지속 가능한 방안을 지원하기 위해, 자원 효율적인 전략 개발에 상당한 초점이 맞춰졌습니다. 
이 연구는 이러한 연구의 중요성을 강조하며, 알고리즘적 및 시스템적 측면을 모두 검토합니다.
모델의 아키텍처, 학습, 서빙 알고리즘, 실용적 시스템 설계 및 구현에 이르는 다양한 주제를 포괄하여 귀중한 통찰력을 제공합니다.

[Paper Link](https://arxiv.org/pdf/2401.08092)  
[Code Link](https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey)  

## Introduction

GPT 같은 대형 LLM, ViTs, Latent Diffusion Models(LDMs), CLIP 등의 모델들은 더 많은 데이터와 매개변수를 바탕으로 성능을 확장할 수 있지만, 이는 막대한 자원 소모를 동반합니다.
이런 대형 기초 모델의 자원 요구로 인해 소수의 주요 업체만이 이를 훈련하고 배포할 수 있으며, 이는 데이터 프라이버스 문제를 야기합니다. 
그래서 이를 해결하기 위해, 알고리즘 최적화, 시스템 혁신 등을 통해 기초 모델의 자원 효율성을 높이기 위한 다양한 연구가 진행중입니다. 이러한 연구는 자원 소모를 줄이면서 성능으 ㄹ유지하는데 중점을 둡니다.



## FOUNDATION MODEL OVERVIEW

### Model Architectures
- Transformer Pipeline
	- Embedding: 입력 단어를 임베딩 레이어를 통해 고차원 벡터로 변환
	- Attention 매커니즘: 입력 벡터의 다양한 부분에 가중치를 할당하여 중요한 정보를 강조
	- Layer Norm: 활성화를 안정화하고, 표준화하여 모델의 안정성 확보
	- Feed Forward Network: 각 위치별 벡터를 비 선형적으로 변환하여 복잡한 데이터 패턴을 포착
	- Multi layer train: 여러 레이어를 통한 입력 데이터의 계층적 표현 학습
	- Final Prediction: 마지막 출력을 선형 레이어로 전달하여 최종 예측 도출

- Embedding	
	- 입력 단어를 시퀀스 토큰으로 변환합니다. 주로 사용되는 토크나이저로는 WordPiece, BPE 가 있습니다.
	- Embedding Layer: 시퀀스 토큰을 벡터 시퀀스로 변환합니다. 단어 순서는 의미에 중요하므로, 위치 인코딩을 임베딩에 추가하여 순서 정보를 포함시킵니다.

- Attention
	- 시퀀스 내 단어들간의 관계를 포착하는 데 중요한 역할을 함
	- Self-Attention: 쿼리, 키, 값 쌍이 모두 동일한 입력 시퀀스에서 유도됩. 입력의 각 위치에서 다른 부분에 집중할 수 있게 함.
	- Multi-head Attention: Self-Attention의 변형으로, 다양한 표현 하위 공간에서 정보를 동시에 주목할 수 있게함. 
	- Sparse Attention: 효율성을 위해 설계
	- Multi-Query Attention: 다양한 다운스트림 작업을 위해 설계
		- 여러 쿼리에 대해 동일한 키-값 쌍을 사용하여 메모리 사용량과 계산 복잡도를 줄일 수 있습니다. 
		- 구현 예시 

```python
class MultiQueryAttention:
	def __init__(self, d_model, n_heads):
		super(MultiQueryAttention, self).__init__()
		self.d_model = d_model
		self.n_heads = n_heads
		self.d_heads = d_model // n_heads
		
		assert d_model % n_heads == 0
	
		self.query_linear = nn.Linear(d_model, d_model)
		self.key_linear = nn.Linear(d_model, d_heads)
		self.value_linear = nn.Linear(d_model, d_heads)

		self.out_linear = nn.Linear(d_heads * n_heads, d_model)

	def forward(self, x):
		
		batch_size = x.size(0)
		Q = self.query_linear(x)  # [Batch, Seq_len, d_model]
		K = self.key_linear(x)		# [Batch, seq_len, d_heads]
		V = self.value_linear(x)	# [Batch, seq_len, d_heads]
	
		Q = Q.view(batch_size, -1, self.n_heads, self.d_heads).transpose(0, 2, 1, 3) # [batch, n_heads, seq_len, d_heads]
		K = K.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
		V = V.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

		scores = torch.bmm(Q, K.transpose(-2, -1)) / torch.sqrt(self.d_heads)
		attn = nn.Softmax(scores, dim=-1)
		context = torch.bmm(attn, V)
		return context
```

- Encoder-Decoder Architecture
	- Encoder
		- 셀프어텐션: 입력 시퀀스를 셀프 어텐션 매커니즘을 통해 처리하여 입력 시퀀스의 각 부분에 가중치를 할당하여 정보를 강조
		- 입력 데이터 내 복잡한 패턴과 의존성을 이해하는 데 필수적 
	- Decoder: 출력 시퀀스를 생성합니다.
		- 셀프 어텐션: 이미 생성된 출력 내 관계를 이해
		- 크로스 어텐션: 인코더 출력과, 디코더 출력간의 관련 정보를 추출
		- Auto Regressive: 출력 토큰을 순차적으로 생성하며, 각 토큰의 생성은 이전 생성된 토큰에 의존


- Auto-Regressive Decoding and KV Cache
	
		
