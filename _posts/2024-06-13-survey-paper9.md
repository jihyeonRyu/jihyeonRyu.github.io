---
layout: post   
title: An Introduction to Vision-Language Modeling     
subtitle: AI Survey     
tags: [ai, ml, Large Language Model, Vision Language Model, LLM, LVLM, VLM]    
comments: true  
---

최근 LLM의 인기로 인해 이를 비전 영역으로 확장하려는 시도가 여러차례 있었습니다. 비전-언어 모델 (VLM)은 시각 정보를 텍스트와 결합하여 다양한 프로그램에서 사용될 수 있습니다.
이 논문은 VLM의 기본 개념, 훈련 방법, 평가 방법을 소개하고, 특히 이미지에서 언어로의 매핑을 중십으로 설명합니다.

[Paper Link](https://arxiv.org/pdf/2405.17247)

## Summary

### VLM의 유형

주로 네가지 훈련 패러다임으로 나눌 수 있습니다.
- Contrastive Training: Pos, Neg 예제 쌍을 사용하여 이미지-텍스트 간의 유사한 표현을 학습합니다. (ex: CLIP)
- Masking: 이미지 패치나 텍스트를 마스킹하고 이를 재구성하는 방식입니다. (ex: FLAVA, MaskVLM)
- Generative Models: 텍스트와 이미지를 생성할 수 있는 모델입니다. (ex: CoCa, CM3leon)
- Pretrained Backbones: 사전 학습된 언어모델과 이미지 인코더를 사용하여 학습을 효율화 합니다. (ex: MiniGPT)

### VLM 훈련 가이드
- 훈련 데이터: 적절한 데이터셋 선택, 합성 데이터 사용, 데이터 증강, 인간 전문가의 데이터 주석 활용 등이 포함됩니다.
- 소프트웨어: 공개 소프트웨어 저장소 활용, GPU 수, 훈련 속도 향상, 하이퍼파라미터의 중요성 등이 다룹니다.
- 모델 선택: 대조 모델, 마스킹 모델, 생성 모델, 사전 학습된 백본 모델의 사용 시기를 논의합니다. 

### 책임 있는 VLM 평가 접근법
- 평가 벤치마크: 이미지 캡셔닝, 텍스트-이미지 일관성, 시각적 질문 응답, 제로샷 이미지 분류 등 다양한 벤치마크를 통해 VLM의 능력을 평가합니다.
- 편향 및 불균형 평가: 분류 및 임베딩에서의 편향 측정, 언어 편향 평가 등이 포함됩니다.
- 환각 및 기억 측정: VLM의 환각 현상과 모델의 기억 능력을 평가합니다.

### 비디오로의 확장
VLM을 비디오 데이터로 확장하는 방법을 논의합니다. 비디오는 이미지보다 계산 비용이 더 높고 시간적 차원을 텍스트로 매핑하는 도전 과제가 있습니다.


## The Families of VLMs

![](./../assets/resource/survey/paper9/1.png)

### 1. Early work on VLMs based on transformers
비전-언어 모델의 초기 연구는 트랜스포머 아키텍처를 사용하여 언어 모델링을 시각 데이터로 확장하는 것을 목표로 했습니다. 
특히 BERT의 성공은 연구자들이 이를 시각 데이터 처리에 적용하도록 유도하였습니다.

- 주요 모델
  - VisualBERT
- 읻력 형식
  - 텍스트 입력: BERT 모델과 동일하게 토큰화되어 입력되어 임베딩 벡터로 변환합니다. 
    - [CLS] A cat sitting on a chair [SEP]
  - 이미지 입력
    - Faster R-CNN 같은 Object Detection 모델을 이용해서 이미지에서 객체 영역 (RoI)을 탐지하고, RoI Pooling을 통해 고정된 크기의 특징 맵을 생성합니다.
    - Object-based Positional Embedding: 각 이미지의 BBOX(cx, cy, w, h) 를 바탕으로 각 요소에 대해 positional embdding을 생성하고 이를 concat 하여 사용합니다.
    - 이 2차원 특징 벡터를 flattening 하여 1차원 임베딩 벡터로 변환하고 linear layer를 거쳐 positional embdding과 결합합니다. 
    - [CLS] [Image Feature 1] [Image Feature 2] ... [Image Feature N] 
- 훈련 목표
  - Masked Language Modeling: 문장의 일부 단어를 마스킹하고, 모델이 마스킹된 단어를 예측
  - Sentence-Image Prediction: 이미지와 문장이 주어졌을때, 해당 문장이 실제로 주어진 이미지를 설명하는지 예측하는 목표
    - Sigmoid 함수를 사용하여 이진 분류 문제로 처리

- 트랜스포머의 역할: 입력 데이터를 토큰화하여 학습하고, attention 메커니즘을 통해 단어와 이미지 간의 관계를 학습 

### 2. Contrastive-based VLMs
이미지와 텍스트 간의 관계를 학습하기 위해 대조 학습을 사용하는 모델들입니다. 이미지와 텍스트 쌍을 이용하여 관련있는 쌍은 유사한 임베딩 공간에 위치 시키고, 관련 없는 쌍은 멀리 떨어뜨리는 방식으로 학습합니다.
- 주요 모델
  - CLIP, SigLIP, Lip
- Contrastive Loss
  - InfoNCE (Noise Contrastive Estimation)
    - ![](./../assets/resource/survey/paper9/0.png)
    - 모델 분포에서 샘플링 하는 대신 노이즈 분포에서 샘플링하여 모델 분포를 근사
    - 큰 배치를 사용해야 성능이 좋습니다.
    - 구현은 아래와 같습니다. 

  ```python
  logits_per_image = image_features @ text_features.T / self.temperature # cosine similarity
  logits_per_text = text_features @ image_features.T / self.temperature 
  
  labels = torch.arange(batch_size)
  
  loss_image = F.cross_entropy(logits_per_image, labels)
  loss_text = F.cross_entropy(logits_per_text, labels)
  ```
  - Binary Cross-Entropy Loss
    - 이진 분류 문제로 접근할 수 있다. 긍정 예제는 1, 부정 예제는 0
      
- CLIP (Contrastive Language-Image Pretraining)
  - 이미지와 텍스트를 각각 처리하는 두 개의 독립된 인코더를 사용
    - 이미지 인코더는 ResNet 과 같은 CNN 모델과 마지막 Flatten - Linear layer 로 구성될 수 있다. 
    - 텍스트 인코더는 BERT 가 사용 
  - InfoNCE를 사용하여 동일한 임베딩에 위치하도록 학습
  - 응용
    - Zero-Shot Image Classification
    - Image retrieval: 텍스트를 기반으로 이미지 검색
    - Text Generation: 이미지를 기반으로 텍스트 설명 생성
- SigLIP
  - CLIP과 유사하지만, BCE를 사용하여 작은 배치에서도 좋은 성능을 보입니다. 

- Llip (Latent Language Image Pretraining)
  - 이미지가 다양한 방식으로 캡션 될 수 있다는 점을 고려한 모델입니다.
  - Cross-Attention 모듈을 사용하여 캡션의 다양성을 고려하여 인코딩을 수행합니다. 


### 3. VLMs with masking objectives
마스킹 전략을 사용하여 이미지와 텍스트 간의 관계를 학습하는 모델들입니다. 이러한 모델들은 주로 입력 데이터의 일부를 마스킹하고, 마스킹된 부분을 예측하는 방식으로 학습됩니다.
이는 모델이 컨텍스트를 이해하고, 누락된 정보를 복원하는 능력을 향상시키는 데 도움을 줍니다. 

- 주요 모델 
  - FLAVA(Foundational Language And Vision Alignment), MaskVLM
  
- FLAVA
  - 아키텍처
    - 이미지 인코더: ViT를 기반으로 함
    - 텍스트 인코더: BERT를 기반으로 함 
    - Multimodal Encoder: 이미지와 텍스트의 히든 상태 벡터를 결합하고, 크로스 어텐션 메커니즘을 사용하여 상호 정보를 통합
  - 학습 목표
    - Masked Language Model(MLM): 텍스트 일부를 마스킹하고, 이르 예측
    - Masked Image Model(MIM): 이미지 패치의 일부를 마스킹하고 이를 예측 
    - Contrastive Learning: 텍스트와 이미지 임베딩 간의 유사도를 최대화 하는 대조학습

- MaskedVLM
  - FLAVA와 유사하지만, 이 모델은 텍스트와 이미지 간의 정보 흐름을 강조합니다
  - 학습 목표
    - MLM
    - MIM
    - Cross-Modality Prediction: 텍스트 정보를 사용하여 마스킹된 이미지 패치를 예측하거나, 이미지 정보를 사용하여 마스킹된 텍스트 토큰을 예측

----

#### 참조
- Vision Transformer (ViT)
  - Transformer 아키텍처를 사용하여 이미지 인식 작업을 수행하는 모델
  - 기존의 CNN과는 다른 접근 방식을 통해 이미지 데이터를 처리
  - 기본 개념 및 아키텍처
    - Image Patch Division
      - 입력 이미지를 작은 패치로 분할 
        - 224x224 -> 16x16
      - 각 패치는 독립적으로 처리, 텍스트에서 사용되는 토큰과 유사하게 다루어짐
    - Patch Embedding
      - 각 패치를 1차원 임베딩 벡터로 변환
  ```python
  class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channel, emb_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.projection = nn.Conv(in_channel, emb_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
      x = self.projection(x) # [B, emb_dim, H/patch_size, W/patch_size]
      x = x.flatten(2) # [B, emb_dim, num_patches]
      x = x.transpose(0, 2, 1) # [B, num_patches, emb_dim]
      
      return x
  ```
    - Positional Embedding
      - 각 패치의 정보를 보전하기 위해 위치 정보를 추가
      - 트랜스포머와 유사하게 패치의 순서를 인코딩하여 모델에 제공
    - Transformer Encoder
      - 텍스트 데이터를 처리할 때와용동일한 Transformer를 사용
      - Multi-Head Self-Attention과 FNN을 으로 구성
      - 입력 패치 임베딩 시퀀스를 인코더에 입력하여, 이미지의 전역적인 Contextual 한 정보를 학습 
    -  Classification Head
      - 이미지 분류 작업을 수행 
      - CE Loss 사용
----

- Information Theoretic View on VLM Objectives
  - 주요 목표는 정보 압축과 데이터 복원을 균형있게 달성하는 것입니다.
  - 모델의 불필요한 정보를 최소화하고, 중요한 정보를 최대화하여, 중요한 예측 정보를 최대화 하는데 도움을 줍니다
  - 정보 이론의 기본 개념
    - Information Entropy
      - 불확실성 또는 정보량의 척도
      - 엔트로피가 높을 수록 불확실성이 크고, 엔트로피가 낮을 수록 예측 가능성이 큽니다. 
      > H(x) = - p(x)log{p(x)}
    - Mutual Information
      - 두 변수 간의 의존성을 측정
      - 하나의 확률변수가 다른 하나의 확률변수에 대해 제공하는 정보의 양
      - 한 변수를 알면 다른 변수에 대한 정보를 얼마나 얻을 수 있는 지를 나타냅니다. 
        > I(x;y) = H(x) + H(Y) - H(x, y)  ---- (교집합 구하는 식과 비슷)
    - Rate-Distortion Theory
      - 데이터의 정보량을 줄이면서도, 원본 데이터와의 유사성을 유지하는 것을 목표로 합니다.
      - 이는 정보 압축과 데이터 복원의 균형을 맞추는 과정입니다.
      
  - VLM Objective와의 연관성
    - Masking기반 학습: Auto-Encoding 방식 처럼 마스킹된 입력데이터를 기반으로 원래 데이터로 복원하면서 중요한 부분을 유지하고, 불필요한 정보를 제거하는 효과 
    - Contrastive 학습: 유사도와 차이점을 학습하여 정보 압축을 최적화 

### 4 Generative-based VLMs
텍스트와 이미지를 생성하는 능력을 학습하는 모델들입니다. 이러한 모델들은 이미지나 텍스트를 입력 받아, 이를 바탕으로 새로운 텍스트를 생성하거나, 반대로 텍스트를 입력받아 이미지를 생성할 수 있습니다.

- 주요 모델
  - CoCa(Contrastive Captioner), Chameleon, CM3leon, Stable Diffusion, Imagen, Parti
- CoCa
  - 이미지 캡셔닝 
  - 주요 특징
    - CLIP에서 사용된 대조 손실 외에 생성 손실을 추가로 사용
    - 생성 손실: 멀티모달 텍스트 디코더가 이미지 인코더 출력과 단일 모달 텍스트 디코더의 표현을 입력으로 받아 생성한 캡션에 대한 손실
- CM3Leon
  - 텍스트-이미지 및 이미지-텍스트 생성.
  - 구성 요소:
    - 이미지 토크나이저: 256x256 이미지를 8192개의 어휘로 1024개의 토큰으로 인코딩.
    - 텍스트 토크나이저: 56320개의 어휘 크기
    - `<break>` 토큰: 모달리티 간 전환을 나타내는 특별한 토큰
    - 디코더 전용 트랜스포머 모델: 토크나이즈된 이미지와 텍스트를 입력으로 받아 다음 토큰을 예측
- Chemeleon
  - 텍스트와 이미지가 혼합된 시퀀스를 생성하고 추론.
  - 특징:
    - 모달리티 통합: 이미지와 텍스트를 동일한 토큰 기반 표현으로 변환하여 통합된 트랜스포머 아키텍처로 처리.
    - 초기 통합 전략: 모든 모달리티를 공유된 표현 공간으로 매핑하여 다양한 모달리티 간의 원활한 추론 및 생성 가능.
  - 기술적 도전과제: 최적화 안정성과 확장을 위한 아키텍처 혁신 및 훈련 기술 사용.
- Text-Image 생성 모델을 이용한 Downstream 비전-언어 작업
  - 생성 모델: Stable Diffusion, Imagen, Parti
  - 초점: 텍스트 조건부 이미지 생성

---
#### 참조 
- Stable Diffusion
  - 텍스트에서 이미지를 생성하는 모델
  - 모델 구조
    - Text Encoder: BERT, CLIP 같은 사전 훈련된 언어 모델을 사용
    - Image Denoiser: 
      - 노이즈가 있는 이미지를 입력받아 이미지를 임베딩으로 인코딩하고 time step 임베딩과 텍스트 임베딩과 결합하여 이를 점진적으로 복원하는 역할을 합니다.
      - UNet 구조를 기반으로 하여, 각 단계에서 이미지의 노이즈를 줄이는 작업을 수행합니다. 
    - Time Encoder
      - 확산 과정의 각 시간 단계를 인코딩하여 모델이 현재 단계의 정보를 인식할 수 있도록 합니다.
      - 각 시간 단계에 해당하는 임베딩 벡터를 생성합니다.
    - Cross-Attention
      - 텍스트 임베딩과 이미지 임베딩을 결합하여 정보를 통합합니다.
  - 학습 방법
    - Forward Process
      - 원본 이미지에 점진적으로 노이즈를 추가하여 노이즈 이미지를 생성합니다.
      - 각 시간 단계에서 노이즈가 추가됩니다.
    - Reverse Process
      - 노이즈가 추가된 이미지를 입력받아, 이를 점진적으로 복원하여 원본 이미지를 재구성 합니다. 
      - Image Denoiser (U-Net)가 사용되며, 각 단계에서 노이즈를 줄이는 작업을 수행합니다.
    - 손실 함수
      - 모델이 예측한 노이즈와 실제 노이즈 간의 차이를 최소화 하는 손실함수를 사용하여 학습 
      - 일반적으로 MSE Loss를 사용합니다.
    - 간단한 코드 

  ```python
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from transformers import CLIPTextModel, CLIPTokenizer
  from torchvision.models import resnet50
  
  class TimeEncoder(nn.Module):
      def __init__(self, embed_dim):
          super(TimeEncoder, self).__init__()
          self.time_embed = nn.Embedding(1000, embed_dim)  # 최대 1000 timesteps
  
      def forward(self, t):
          return self.time_embed(t)
  
  class CrossAttentionBlock(nn.Module):
      def __init__(self, embed_dim, num_heads):
          super(CrossAttentionBlock, self).__init__()
          self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
          self.norm = nn.LayerNorm(embed_dim)
          self.ffn = nn.Sequential(
              nn.Linear(embed_dim, embed_dim * 4),
              nn.GELU(),
              nn.Linear(embed_dim * 4, embed_dim),
          )
  
      def forward(self, x, context):
          attn_output, _ = self.cross_attention(x, context, context)
          x = self.norm(x + attn_output)
          ffn_output = self.ffn(x)
          x = self.norm(x + ffn_output)
          return x
  
  class UNet(nn.Module):
      def __init__(self, in_channels, out_channels, embed_dim, num_heads):
          super(UNet, self).__init__()
          self.encoder = resnet50(pretrained=True)
          self.fc = nn.Linear(self.encoder.fc.in_features, embed_dim)
          self.cross_attention = CrossAttentionBlock(embed_dim, num_heads)
          self.decoder = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=3, stride=1, padding=1)
  
      def forward(self, x, t_embed, text_embed):
          # 이미지 인코딩
          x = self.encoder(x)
          x = torch.flatten(x, 1)
          x = self.fc(x)  # [B, embed_dim]
  
          # 이미지 임베딩과 시간 임베딩 결합
          x = x + t_embed
  
          # 크로스 어텐션 적용 - 이미지와 텍스트간 
          x = x.unsqueeze(0)  # [1, B, embed_dim]
          text_embed = text_embed.unsqueeze(0)  # [1, B, embed_dim]
          x = self.cross_attention(x, text_embed)
          x = x.squeeze(0)  # [B, embed_dim]
  
          # 디코딩
          x = x.view(x.size(0), -1, 1, 1)
          x = self.decoder(x)
          return x
  
  class StableDiffusion(nn.Module):
      def __init__(self, text_dim, image_dim, embed_dim, num_heads):
          super(StableDiffusion, self).__init__()
          self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
          self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
          self.time_encoder = TimeEncoder(embed_dim)
          self.unet = UNet(image_dim, image_dim, embed_dim, num_heads)
  
      def forward(self, images, texts, timesteps):
          # 텍스트 임베딩
          text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
          text_outputs = self.text_encoder(**text_inputs)
          text_embeds = text_outputs.last_hidden_state[:, 0, :]  # [B, D]
  
          # 시간 임베딩
          t_embeds = self.time_encoder(timesteps)  # [B, D]
  
          # 이미지 디노이징
          denoised_images = self.unet(images, t_embeds, text_embeds)
  
          return denoised_images
  
  # 예시 데이터
  batch_size = 8
  image_dim = 3  # RGB 이미지
  embed_dim = 512
  num_heads = 8
  
  images = torch.randn(batch_size, image_dim, 224, 224)  # 임의의 노이즈 이미지 배치
  texts = ["a photo of a cat", "a picture of a dog", "a snapshot of a bird"] * (batch_size // 3)
  timesteps = torch.randint(0, 1000, (batch_size,))  # 임의의 timesteps
  # Stable Diffusion 모델 초기화
  stable_diffusion = StableDiffusion(text_dim=512, image_dim=image_dim, embed_dim=embed_dim, num_heads=num_heads)
  
  # 손실 함수와 옵티마이저 초기화
  criterion = nn.MSELoss()
  optimizer = optim.Adam(stable_diffusion.parameters(), lr=1e-4)
  
  # 학습 과정
  num_epochs = 10
  for epoch in range(num_epochs):
      optimizer.zero_grad()
      
      # Forward pass
      denoised_images = stable_diffusion(images, texts, timesteps)
      
      # 각 타임스텝에 맞는 손실 계산
      loss = 0
      for t in range(1000):
          mask = (timesteps == t)
          if mask.sum() == 0:
              continue
          current_images = gt_images[mask] # 각 타임 스템프 별로 디노이즈 된 이미지 
          current_denoised_images = denoised_images[mask]
          current_loss = criterion(current_denoised_images, current_images)
          loss += current_loss
      
      # 역전파 및 최적화
      loss.backward()
      optimizer.step()
      
      print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
  
  print("Training completed.")

  ```
---
  
### 5. VLMs from Pretrained Backbones
Pretrained Backbone을 활용하여 텍스트와 이미지를 함께 처리하는 멀티모달 모델을 구축하는 접근 방식 입니다.
사전 훈련된 모델의 강력한 표현력을 활용하여 멀티모달 작업에서 높은 성능을 발휘할 수 있습니다.

- Pretrained Backbones의 장점
  - 학습 시간 단축
  - 성능 향상
  - 전이 학습
- 주요 모델 및 방법론 
  - Frozen: 사전 학습된 대형 언어 모델(LLM)을 활용하는 첫 번째 예시입니다. 이 모델은 비전 인코더와 고정된(가중치가 업데이트되지 않는) 언어 모델을 연결하여 작동합니다.
  - MiniGPT-4: 사전 학습된 비전 인코더와 언어 모델을 사용하여 선형 프로젝션 레이어만 학습하여 텍스트를 생성 
  - MiniGPT-5: 텍스트와 이미지를 생성할 수 있도록 학습되며, 생성적 토큰을 사용하여 Stable Diffusion 모델을 통해 이미지를 생성
  - MiniGPT-v2: 다양한 비전-언어 작업을 통합된 인터페이스를 통해 수행할 수 있도록 학습되며, 각 작업에 고유 식별자를 도입하여, 모델이 작업을 구별할 수 있게 합니다. 
  
- BlIP-2 (Bootstrapped Language-Image Pretraining)
  - 사전 학습된 모델을 활용하여 학습 시간을 크게 줄이면서도 높은 성능을 발휘 할 수 있도록 설계되었습니다.
  - 주요 구성 요소
    - 비전 인코더: 사전 학습된 CLIP 모델 사용, 가중치 고정
    - 텍스트 인코더: 사전 학습된 OPT 모델을 사용, 가중치 고정 
    - Q-Former: 상대적으로 작은 트랜스포머 모델 (100-200M 파라미터)로 구성됩니다.
      - 이미지 임베딩을 LLM 입력공간으로 매핑하기 위해 학습됩니다. 
      - 비교적 작은 파라미터 수로 학습이 빠르게 진행됩니다. 
      - 쿼리 (텍스트) 벡터는 이미지 임베딩과 크로스 어텐션을 통해 상호작용하고, 선형 레이어를 통해 맵핑 됩니다. 

## A Guide to VLM Training

![](./../assets/resource/survey/paper9/3.png)

이 문서는 딥 뉴럴 네트워크 성능을 향상시키기 위한 스케일링의 중요성을 논의합니다. 최근 연구들은 더 나은 모델을 학습하기 위해 계산 자원과 규모를 증가시키는 것에 중점을 두고 있습니다. 예를 들어, CLIP 모델은 4억 개의 이미지를 사용하여 고도로 높은 계산 예산으로 학습되었습니다. 그러나 최신 연구는 데이터 큐레이션 파이프라인을 사용하여 스케일링 법칙을 극복할 수 있음을 보여줍니다.

문서는 다음과 같은 주제를 다룹니다:
1. **데이터의 중요성**: VLM(Visual Language Models) 학습을 위한 데이터셋 생성 방법.
2. **효율적인 학습을 위한 소프트웨어와 도구**: VLM을 더 효율적으로 학습시키는 데 사용되는 일반적인 소프트웨어, 도구 및 요령.
3. **모델 선택 방법**: 특정 상황에서 어떤 유형의 모델을 선택할지에 대한 논의.
4. **기반 강화 기술**: 텍스트와 시각적 단서를 정확히 매핑하는 능력을 향상시키는 방법.
5. **인간 선호도와의 정렬 기술**: 정렬을 개선하는 기술 소개.
6. **OCR 기능 향상**: VLM의 OCR(Optical Character Recognition) 능력을 향상시키는 기술.
7. **파인튜닝 방법**: 일반적인 파인튜닝 방법에 대한 논의.

이 섹션은 데이터의 중요성을 강조하며, 다양한 방법과 도구를 사용하여 VLM을 더 효율적으로 학습시키는 방법을 설명합니다.

### 1. Training Data

학습 데이터의 품질과 다양성은 VLM 성능에 매우 중요합니다. 적절하게 큐레이션된 데이터는 모델이 다양한 작업을 처리할 수 있도록 도와줍니다.

####  Improving the training data with synthetic data
- 인공적 이미지와 해당 캡션을 생성하여 학습 데이터를 보강할 수 있습니다.
- 장점: 이 방법은 희귀한 시나리오에 대한 데이터를 생성하거나 실제 데이터가 부족한 경우 데이터 세트를 확대하는 데 유용합니다.
- 캡션 생성 및 필터링을 통한 학습 데이터 개선 
  - **BLIP**
    - 데이터 품질을 개선하기 위해 합성 샘플을 생성하고 노이즈 많은 캡션은 필터링 하는 과정을 거쳐, 더 일관된 캡션을 생성하도록 하였습니다.
    - 방법
      - 합성 캡션 생성: 사전 학습된 모델을 사용하여 이미지에 대한 합성 캡션을 생성합니다. 각 이미지에 대해서 다양한 캡션을 생성하여 여러 후보 캡션을 확보합니다.
      - 노이즈 캡션 필터링: 각 캡션의 설명성(Descriptiveness)를 평가하여 얼마나 이미지의 내용을 잘 설명하는 지 측정합니다. 그리고, 캡션의 일관성 검사를 위해 이미지와 캡션 사이 유사도를 측정하는 모델을 사용하거나, 캡션이 이미지의 주요 특징을 정확하게 설명하는 지 평가합니다. 
      - 캡션 선택: 가장 높은 점수를 받은 캡션을 선택하여 노이즈가 많은 캡션을 걸러냅니다. 선택한 캡션만을 학습 데이터로 사용합니다.
  - Nguyen et al. [2023]
    - 기존 데이터셋에서 이미지와 제대로 일치하지 않는 Alt-text 레이블을 식별합니다. 
    - BLIP2를 사용하여 Alt-text 레이블을 대체하는 합성 캡션을 생성합니다. 
    - 실제 캡션과 BLIP2를 통해 생성된 합성 캡션을 혼합하여 새로운 학습 데이터 셋을 만듭니다. 
- 텍스트-이미지 생성 모델을 통한 학습 데이터 생성 

#### Using Data Augmentation
- 자르기, 회전, 뒤집기, 색상 조정 등 다양한 data augmentation 기법을 학습 데이터에 적용 
- 목적: 이러한 기법은 데이터 세트의 변동성을 증가시켜 모델이 다양한 시각적 입력에 대해 더 강력해 지도록 합니다. 
- Self-Supervised Learning (SSL)과 데이터 증강을 함께 사용합니다.
- **SLIP** [Mu et al., 2022]
  - SLIP는 vision 인코더에 보조적인 자기 지도 학습(SSL) 손실 항목을 도입합니다. 이는 SimCLR과 유사하게 입력 이미지를 두 개의 증강된 이미지로 생성하여 긍정 쌍(positive pair)을 만들고, 이를 배치 내의 다른 모든 이미지와 대조하는 방식입니다.
  - 장점: 이 추가 손실 항목은 작은 오버헤드를 가지면서도 학습된 표현을 개선하는 정규화 항목을 제공합니다.
  - 한계: 그러나, 이 SSL 손실이 텍스트로부터 오는 중요한 신호를 충분히 활용하지 못한다는 점에서 한계가 있습니다.
- **CLIP-rocket** [Fini et al., 2023]
  - 크로스 모달 SSL 손실: CLIP-rocket은 SSL 손실을 크로스 모달로 변환하는 것을 제안합니다. 즉, CLIP 대조 손실을 이미지-텍스트 쌍의 여러 증강을 활용하는 방식으로 사용합니다.
  - 이미지-텍스트 쌍을 비대칭적으로 증강합니다. (하나는 Strong Augmentation 세트, 다른 하나는 Weak Augmentation 세트)
  - 프로젝터: Weak 쌍의 프로젝터는 원래의 CLIP의 linear layer를 유지하고, Strong 쌍의 프로젝터는 2개 layer의 MLP를 사용하여 더 노이즈가 많은 임베딩을 처리하도록 합니다. 서로 다른 프로젝터를 이용하여 멀티모달 임베딩 공간으로 투영합니다. 
  - 추론: 추론 시에는 Weak 하게 학습된 표현과 Strong 하게 학습된 표현을 결합하여 최종 벡터를 만듭니다. 

#### Interleaved Data Curation
이 방법은 텍스트와 이미지 데이터를 교차하여 포함시키는 것이며, 이를 통해 모델의 few-shot 성능을 향상시킬 수 있습니다.
텍스트와 이미지를 교차한다는 것은 텍스트 데이터와 이미지 데이터를 함께 사용하여 모델을 학습시키는 것을 의미합니다. 이 과정에서 텍스트와 이미지가 자연스럽게 연결된 데이터를 사용하거나, 텍스트와 이미지를 인위적으로 쌍을 이루는 방식으로 데이터를 구성할 수 있습니다. 

- **Natural Interleaved Data**
  - 정의: 웹 문서나 다른 출처에서 자연스럽게 텍스트와 이미지가 함께 등장하는 데이터를 수집하는 것입니다.
  - 예시: 뉴스 기사, 블로그 포스트, 소셜 미디어 게시물 등에서 텍스트와 함께 포함된 이미지를 사용하는 것입니다.
  - OBELICS Dataset
  - 큐레이션
    - 데이터 수집: Common Crawl에서 영어 데이터를 수집하고 중복 제거를 수행합니다.
    - HTML 문서 전처리: 유용한 DOM 노드를 식별하고 유지하며, 로고를 제거하는 이미지 필터링을 적용합니다. 이후 각 DOM 노드에 대해 텍스트와 이미지를 함께 보존합니다.
    - 문서 수준의 텍스트 필터링: 다양한 휴리스틱을 사용하여 잘못된 형식이나 일관성이 없는 텍스트를 처리합니다.
- **Synthetic Interleaved Data**
  - 정의: 기존에 텍스트만 있는 데이터에 이미지를 추가하여 인위적으로 텍스트-이미지 쌍을 만드는 것입니다.
  - 예시: 텍스트 데이터셋에 인터넷에서 수집한 이미지에 대해 CLIP 모델 등을 사용하여 적절한 이미지를 찾아서 추가하는 방식입니다.
  - MMC4 Dataset
  - 큐레이션
    - 텍스트 데이터 선택: 대규모의 기존 텍스트 데이터셋을 선택합니다.
    - 이미지 추가: 선택한 텍스트 데이터에 적합한 이미지를 찾아 텍스트와 이미지 쌍을 만듭니다. CLIP 모델을 사용하여 텍스트와 이미지 간의 유사도를 평가하고, 관련성이 높은 쌍을 생성합니다.

#### Assessing multimodal data quality
아래는 학습 데이터의 품질을 평가하는 방법에 대해 설명하고 있습니다. 
VLM의 성능을 최적화 하기 위해서는 고품질의 교차 멀티모달 데이터가 필요합니다. 그러나 데이터 품질은 주관적인 지표이기 떄문에, 사전에 어떤 데이터가 좋은지 판단하기 어렵습니다.

- 데이터 품질 평가의 측면
  - 텍스트 품질
    - 텍스트의 일관성(Consistency), 명확성(Clarity), 문법적 정확성(Grammatical Accuracy) 등을 평가
    - Consistency 평가
      - BERT, GPT 등을 사용하여 텍스트의 앞 뒤 문장이 자연스럽게 이어지는 지 평가할 수 있습니다. 
      - Perplexity를 사용하여 텍스트의 예측 가능한 정도를 측정합니다. 낮은 Perplexity는 모델이 잘 예측 할 수 있음을 의미하며, 이는 일관성이 있음을 의미합니다. 
        - > Perplexity: 2^-l ( where l is sum of log p(w) )
    - Clarity 평가
      - 텍스트의 난이도를 평가하기 위해 다양한 읽기 점수 (Flesch-Kincaid, Gunning Fog Index) 를 사용합니다. 
      - 이러한 점수는 문장의 길이, 단어의 복잡성 등을 기반으로 계산됩니다. 
      - 점수가 높을 수록 이해하기 쉬움을 의미합니다. 
  - 이미지 품질
    - 이미지 해상도, 명확성, 미적 요소 등을 평가
  - 텍스트와 이미지 간의 일치도
    - CLIP 등을 사용하여, 텍스트와 이미지가 얼마나 잘 맞아 떨어지는 지 평가 
-  한계: 텍스트, 이미지, 그리고 텍스트-이미지 일치도를 각각 평가하는 방법은 있지만, 전체적으로 멀티모달 및 교차 데이터의 품질을 평가하는 통합적인 방법은 아직 부족합니다.

#### Harnessing human expertise
- 인간 주석의 중요성: 인간의 데이터를 활용하여 모델이 복잡한 장면을 더 잘 이해하고 정확한 설명을 생성하도록 할 수 있습니다.
- 기존 데이터셋의 한계: 기존 데이터셋은 오래된 이미지 소스에 의존하기 때문에, 더 다양한 최신 이미지 소스가 필요합니다.
- 새로운 데이터셋: DCI 데이터셋과 같은 새로운 데이터셋은 더 다양한 이미지와 세밀한 주석을 포함하고 있습니다.
- 한계: 인간 주석 데이터는 비용이 많이 들고, 세밀한 주석이 포함된 이미지 수가 적기 때문에, 대규모 사전 학습보다는 평가나 미세 조정에 더 적합합니다.

### 2. Software

이 섹션에서는 VLM(Vision-Language Model)을 평가하고 학습시키기 위해 사용할 수 있는 기존 소프트웨어와 필요한 리소스에 대해 논의합니다.

#### 기존 공개 소프트웨어 리포지토리 사용
- **소프트웨어 예시**: OpenCLIP (https://github.com/mlfoundations/open_clip), transformers (https://github.com/huggingface/transformers)
- **용도**: 대부분의 VLM을 구현하고 있으며, 벤치마킹이나 다양한 모델을 비교하는 데 매우 유용합니다.
- **활용**: 사전 학습된 VLM을 특정 다운스트림 작업에서 비교하고 평가하는 데 좋은 플랫폼을 제공합니다.

#### 필요한 GPU 수
- **리소스 중요성**: 필요한 계산 자원은 모델 학습 비용을 결정하는 중요한 요소입니다.
- **예시**: CLIP과 OpenCLIP은 모델을 학습시키기 위해 500개 이상의 GPU를 사용했습니다. 이러한 리소스를 공용 클라우드에서 사용하면 수십만 달러의 비용이 소요됩니다.
- **효율적인 학습**: 고품질 데이터셋과 마스킹 전략을 사용하면, 수백만 개의 이미지를 사용하는 대조 모델을 학습시키는 데 64개 정도의 GPU가 필요합니다(약 10K USD 소요). 기존의 사전 학습된 이미지나 텍스트 인코더 또는 LLM을 활용하면 학습 비용이 훨씬 낮아질 수 있습니다.

#### 학습 속도 향상
- **최근 소프트웨어 개발**: PyTorch 팀의 torch.compile(https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)은 모델 학습 속도를 크게 향상시킵니다.
- **xformers 라이브러리**: 효율적인 어텐션 메커니즘을 사용하여 추가적인 속도 향상을 제공합니다.
- **데이터 로딩 병목 현상**: 큰 미니배치의 이미지를 로드하는 것이 학습 속도를 크게 저하시킬 수 있습니다. 대규모 데이터셋은 tar 파일로 저장되어 압축을 풀어야 하는데, 이 과정이 학습 속도를 늦춥니다.
  - **해결책**: 가능한 한 많은 파일을 압축하지 않고 저장하고, FFCV (Fast Forward Computer Vision) 라이브러리(https://github.com/libffcv/ffcv)를 사용하여 데이터를 빠르게 로드합니다. FFCV를 사용하면 웹 데이터셋보다 학습 속도가 훨씬 빨라질 수 있습니다. 단점은 압축되지 않은 파일을 저장하는 데 더 많은 저장 공간이 필요하지만, 학습 속도 향상으로 인해 추가 저장 비용이 보상될 수 있습니다.

#### 마스킹
- **효율성 향상**: 큰 모델(수억 또는 수십억 개의 파라미터)을 사용할 때, 이미지 토큰을 무작위로 마스킹하면 학습 시간을 크게 단축하면서 모델 성능을 향상시킬 수 있습니다.
- **예시 연구**: Li et al. [2023f]의 연구는 마스킹을 통해 학습 효율성을 높이는 방법을 보여줍니다.

#### 다른 학습 파라미터의 중요성
McKinzie et al. [2024] VLM을 훈련할 때 가장 중요한 설계 선택사항을 연구하였습니다.
- 주요 하이퍼파라미터
  - 이미지 해상도
  - Visual Encoder Capacity
  - Visual Pretraining Data
  - 다양한 유형의 훈련 데이터: 텍스트 전용 데이터와 이미지-캡션 페어 데이터를 적절히 혼합하면, 제로샷 분류(zero-shot classification)와 시각적 질문 응답(visual-question answering) 작업에서 최고의 성능을 달성
- 연결 방식의 중요성
- 모달리티를 연결하는 방법은 상대적으로 모델 성능에 덜 중요한 영향을 미칩니다. 

### 3. 다양한 VLM 학습 방법

VLM을 학습하는 다양한 방법들이 존재하며, 각각의 접근법에는 장단점이 있습니다. 여기서는 대조 학습, 마스킹, 생성 모델, 그리고 사전 학습된 백본 사용의 각 상황에서의 활용 방안을 요약합니다.

#### 3.3.1 대조 모델 (Contrastive Models) 사용 시기
- **설명**: 대조 학습 모델인 CLIP은 텍스트와 시각적 개념을 연관시켜 동일한 표현 공간에서 텍스트와 이미지 표현을 매칭시킵니다.
- **장점**: 텍스트 인코더를 프롬프트로 사용하여 해당 텍스트 표현에 매핑되는 이미지를 검색할 수 있습니다. 데이터 큐레이션 파이프라인에서 유용하며, 특히 모델 구조 개선이나 관계 이해를 위한 추가 학습 기준을 시도하는 연구자들에게 좋습니다.
- **단점**: CLIP은 생성 모델이 아니므로 특정 이미지에 대해 캡션을 생성할 수 없습니다. 대규모 데이터셋과 큰 배치 크기를 필요로 하므로 많은 리소스가 필요합니다.

#### 3.3.2 마스킹 (Masking) 사용 시기
- **설명**: 마스킹 전략은 마스킹된 이미지나 텍스트 데이터를 재구성하는 것을 학습하여 그들의 분포를 공동 모델링합니다.
- **장점**: 각 예제를 개별적으로 처리할 수 있어 배치 의존성을 제거합니다. 이는 작은 미니 배치를 사용할 수 있게 하고 추가 하이퍼파라미터 조정이 필요 없게 합니다.
- **단점**: 추가 디코더를 훈련해야 하므로 효율성이 떨어질 수 있습니다.

#### 3.3.3 생성 모델 (Generative Models) 사용 시기
- **설명**: 생성 모델은 텍스트 프롬프트를 기반으로 포토리얼리스틱 이미지를 생성하는 능력을 가지고 있습니다. 많은 대규모 학습 노력은 이미지 생성 구성 요소를 통합하기 시작했습니다.
- **장점**: 텍스트와 이미지 간의 암묵적 공동 분포를 학습할 수 있으며, 모델이 학습한 내용을 이해하고 평가하기가 더 쉽습니다.
- **단점**: 대조 학습 모델보다 계산 비용이 더 많이 듭니다.

#### 3.3.4 사전 학습된 백본 (Pretrained Backbone) 사용 시기
- **설명**: 사전 학습된 텍스트나 시각적 인코더를 사용하는 것은 제한된 리소스를 사용할 때 좋은 대안이 될 수 있습니다.
- **장점**: 텍스트 표현과 시각적 표현 간의 매핑만 학습하면 됩니다.
- **단점**: LLM의 잠재적인 환각이나 사전 학습된 모델의 편향에 영향을 받을 수 있습니다. 이러한 결함을 수정하는 데 추가적인 작업이 필요할 수 있습니다.

### 요약
- **대조 모델**: 단순한 학습 기준으로 텍스트와 이미지 표현을 매칭, 대규모 데이터셋과 배치 크기 필요.
- **마스킹**: 개별 예제를 처리하여 배치 의존성 제거, 추가 디코더 필요.
- **생성 모델**: 텍스트 기반 포토리얼리스틱 이미지 생성, 계산 비용 높음.
- **사전 학습된 백본**: 제한된 리소스 사용 시 적합, LLM의 환각 및 편향 영향을 받을 수 있음.

각 접근법은 특정 상황에서 장단점이 있으며, 사용자는 자신의 리소스와 목표에 따라 적절한 방법을 선택해야 합니다.