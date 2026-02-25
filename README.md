# RT-1: Robotics Transformer

RT-1 논문의 PyTorch 구현체.

## Architecture

```
Instruction (text)
    │
    ▼
SentenceTransformer (frozen) → Linear(384→512) → lang_embed (B, 512)
                                                       │
Images (B, 6, 3, 300, 300)                             │
    │                                                   │
    ▼                                                   ▼
  ┌─────────────────────────────────────────────────────────┐
  │              FiLM-EfficientNet (×6 frames)              │
  │                                                         │
  │  conv_stem + bn1                                        │
  │       ↓                                                 │
  │  26× [MBConv block → FiLM layer(lang_embed)]            │
  │       ↓                                                 │
  │  AdaptiveAvgPool2d(9)        (B, 384, 9, 9)             │
  │  Conv2d(384→512, 1×1)        (B, 512, 9, 9)             │
  │  flatten + permute            (B, 81, 512)              │
  └─────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────┐
  │                  TokenLearner (×6 frames)                │
  │                                                         │
  │  LayerNorm                    (B, 81, 512)              │
  │  MLP(512→64→8)                (B, 81, 8)   위치별 점수   │
  │  sigmoid + normalize          (B, 8, 81)   attention    │
  │  bmm(attn, x)                 (B, 8, 512)  가중합       │
  └─────────────────────────────────────────────────────────┘
    │
    ▼  6 frames × 8 tokens = 48 tokens
  ┌─────────────────────────────────────────────────────────┐
  │              CausalTransformer                          │
  │                                                         │
  │  8 layers, 8 heads, d_model=512                         │
  │  입력: (B, 48, 512) → 출력: (B, 48, 512)               │
  └─────────────────────────────────────────────────────────┘
    │  마지막 토큰
    ▼
  ActionHead → (B, 11, 256) logits
    │
    ▼
  11개 action 차원 × 256 bins
  (arm 7 + base 3 + mode 1)
```

## Components

### 1. FiLM Layer

Feature-wise Linear Modulation. 이미지 feature map을 언어 임베딩으로 조건화한다.

```
output = (1 + gamma) * x + beta
gamma, beta = Linear(lang_embed)
```

- zero-init으로 학습 초기에는 identity (`gamma=0, beta=0 → output=x`)
- pretrained EfficientNet 가중치를 초기에 보존하면서 점진적으로 언어 조건화를 학습

### 2. FiLM-EfficientNet

EfficientNet-B3 backbone의 26개 MBConv 블록 각각에 FiLM layer를 삽입한 구조.

- **EfficientNet**: 이미지 feature 추출기. 입력 이미지를 여러 단계의 convolution을 거쳐 저수준 특징(에지, 텍스처)에서 고수준 특징(물체, 부분)으로 추상화한다
- **Backbone**: pretrained 모델에서 분류 head를 제거한 순수 feature 추출 부분
> **분류 head**: EfficientNet 원래 목적인 ImageNet 분류(1000개 클래스)를 위한 마지막 부분 (`AvgPool → Linear → softmax`). RT-1에서는 분류가 필요 없으므로 제거하고 backbone만 사용
- **384**: EfficientNet-B3 마지막 블록의 출력 채널 수 (모델 구조에 의해 고정)
- **AdaptiveAvgPool2d(9)**: backbone 출력의 공간 해상도가 입력 크기에 따라 달라질 수 있는데, 이를 고정된 9×9로 만들어줌. 인접 영역을 평균 풀링하여 축소하며, "Adaptive"는 입력 크기에 관계없이 원하는 출력 크기(9×9)를 보장한다는 의미
- **9×9 = 81**: 공간 해상도를 9×9로 줄여 81개 위치(토큰)를 생성
- **512**: Transformer와 통일된 토큰 차원 (`token_dim`)
- **1×1 Conv2d**: 공간 크기 유지, 채널 수만 384→512로 변환. `nn.Linear`와 동일한 선형 변환이지만, 입력이 4D 텐서 `(B, C, H, W)`일 때 permute 없이 채널 차원에 바로 작용할 수 있어서 CNN 파이프라인에서 관용적으로 사용
> **Linear (nn.Linear)**: 학습 가능한 행렬곱으로 차원 간 매핑을 배우는 선형 변환. `output = input @ W^T + b`

### 3. TokenLearner

81개 vision-language 토큰을 8개 learned 토큰으로 압축한다.

각 learned 토큰은 81개 입력 토큰의 가중합(weighted sum):
- MLP가 81개 위치 각각에 "이 위치가 i번째 learned token에 얼마나 중요한가"를 8개 점수로 매김
- sigmoid + normalize로 확률 분포처럼 만듦
> **왜 sigmoid + /sum인가**: sigmoid는 값을 0~1로 변환하여 양수를 보장하고, /sum은 합=1로 정규화하여 "비율"로 만든다. softmax와 달리 원래 비율을 보존하여 부드러운 soft selection을 구현. sigmoid 없이 MLP 출력을 바로 /sum하면 음수가 섞여 비율로서 의미가 깨진다
- `bmm`으로 가중합 계산 → 8개의 새 토큰 생성
> **learned token**: 격자 위치처럼 사람이 정한 규칙이 아니라, 모델이 학습 과정에서 어떤 정보를 담을지 스스로 배운 토큰

### 4. CausalTransformer

6 프레임 × 8 토큰 = 48개 토큰 시퀀스를 처리. 마지막 토큰의 출력을 ActionHead에 전달.

- **Positional Embedding**: 48개 위치 각각에 학습 가능한 512차원 벡터를 더해줌. Transformer는 구조상 입력 순서를 모르기 때문에 "몇 번째 위치인지" 정보를 주입
- **TransformerEncoderLayer**: Self-Attention → FFN을 수행하는 하나의 레이어. 이를 8층 쌓음
  - `nhead=8`: 512차원을 8개(각 64차원)로 나눠 병렬 attention
  - `norm_first=True`: LayerNorm을 attention/FFN 앞에 배치 (Pre-LN). 학습이 더 안정적
> **FFN (Feed-Forward Network)**: 토큰별로 독립 적용되는 MLP. Self-Attention이 "토큰 간 관계"를 처리했다면, FFN은 "각 토큰의 표현을 변환". `512→1024→512`로 더 넓은 공간에서 비선형 변환 후 축소하여 표현력을 높임

> **Dropout**: 학습 중 뉴런 출력을 랜덤하게 10% 꺼서 과적합 방지. 특정 뉴런에 의존하지 못하게 하여 일반적인 패턴을 학습하도록 강제

> **왜 TransformerEncoder인가**: "Decoder-only"(GPT 스타일)는 self-attention + causal mask. PyTorch의 `TransformerDecoder`는 cross-attention 포함(encoder-decoder 구조용)이므로, `TransformerEncoder` + causal mask가 decoder-only의 올바른 구현

### 5. ActionHead

512차원 벡터를 11개 action 차원 × 256 bins의 logits로 변환. 연속 action을 이산화(discretize)하여 분류 문제로 처리한다. 각 action 차원마다 독립적인 `Linear(512, 256)`을 하나씩 두어 총 11개의 head로 구성 (`nn.ModuleList`).

> **bins**: 연속값을 일정 구간으로 나눈 것. 예를 들어 arm_x 범위 -1.0~1.0을 256개 구간으로 쪼개서, "0.35만큼 움직여"가 아니라 "bin 170이야"로 처리. 연속 회귀 문제를 분류 문제로 변환

> **logits**: softmax 적용 전의 raw 점수. 256개 bin 각각에 대해 "이 bin이 정답일 가능성"을 나타냄. 학습 시 `cross_entropy`가 내부에서 softmax를 처리하므로 logits 상태로 넘김

### 6. ActionTokenizer

모델 외부에서 연속 action 값 ↔ bin index 변환을 담당하는 유틸리티.

- `encode`: 연속 action `0.35` → bin index `170` (학습 데이터의 정답을 bin으로 변환)
- `decode`: bin index `170` → 연속 action `0.35` (모델 출력을 실제 로봇 명령으로 복원)

```
학습 시: 정답 action → encode → bin index → cross_entropy(모델출력, 정답bin)
추론 시: 모델출력 logits → argmax → bin index → decode → 로봇에 보낼 연속값
```
