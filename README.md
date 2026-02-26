# RT-1: Robotics Transformer

PyTorch implementation of the RT-1 paper.

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
  │  MLP(512→64→8)                (B, 81, 8)   per-position │
  │  sigmoid + normalize          (B, 8, 81)   attention    │
  │  bmm(attn, x)                 (B, 8, 512)  weighted sum │
  └─────────────────────────────────────────────────────────┘
    │
    ▼  6 frames × 8 tokens = 48 tokens
  ┌─────────────────────────────────────────────────────────┐
  │              CausalTransformer                          │
  │                                                         │
  │  8 layers, 8 heads, d_model=512                         │
  │  input: (B, 48, 512) → output: (B, 48, 512)            │
  └─────────────────────────────────────────────────────────┘
    │  last token
    ▼
  ActionHead → (B, 11, 256) logits
    │
    ▼
  11 action dims × 256 bins
  (arm 7 + base 3 + mode 1)
```

## Components

### 1. FiLM Layer

Feature-wise Linear Modulation. Conditions image feature maps with language embeddings.

```
output = (1 + gamma) * x + beta
gamma, beta = Linear(lang_embed)
```

- Zero-init ensures identity at start (`gamma=0, beta=0 → output=x`)
- Preserves pretrained EfficientNet weights initially, gradually learns language conditioning

### 2. FiLM-EfficientNet

Inserts a FiLM layer after each of the 26 MBConv blocks in the EfficientNet-B3 backbone.

- **EfficientNet**: Image feature extractor. Abstracts input images from low-level features (edges, textures) to high-level features (objects, parts) through multiple convolution stages
- **Backbone**: The pure feature extraction part of a pretrained model, with the classification head removed
> **Classification head**: The final part of EfficientNet originally used for ImageNet classification (1000 classes) (`AvgPool → Linear → softmax`). Removed in RT-1 since classification is not needed; only the backbone is used
- **384**: Output channel count of the last EfficientNet-B3 block (fixed by model architecture)
- **AdaptiveAvgPool2d(9)**: Reduces the backbone output to a fixed 9×9 spatial resolution via average pooling. "Adaptive" means it guarantees the desired output size (9×9) regardless of input size
- **9×9 = 81**: Reduces spatial resolution to 9×9, creating 81 spatial positions (tokens)
- **512**: Unified token dimension matching the Transformer (`token_dim`)
- **1×1 Conv2d**: Preserves spatial size, only transforms channel count 384→512. Functionally identical to `nn.Linear`, but operates directly on the channel dimension of 4D tensors `(B, C, H, W)` without needing permute — idiomatic in CNN pipelines
> **Linear (nn.Linear)**: A learnable linear transformation that maps between dimensions via matrix multiplication. `output = input @ W^T + b`

### 3. TokenLearner

Compresses 81 vision-language tokens into 8 learned tokens.

Each learned token is a weighted sum of 81 input tokens:
- MLP scores each of the 81 positions with 8 values: "how important is this position for learned token i?"
- sigmoid + normalize produces a probability-like distribution
> **Why sigmoid + /sum**: sigmoid converts values to 0–1 ensuring positivity, /sum normalizes to sum=1 creating "ratios". Unlike softmax, preserves original ratios for smoother soft selection. Without sigmoid, negative MLP outputs would break the ratio interpretation
- `bmm` computes weighted sum → 8 new tokens
> **Learned token**: Unlike grid positions defined by human rules, these are tokens whose content the model learns to determine during training

### 4. CausalTransformer

Processes the 48-token sequence (6 frames × 8 tokens). Passes the last token's output to ActionHead.

- **Positional Embedding**: Adds a learnable 512-dim vector to each of the 48 positions. Transformer has no inherent notion of order, so positional information must be injected
- **TransformerEncoderLayer**: A single layer performing Self-Attention → FFN. Stacked 8 layers deep
  - `nhead=8`: Splits 512 dimensions into 8 parallel attention heads (64-dim each)
  - `norm_first=True`: Places LayerNorm before attention/FFN (Pre-LN). More stable training
> **FFN (Feed-Forward Network)**: Per-token MLP applied independently. While Self-Attention handles "inter-token relationships", FFN "transforms each token's representation". Expands `512→1024→512` for richer nonlinear transformations

> **Dropout**: Randomly zeroes 10% of neuron outputs during training to prevent overfitting. Forces the model to learn general patterns instead of relying on specific neurons

> **Why TransformerEncoder**: "Decoder-only" (GPT-style) = self-attention + causal mask. PyTorch's `TransformerDecoder` includes cross-attention (for encoder-decoder architectures), so `TransformerEncoder` + causal mask is the correct decoder-only implementation

### 5. ActionHead

Transforms a 512-dim vector into 11 action dims × 256 bins of logits. Discretizes continuous actions into a classification problem. Each action dimension has an independent `Linear(512, 256)`, totaling 11 heads (`nn.ModuleList`).

> **Bins**: Continuous values divided into equal intervals. e.g., arm_x range -1.0~1.0 split into 256 bins — instead of "move 0.35", it becomes "bin 170". Converts continuous regression into classification

> **Logits**: Raw scores before softmax. Represents "likelihood of each bin being correct" for 256 bins. During training, `cross_entropy` handles softmax internally, so logits are passed as-is

### 6. ActionTokenizer

Utility outside the model that converts between continuous action values and bin indices.

- `encode`: continuous action `0.35` → bin index `170` (converts training labels to bin indices)
- `decode`: bin index `170` → continuous action `0.35` (converts model output back to robot commands)

```
Training:   ground truth action → encode → bin index → cross_entropy(model output, target bin)
Inference:  model output logits → argmax → bin index → decode → continuous value for robot
```
