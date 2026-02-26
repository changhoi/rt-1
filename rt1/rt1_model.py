import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import SentenceTransformer

from rt1.config import RT1Config
from rt1.film import FiLMEfficientNet
from rt1.token_learner import TokenLearner
from rt1.transformer import CausalTransformer
from rt1.action_tokenizer import ActionHead


class RT1(nn.Module):
    """RT-1: Robotics Transformer for Real-World Control at Scale.

    Pipeline:
        1. Instruction → SentenceTransformer (frozen) → Linear(384→512) → lang_embed
        2. 6 images each → FiLM-EfficientNet(lang_embed) → 81 tokens → TokenLearner → 8 tokens
        3. 6×8 = 48 tokens → CausalTransformer → 48 outputs
        4. Last token → ActionHead → (11, 256) logits
    """

    def __init__(self, config: RT1Config):
        super().__init__()
        self.config = config
        self.lang_encoder = SentenceTransformer(config.lang_model)
        if config.freeze_lang_encoder:
            self.lang_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(config.lang_raw_dim, config.lang_embed_dim)
        self.film_efficientnet = FiLMEfficientNet(
            lang_dim=config.lang_embed_dim,
            pretrained=config.use_pretrained,
        )
        self.token_learner = TokenLearner(
            input_dim=config.token_dim,
            num_tokens=config.num_learned_tokens,
            hidden_dim=config.token_learner_hidden_dim,
        )
        self.transformer = CausalTransformer(
            d_model=config.token_dim,
            nhead=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.action_head = ActionHead(
            d_model=config.token_dim,
            action_dims=config.action_dims,
            num_bins=config.num_action_bins,
        )

    def encode_instruction(self, instructions: list[str]) -> Tensor:
        """Encode natural language instructions to 512-dim vectors.

        Args:
            instructions: List of B instruction strings
                e.g. ["pick up the red ball", "move to the left"]

        Returns:
            (B, 512) language embeddings
        """
        with torch.no_grad():
            raw = self.lang_encoder.encode(instructions, convert_to_tensor=True)
            raw = raw.to(self.lang_proj.weight.device)
        return self.lang_proj(raw)

    def tokenize_images(self, images: Tensor, lang_embed: Tensor) -> Tensor:
        """Convert 6 images into 48 tokens.

        Args:
            images: (B, 6, 3, 300, 300) - 6 timesteps of images
            lang_embed: (B, 512)

        Returns:
            (B, 48, 512)
        """
        all_tokens = []
        T = images.size(1)
        for t in range(T):
            img_t = images[:, t]
            vl_tokens = self.film_efficientnet(img_t, lang_embed)
            compressed = self.token_learner(vl_tokens)
            all_tokens.append(compressed)

        return torch.cat(all_tokens, dim=1)

    def forward(self, images: Tensor, instructions: list[str]) -> Tensor:
        """RT-1 forward pass.

        Args:
            images: (B, 6, 3, 300, 300)
            instructions: List of B instruction strings

        Returns:
            (B, 11, 256) action logits
        """
        lang_embed = self.encode_instruction(instructions=instructions)
        tokens = self.tokenize_images(images=images, lang_embed=lang_embed)
        output = self.transformer(tokens)
        last_token = output[:, -1, :]
        action_logits = self.action_head(last_token)
        return action_logits

    def compute_loss(self, action_logits: Tensor, action_targets: Tensor) -> Tensor:
        """Mean cross-entropy loss across all action dimensions.

        Args:
            action_logits: (B, 11, 256) - model output
            action_targets: (B, 11) - target bin indices (long)

        Returns:
            scalar loss
        """
        loss = torch.tensor(0.0)
        for d in range(self.config.action_dims):
            loss += F.cross_entropy(action_logits[:, d, :], action_targets[:, d])

        return loss / self.config.action_dims
