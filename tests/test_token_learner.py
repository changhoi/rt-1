import torch
from rt1.token_learner import TokenLearner


class TestTokenLearner:
    def test_output_shape(self):
        """81 tokens → 8 tokens 압축 shape 확인."""
        tl = TokenLearner(input_dim=512, num_tokens=8, hidden_dim=64)
        x = torch.randn(4, 81, 512)
        out = tl(x)
        assert out.shape == (4, 8, 512)

    def test_param_count(self):
        """파라미터 수가 ~34K인지 확인."""
        tl = TokenLearner(input_dim=512, num_tokens=8, hidden_dim=64)
        total = sum(p.numel() for p in tl.parameters())
        # LayerNorm(512): 1024, Linear(512,64): 32832, Linear(64,8): 520
        # Total: ~34376
        assert 33_000 < total < 36_000, f"Expected ~34K, got {total}"

    def test_gradient_flow(self):
        """gradient가 입력까지 전파되는지 확인."""
        tl = TokenLearner(input_dim=512, num_tokens=8, hidden_dim=64)
        x = torch.randn(2, 81, 512, requires_grad=True)
        out = tl(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_different_num_tokens(self):
        """다른 num_tokens 값으로도 동작하는지 확인."""
        tl = TokenLearner(input_dim=256, num_tokens=4, hidden_dim=32)
        x = torch.randn(2, 49, 256)
        out = tl(x)
        assert out.shape == (2, 4, 256)
