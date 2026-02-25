import torch
from rt1.action_tokenizer import ActionTokenizer, ActionHead


class TestActionTokenizer:
    def setup_method(self):
        self.mins = torch.tensor([-1.0] * 11)
        self.maxs = torch.tensor([1.0] * 11)
        self.tokenizer = ActionTokenizer(self.mins, self.maxs, num_bins=256)

    def test_encode_shape(self):
        """encode 출력이 (B, 11) long tensor인지 확인."""
        actions = torch.rand(4, 11) * 2 - 1  # [-1, 1]
        bins = self.tokenizer.encode(actions)
        assert bins.shape == (4, 11)
        assert bins.dtype == torch.long

    def test_encode_range(self):
        """bin index가 [0, 255] 범위인지 확인."""
        actions = torch.rand(4, 11) * 2 - 1
        bins = self.tokenizer.encode(actions)
        assert (bins >= 0).all()
        assert (bins <= 255).all()

    def test_encode_boundaries(self):
        """최솟값 → bin 0, 최댓값 → bin 255."""
        min_actions = self.mins.unsqueeze(0)  # (1, 11)
        max_actions = self.maxs.unsqueeze(0)  # (1, 11)
        assert (self.tokenizer.encode(min_actions) == 0).all()
        assert (self.tokenizer.encode(max_actions) == 255).all()

    def test_decode_shape(self):
        """decode 출력이 (B, 11) float tensor인지 확인."""
        bins = torch.randint(0, 256, (4, 11))
        actions = self.tokenizer.decode(bins)
        assert actions.shape == (4, 11)
        assert actions.dtype == torch.float32

    def test_roundtrip(self):
        """encode → decode 왕복 시 오차가 bin 크기 이내인지 확인."""
        actions = torch.rand(100, 11) * 2 - 1
        bins = self.tokenizer.encode(actions)
        recovered = self.tokenizer.decode(bins)
        # bin 크기 = 2.0 / 255 ≈ 0.0078
        assert (actions - recovered).abs().max() < 0.01

    def test_clamp(self):
        """범위 밖 값이 clamp되는지 확인."""
        actions = torch.tensor([[-5.0] * 11, [5.0] * 11])
        bins = self.tokenizer.encode(actions)
        assert (bins[0] == 0).all()
        assert (bins[1] == 255).all()


class TestActionHead:
    def test_output_shape(self):
        """ActionHead 출력이 (B, 11, 256)인지 확인."""
        head = ActionHead(d_model=512, action_dims=11, num_bins=256)
        x = torch.randn(4, 512)
        logits = head(x)
        assert logits.shape == (4, 11, 256)

    def test_gradient_flow(self):
        """loss.backward()가 정상 동작하는지 확인."""
        head = ActionHead(d_model=512, action_dims=11, num_bins=256)
        x = torch.randn(2, 512, requires_grad=True)
        logits = head(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None
