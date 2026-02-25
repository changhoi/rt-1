import torch
from rt1.film import FiLMLayer, FiLMEfficientNet


class TestFiLMLayer:
    def test_identity_at_init(self):
        """zero-init 상태에서 FiLM이 identity 함수인지 확인.

        (1 + 0) * x + 0 = x 이어야 한다.
        """
        film = FiLMLayer(lang_dim=512, channel_dim=64)
        x = torch.randn(2, 64, 10, 10)
        lang = torch.randn(2, 512)
        out = film(x, lang)
        assert torch.allclose(out, x, atol=1e-6), "FiLM should be identity at initialization"

    def test_output_shape(self):
        """출력 shape이 입력과 동일한지 확인."""
        film = FiLMLayer(lang_dim=512, channel_dim=128)
        x = torch.randn(4, 128, 8, 8)
        lang = torch.randn(4, 512)
        out = film(x, lang)
        assert out.shape == x.shape

    def test_different_lang_changes_output(self):
        """다른 언어 임베딩이 다른 출력을 만드는지 확인 (학습 후)."""
        film = FiLMLayer(lang_dim=512, channel_dim=64)
        # zero-init을 깨고 학습된 상태 시뮬레이션
        with torch.no_grad():
            film.gamma_linear.weight.fill_(0.1)
            film.beta_linear.weight.fill_(0.1)

        x = torch.randn(1, 64, 5, 5)
        lang_a = torch.randn(1, 512)
        lang_b = torch.randn(1, 512)
        out_a = film(x, lang_a)
        out_b = film(x, lang_b)
        assert not torch.allclose(out_a, out_b), "Different languages should produce different outputs"


class TestFiLMEfficientNet:
    def test_output_shape(self):
        """300x300 이미지 → (B, 81, 512) 토큰."""
        model = FiLMEfficientNet(lang_dim=512, pretrained=False)
        images = torch.randn(2, 3, 300, 300)
        lang = torch.randn(2, 512)
        tokens = model(images, lang)
        assert tokens.shape == (2, 81, 512)

    def test_film_layer_count(self):
        """FiLM 레이어가 26개인지 확인."""
        model = FiLMEfficientNet(lang_dim=512, pretrained=False)
        assert len(model.film_layers) == 26

    def test_gradient_flow(self):
        """gradient가 FiLM layers와 token_proj까지 전파되는지."""
        model = FiLMEfficientNet(lang_dim=512, pretrained=False)
        images = torch.randn(1, 3, 300, 300)
        lang = torch.randn(1, 512, requires_grad=True)
        tokens = model(images, lang)
        loss = tokens.sum()
        loss.backward()
        assert lang.grad is not None
        assert model.film_layers[0].gamma_linear.weight.grad is not None
