import torch
from rt1.config import RT1Config
from rt1.rt1_model import RT1


class TestRT1:
    def test_forward_shape(self):
        """전체 모델 forward pass 출력 shape 확인."""
        config = RT1Config(use_pretrained=False)
        model = RT1(config)

        images = torch.randn(2, 6, 3, 300, 300)
        instructions = ["pick up the red ball", "move to the left"]
        logits = model(images, instructions)

        assert logits.shape == (2, 11, 256)

    def test_backward(self):
        """전체 모델에 gradient가 흐르는지 확인."""
        config = RT1Config(use_pretrained=False)
        model = RT1(config)

        images = torch.randn(1, 6, 3, 300, 300)
        instructions = ["pick up the cup"]
        targets = torch.randint(0, 256, (1, 11))

        logits = model(images, instructions)
        loss = model.compute_loss(logits, targets)
        loss.backward()

        # FiLM layers에 gradient가 있는지
        assert model.film_efficientnet.film_layers[0].gamma_linear.weight.grad is not None
        # Language projection에 gradient가 있는지
        assert model.lang_proj.weight.grad is not None

    def test_param_count(self):
        """학습 가능 파라미터가 ~35M인지 확인 (frozen lang encoder 제외)."""
        config = RT1Config(use_pretrained=False)
        model = RT1(config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable / 1e6:.1f}M")
        assert 30_000_000 < trainable < 40_000_000, f"Expected ~35M, got {trainable/1e6:.1f}M"
