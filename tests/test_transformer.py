import torch
from rt1.transformer import CausalTransformer


class TestCausalTransformer:
    def test_output_shape(self):
        """출력 shape이 입력과 동일한지 확인."""
        transformer = CausalTransformer(
            d_model=512, nhead=8, num_layers=8, d_ff=1024, max_seq_len=48
        )
        x = torch.randn(2, 48, 512)
        out = transformer(x)
        assert out.shape == (2, 48, 512)

    def test_causal_masking(self):
        """미래 토큰을 변경해도 과거 토큰 출력이 변하지 않는지 확인.

        Causal mask가 제대로 동작하면:
        - 토큰 24~47을 바꿔도 토큰 0~23의 출력은 동일해야 한다.
        """
        # 작은 모델로 테스트 (속도)
        transformer = CausalTransformer(
            d_model=64, nhead=4, num_layers=2, d_ff=128, max_seq_len=48
        )
        transformer.eval()

        x = torch.randn(1, 48, 64)
        out1 = transformer(x)

        x_modified = x.clone()
        x_modified[:, 24:, :] += 10.0  # 뒷절반만 변경
        out2 = transformer(x_modified)

        # 앞 24개 토큰의 출력은 동일해야 함
        assert torch.allclose(out1[:, :24, :], out2[:, :24, :], atol=1e-5), (
            "Causal mask failed: changing future tokens affected past outputs"
        )

    def test_gradient_flow(self):
        """gradient가 정상 전파되는지 확인."""
        transformer = CausalTransformer(
            d_model=64, nhead=4, num_layers=2, d_ff=128, max_seq_len=48
        )
        x = torch.randn(1, 48, 64, requires_grad=True)
        out = transformer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
