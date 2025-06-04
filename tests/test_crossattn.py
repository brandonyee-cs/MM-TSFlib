import pytest
import torch

from layers.CrossAttention import MultiheadLatentAttention

def test_output_shape_and_dtype():
    B, L_q, L_k = 2, 5, 7
    model = MultiheadLatentAttention(
        query_dim=32, key_dim=48, latent_dim=64, num_heads=8, dropout=0.0
    )

    q = torch.randn(B, L_q, 32)
    k = torch.randn(B, L_k, 48)
    v = torch.randn(B, L_k, 48)

    out = model(q, k, v)
    assert out.shape == (B, L_q, 64)
    assert out.dtype == q.dtype


def test_gradients_flow():
    model = MultiheadLatentAttention(16, 16, 32, num_heads=8, dropout=0.0)
    q = torch.randn(4, 3, 16, requires_grad=True)
    k = torch.randn(4, 6, 16, requires_grad=True)
    v = torch.randn(4, 6, 16, requires_grad=True)

    loss = model(q, k, v).sum()
    loss.backward()

    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.any(tensor.grad != 0)


def test_attention_mask_effect():
    torch.manual_seed(0)  # deterministic

    model = MultiheadLatentAttention(8, 8, 16, num_heads=4, dropout=0.0)
    q = torch.randn(1, 1, 8)
    k = torch.randn(1, 3, 8)
    v = torch.randn(1, 3, 8)

    full_mask = torch.ones(1, 1, 1, 3)        # allow all keys
    half_mask = torch.tensor([[[[1, 1, 0]]]])  # block the last key

    out_full = model(q, k, v, full_mask)
    out_half = model(q, k, v, half_mask)

    # With the last key blocked, outputs should differ.
    assert not torch.allclose(out_full, out_half, atol=1e-5)
