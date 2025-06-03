import pytest
import torch

from layers.GatingMechanism import FeatureGate

# -------------------- helpers -------------------- #
B, L, DF, DT = 2, 4, 8, 6                      # small but non‑trivial

GATES_WEIGHT_FUSED   = {"mlp", "vector_gate_linear"}
GATES_WEIGHT_TS      = {
    "simple_linear",
    "lightweight_linear",
    "per_token_scalar",
    "global_scalar",
}


def expected_gate_shape(gate_type: str):
    if gate_type in {"mlp", "simple_linear", "lightweight_linear", "vector_gate_linear"}:
        return (B, L, DT)
    if gate_type == "per_token_scalar":
        return (B, L, 1)
    if gate_type == "global_scalar":
        return (B, 1, 1)
    raise ValueError(gate_type)


# -------------------- parameterised test -------------------- #
@pytest.mark.parametrize(
    "gate_type",
    [
        "mlp",
        "simple_linear",
        "lightweight_linear",
        "vector_gate_linear",
        "per_token_scalar",
        "global_scalar",
    ],
)
def test_feature_gate_forward(gate_type):
    torch.manual_seed(0)

    F = torch.randn(B, L, DF, requires_grad=True)   # fused_latent_features
    T = torch.randn(B, L, DT, requires_grad=True)   # ts_features

    gate = FeatureGate(fused_dim=DF, ts_dim=DT, gate_type=gate_type)
    O, G = gate(F, T)

    # ----- 1.  output shapes -----
    assert O.shape == (B, L, DT)
    assert G.shape == expected_gate_shape(gate_type)

    # ----- 2.  gate value range (sigmoid) -----
    assert torch.all(G >= 0.0) and torch.all(G <= 1.0)

    # ----- 3.  equation correctness -----
    F_proj = gate.fused_projection(F)               # replicate internal projection

    if gate_type in GATES_WEIGHT_FUSED:
        O_expected = G * F_proj + (1 - G) * T
    else:  # gate weights TS
        O_expected = G * T + (1 - G) * F_proj

    torch.testing.assert_close(O, O_expected, atol=1e-6, rtol=1e-6)

    # ----- 4.  gradient flows -----
    loss = O.sum()
    loss.backward()
    assert F.grad is not None and T.grad is not None, "Gradients did not propagate"


# -------------------- edge‑case sanity (broadcast) ------------ #
def test_global_scalar_broadcast():
    torch.manual_seed(42)
    gate = FeatureGate(DF, DT, gate_type="global_scalar")
    F = torch.randn(B, L, DF)
    T = torch.randn(B, L, DT)

    _, beta = gate(F, T)
    # beta shape should be (B,1,1), but values must be identical across L, DT
    assert beta.shape == (B, 1, 1)
    assert torch.allclose(beta.expand(-1, L, DT)[:, 0, 0], beta.squeeze())