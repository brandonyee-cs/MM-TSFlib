import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureGate(nn.Module):
    """
    Gating mechanism to control the fusion of multimodal features with multiple options.
    
    Handles projecting fused features from latent dimension to time series dimension
    and computes a gate value to combine with original time series features.
    
    Options for gate computation (`gate_type`):
    - 'mlp': Original two-layer MLP gate network.
    - 'simple_linear': Simple adaptive gate (alpha) using one linear layer on concatenated features.
    - 'lightweight_linear': Lightweight linear gate using LayerNorm.
    - 'vector_gate_linear': Lightweight linear gate (G) using LayerNorm, applied like MLP gate.
    - 'per_token_scalar': Linear gate producing one scalar per token.
    - 'global_scalar': Mean-pooled inputs to a linear gate producing one scalar per batch.
    """
    def __init__(self, fused_dim, ts_dim, gate_type='mlp', hidden_dim=None):
        super(FeatureGate, self).__init__()
        self.gate_type = gate_type
        self.fused_dim = fused_dim # latent_dim from previous step
        self.ts_dim = ts_dim # d_model

        # --- Projection Layer (Common to all gate types) ---
        # Projects fused features (latent_dim) to time series dimension (ts_dim)
        # Uses hidden_dim only if gate_type is 'mlp' (for compatibility)
        proj_hidden_dim = hidden_dim if hidden_dim is not None else 2 * ts_dim
        self.fused_projection = nn.Sequential(
            nn.Linear(fused_dim, proj_hidden_dim), # Use proj_hidden_dim for potential MLP compatibility
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, ts_dim)
        )

        # --- Gate Computation Network (Varies by gate_type) ---
        if gate_type == 'mlp':
            if hidden_dim is None:
                hidden_dim = 2 * ts_dim
            self.gate_network = nn.Sequential(
                nn.Linear(fused_dim + ts_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, ts_dim),
                nn.Sigmoid()
            )
        elif gate_type == 'simple_linear':
            # Computes alpha = sigma(Linear([F, T]))
            self.gate_network = nn.Sequential(
                nn.Linear(fused_dim + ts_dim, ts_dim),
                nn.Sigmoid()
            )
        elif gate_type == 'lightweight_linear' or gate_type == 'vector_gate_linear': # Use same computation
            # Computes gate = sigma(LayerNorm(Linear([F, T])))
            self.gate_network = nn.Sequential(
                nn.Linear(fused_dim + ts_dim, ts_dim),
                nn.LayerNorm(ts_dim), # Apply LayerNorm before sigmoid
                nn.Sigmoid()
            )
        elif gate_type == 'per_token_scalar' or gate_type == 'global_scalar': # Use same computation
            # Computes beta = sigma(Linear([F+T, 1]))
            self.gate_network = nn.Sequential(
                nn.Linear(fused_dim + ts_dim, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}. Choose 'mlp', 'simple_linear', 'lightweight_linear', 'vector_gate_linear', 'per_token_scalar', 'global_scalar'.")

    def forward(self, fused_latent_features, ts_features):
        """
        Apply gating mechanism to control information flow.
        
        Args:
            fused_latent_features: Features from cross-attention/post-fusion (B, L, fused_dim)
            ts_features: Original time series features from iTransformer (B, L, ts_dim)
            
        Returns:
            gated_output: Combined features with learned gating (B, L, ts_dim)
            gate_value: The computed gate for potential regularization. 
                        For per_token_scalar, returns (B, L, 1)
                        For global_scalar, returns (B, 1, 1)
                        For all other gates, returns (B, L, ts_dim)
        """
        # Project fused features from latent dimension to time series dimension
        projected_fused = self.fused_projection(fused_latent_features) # F' (B, L, ts_dim)
        
        # Compute gate value (G, alpha, or beta)
        # Input features for gate computation need the pre-projection fused features (F)
        concat_features = torch.cat([fused_latent_features, ts_features], dim=-1) # [F, T]

        if self.gate_type == 'global_scalar':
            pooled = concat_features.mean(dim=1)
            beta = self.gate_network(pooled)
            gate_value = beta.unsqueeze(1)
        else:
            gate_value = self.gate_network(concat_features) # G, alpha, or beta
        
        # Apply one of gate G, alpha, or beta: 
        # O = G * F' + (1 - G) * T
        # O = alpha * T + (1 - alpha) * F'
        # O = beta * T + (1 - beta) * F'
        if self.gate_type in {'mlp', 'vector_gate_linear'}:
            # Gate weights the projected fused features (F')
            gated_output = gate_value * projected_fused + (1 - gate_value) * ts_features
        elif self.gate_type in {'simple_linear', 'lightweight_linear', 
                                'per_token_scalar', 'global_scalar'}:
            # Gate weights the original time series features (T)
            gated_output = gate_value * ts_features + (1 - gate_value) * projected_fused
        # else case handled by __init__ check
        
        return gated_output, gate_value