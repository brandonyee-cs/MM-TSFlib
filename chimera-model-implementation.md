# Chimera: Token-Level Multimodal Fusion Model Implementation Plan

This document outlines the implementation plan for the Chimera model, a new multimodal architecture that performs token-level fusion between time series and text data using cross-attention mechanisms.

## 1. Architecture Overview

```
                               ┌───────────────┐
                               │  Text Input   │
                               └───────┬───────┘
                                       │
                                       ▼
                               ┌───────────────┐
                               │    LLM Model  │
                               │    (Frozen)   │
                               └───────┬───────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │ Self-Attention │
                              │     Layers     │◄────── --text_fusion_layers
                              └────────┬───────┘
                                       │
                                       ▼
             ┌───────────────┐  ┌──────────────────┐  ┌───────────────┐
             │  Time Series  │  │  Cross-Attention │  │      Keys     │
             │     Input     │─►│   Multihead      │◄─┤     Values    │
             └───────┬───────┘  │ Latent Attention │  └───────────────┘
                     │          └────────┬─────────┘           ▲
                     │                   │                     │
                     ▼                   ▼                     │
            ┌────────────────┐   ┌───────────────┐             │
            │  iTransformer  │   │ Self-Attention│◄───── --post_fusion_layers
            │     Model      │   │     Layers    │             │
            └────────┬───────┘   └───────┬───────┘             │
                     │                   │                     │
                     │                   ▼                     │
                     │           ┌───────────────┐             │
                     │           │  Gating Mech  │             │
                     │           └───────┬───────┘             │
                     │                   │                     │
                     └─────────┐         │         ┌───────────┘
                               ▼         ▼         │
                         ┌─────────────────────────┘
                         │     Feature Fusion      │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Final Self-Attention │◄───── --final_layers
                         │        Layers         │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Prediction Head    │
                         └───────────────────────┘
```

## 2. New Module Structure

### 2.1 New Files to Create

1. `models/ChimeraTransformer.py`: Main model implementation
2. `layers/CrossAttention.py`: Cross-attention and multihead latent attention mechanisms
3. `layers/GatingMechanism.py`: Gating mechanism for controlling fusion

## 3. Detailed Implementation Plan

### 3.1 `models/ChimeraTransformer.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.CrossAttention import MultiheadLatentAttention
from layers.GatingMechanism import FeatureGate
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # iTransformer components (time series leg)
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)
        # Encoder for time series
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                     output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Text fusion components
        self.text_fusion_layers = configs.text_fusion_layers
        if self.text_fusion_layers > 0:
            self.text_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                         output_attention=configs.output_attention), configs.d_llm, configs.n_heads),
                        configs.d_llm,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.text_fusion_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_llm)
            )
        
        # Cross-attention for modality fusion
        self.latent_dim = configs.latent_dim if hasattr(configs, 'latent_dim') else min(configs.d_model, configs.d_llm)
        self.cross_attention = MultiheadLatentAttention(
            query_dim=configs.d_llm,
            key_dim=configs.d_model,
            latent_dim=self.latent_dim,
            num_heads=configs.fusion_heads if hasattr(configs, 'fusion_heads') else configs.n_heads,
            dropout=configs.dropout
        )
        
        # Post-fusion self-attention layers
        self.post_fusion_layers = configs.post_fusion_layers
        if self.post_fusion_layers > 0:
            self.post_fusion_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                         output_attention=configs.output_attention), self.latent_dim, configs.n_heads),
                        self.latent_dim,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.post_fusion_layers)
                ],
                norm_layer=torch.nn.LayerNorm(self.latent_dim)
            )
        
        # Gating mechanism for controlling fusion
        self.feature_gate = FeatureGate(
            fused_dim=self.latent_dim,
            ts_dim=configs.d_model,
            hidden_dim=configs.gate_hidden_dim if hasattr(configs, 'gate_hidden_dim') else 2*configs.d_model
        )
        
        # Final transformer layers after fusion
        self.final_layers = configs.final_layers
        if self.final_layers > 0:
            self.final_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                         output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.final_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )
        
        # Task-specific projection heads
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
    
    def forward_text_encoder(self, text_embeddings):
        """Process text embeddings through self-attention layers"""
        if self.text_fusion_layers > 0:
            text_features, _ = self.text_encoder(text_embeddings, attn_mask=None)
        else:
            text_features = text_embeddings
        return text_features
    
    def forward_fusion(self, text_features, ts_features):
        """Perform cross-modal fusion with cross-attention"""
        # Cross-attention fusion
        fused_features = self.cross_attention(
            queries=text_features,  # Text as query
            keys=ts_features,       # Time series as key
            values=ts_features      # Time series as value
        )
        
        # Post-fusion self-attention if specified
        if self.post_fusion_layers > 0:
            fused_features, _ = self.post_fusion_encoder(fused_features, attn_mask=None)
            
        return fused_features
    
    def forward_gating_fusion(self, fused_features, ts_features):
        """Apply gating mechanism to control fusion strength"""
        return self.feature_gate(fused_features, ts_features)
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Time series embedding and encoding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        ts_features, _ = self.encoder(enc_out, attn_mask=None)
        
        # Process text if available for multimodal fusion
        if text_embeddings is not None:
            # Text processing
            text_features = self.forward_text_encoder(text_embeddings)
            
            # Cross-modal fusion
            fused_features = self.forward_fusion(text_features, ts_features)
            
            # Gated fusion with time series features
            final_features = self.forward_gating_fusion(fused_features, ts_features)
            
            # Final transformer layers if specified
            if self.final_layers > 0:
                final_features, _ = self.final_encoder(final_features, attn_mask=None)
        else:
            final_features = ts_features
        
        # Task-specific prediction
        dec_out = self.projection(final_features).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    # Implement other task methods (imputation, anomaly_detection, classification) similarly
    # by adapting the existing iTransformer implementations with the fusion components

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_embeddings)
            return dec_out  # [B, L, D]
        
        # Implement other task forward methods as needed
```

### 3.2 `layers/CrossAttention.py`

```python
import torch
import torch.nn as nn
import math

class MultiheadLatentAttention(nn.Module):
    """
    Multihead Latent Cross-Attention mechanism for token-level fusion.
    
    This module projects query and key/value inputs into a shared latent space
    where cross-attention is performed, enabling effective fusion of modalities
    with different dimensions.
    """
    def __init__(self, query_dim, key_dim, latent_dim, num_heads=8, dropout=0.1):
        super(MultiheadLatentAttention, self).__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = latent_dim // num_heads
        assert self.head_dim * num_heads == latent_dim, "latent_dim must be divisible by num_heads"
        
        # Projection layers to latent space
        self.q_proj = nn.Linear(query_dim, latent_dim)
        self.k_proj = nn.Linear(key_dim, latent_dim)
        self.v_proj = nn.Linear(key_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, queries, keys, values, attention_mask=None):
        batch_size = queries.shape[0]
        
        # Project to latent space
        q = self.q_proj(queries)  # (batch_size, seq_len_q, latent_dim)
        k = self.k_proj(keys)     # (batch_size, seq_len_k, latent_dim)
        v = self.v_proj(values)   # (batch_size, seq_len_v, latent_dim)
        
        # Reshape for multihead attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_q, head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_k, head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_v, head_dim)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq_q, seq_k)
        
        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (batch, heads, seq_q, head_dim)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output
```

### 3.3 `layers/GatingMechanism.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureGate(nn.Module):
    """
    Gating mechanism to control the fusion of multimodal features.
    
    This module learns to selectively combine information from the fused
    representation and the original time series features, allowing the model
    to ignore irrelevant text information when needed.
    """
    def __init__(self, fused_dim, ts_dim, hidden_dim=None):
        super(FeatureGate, self).__init__()
        if hidden_dim is None:
            hidden_dim = 2 * ts_dim
            
        # Project fused features to time series dimension for fusion
        self.fused_projection = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ts_dim)
        )
        
        # Gate parameters
        self.gate_network = nn.Sequential(
            nn.Linear(fused_dim + ts_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ts_dim),
            nn.Sigmoid()
        )
        
    def forward(self, fused_features, ts_features):
        """
        Apply gating mechanism to control information flow
        
        Args:
            fused_features: Features from the cross-attention fusion (B, L, fused_dim)
            ts_features: Original time series features (B, L, ts_dim)
            
        Returns:
            Combined features with learned gating (B, L, ts_dim)
        """
        # Project fused features to time series dimension
        projected_fused = self.fused_projection(fused_features)
        
        # Compute gate values based on concatenated features
        concat_features = torch.cat([projected_fused, ts_features], dim=-1)
        gate = self.gate_network(concat_features)
        
        # Apply gate
        gated_output = gate * projected_fused + (1 - gate) * ts_features
        
        return gated_output
```

## 4. Changes to `exp/exp_long_term_forecasting.py`

The existing experiment class needs to be modified to support the Chimera model. Here are the key modifications needed:

```python
class Exp_Long_Term_Forecast(Exp_Basic):
    # Existing code...
    
    def train(self, setting):
        # Existing code...
        
        # Inside the training loop, modify the forward pass to include text embeddings
        for epoch in range(self.args.train_epochs):
            # ... existing code ...
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                # ... existing code ...
                
                # Get text embeddings if available
                text_embeddings = None
                if self.args.model == 'ChimeraTransformer' and self.text_path != 'None':
                    # Code to get text embeddings from LLM
                    # This should reuse existing code for text embedding generation
                    
                # Forward pass with text embeddings
                if self.args.model == 'ChimeraTransformer':
                    outputs = self.model(batch_x, batch_x_mark, batch_y[:, :self.args.label_len, :], 
                                        batch_y_mark[:, :self.args.label_len, :], text_embeddings)
                else:
                    # Existing forward pass code
                
                # ... rest of training loop ...
        
        # ... rest of method ...
    
    def test(self, setting, test=0):
        # Similar modifications to forward pass in the test method
        # ... existing code ...
```

## 5. Run.py Modifications

Add new command-line arguments to `run.py`:

```python
# Chimera model parameters
parser.add_argument('--text_fusion_layers', type=int, default=2, 
                    help='number of self-attention layers to process text embeddings')
parser.add_argument('--post_fusion_layers', type=int, default=2, 
                    help='number of self-attention layers after cross-attention fusion')
parser.add_argument('--final_layers', type=int, default=1, 
                    help='number of final self-attention layers after gated fusion')
parser.add_argument('--fusion_heads', type=int, default=8, 
                    help='number of attention heads in cross-modal fusion')
parser.add_argument('--latent_dim', type=int, default=256, 
                    help='dimension of shared latent space for fusion')
parser.add_argument('--gate_hidden_dim', type=int, default=512, 
                    help='hidden dimension for feature gating network')
```

## 6. Model Registry Update

Update the model import in `exp/exp_long_term_forecasting.py`:

```python
from models import Autoformer, Transformer, TimesNet, DLinear, FEDformer, Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, FiLM, iTransformer, FreTS, ChimeraTransformer
```

Add the new model to the model dictionary:

```python
model_dict = {
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'TimesNet': TimesNet,
    # ... other models ...
    'iTransformer': iTransformer,
    'ChimeraTransformer': ChimeraTransformer,  # Add the new model
}
```

## 7. Example Usage

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model ChimeraTransformer \
    --data ETTh1 \
    --root_path ./dataset/ETT/ \
    --data_path ETTh1.csv \
    --text_path text_data.csv \
    --freq h \
    --seq_len 96 \
    --pred_len 24 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_ff 2048 \
    --text_fusion_layers 2 \
    --post_fusion_layers 2 \
    --final_layers 1 \
    --latent_dim 256 \
    --fusion_heads 8 \
    --batch_size 32 \
    --train_epochs 10 \
    --llm_model BERT \
    --pool_type attention
```

## 8. Implementation Notes

### 8.1 Intuition

The Chimera model significantly enhances the basic multimodal fusion approach by:

1. **Token-level fusion**: Instead of just combining predictions, it fuses features at the token level through cross-attention, allowing the model to learn which time series tokens should attend to which text tokens.

2. **Adaptive gating**: The gating mechanism allows the model to dynamically decide how much to rely on the fused information versus the original time series information, providing both interpretability and performance benefits.

3. **Hierarchical processing**: By using multiple stages of self-attention before and after fusion, the model can better capture complex relationships both within and across modalities.

### 8.2 Design Considerations

1. **Computational Efficiency**: The cross-attention mechanism can be computationally expensive, especially when the sequence lengths are large. Consider using efficient attention variants for long sequences.

2. **Memory Usage**: The model introduces several new components that increase memory usage. For large models, gradient checkpointing may be necessary.

3. **Flexibility**: The architecture is designed to be flexible, with the number of layers at each stage configurable through command-line arguments. Setting these values to 0 effectively disables those components.

### 8.3 Training Strategy

1. Start with pre-training the iTransformer part of the model on time series data only.
2. Then freeze the iTransformer weights and train only the fusion components.
3. Finally, fine-tune the entire model end-to-end with a lower learning rate.

This staged approach can help stabilize training and prevent the fusion components from dominating before they've learned meaningful representations.

### 8.4 Potential Extensions

1. **Bidirectional Cross-Attention**: Currently, queries come from text and keys/values from time series. A bidirectional approach could allow information to flow both ways.

2. **Hierarchical Fusion**: For complex tasks, multiple levels of fusion at different layers of the model could be beneficial.

3. **Regularization Techniques**: Special regularization for the gating mechanism could encourage it to be more selective, improving interpretability.

## 9. Conclusion

The Chimera model represents a sophisticated approach to multimodal fusion, leveraging the strengths of both transformer-based time series models and text embeddings from LLMs. By fusing at the token level and using gating mechanisms, it has the potential to outperform simpler fusion approaches, especially in scenarios where the relationship between text and time series data is complex and context-dependent. 