# Multimodal Fusion in MM-TSFlib

This document provides a detailed explanation of how MM-TSFlib performs multimodal fusion between time series forecasting models and language models. Unlike many multimodal architectures that use deep integrated fusion, MM-TSFlib employs a late fusion approach at the prediction level, allowing for modular design and easier benchmarking across different foundation models.

## 1. Overview

MM-TSFlib's multimodal fusion pipeline consists of these key steps:

1. Time series data is processed through a time series forecasting model
2. Text data is processed through a language model
3. Text embeddings are pooled to create a fixed-length representation
4. The predictions from both modalities are combined using a weighted sum

The approach is intentionally simple to serve as a baseline for multimodal time series forecasting.

## 2. Supported Language Models

MM-TSFlib supports multiple Language Learning Models (LLMs), which are configurable via the `--llm_model` parameter:

### 2.1 Available LLMs

| Model | Identifier | Embedding Dimension | Description |
|-------|-----------|-------------------|-------------|
| **GPT2 variants** | `GPT2`, `GPT2M`, `GPT2L`, `GPT2XL` | 768, 1024, 1280, 1600 | OpenAI GPT-2 in various sizes |
| **LLaMA variants** | `LLAMA2`, `LLAMA3` | 4096, 4096 | Meta's LLaMA 7B and LLaMA-3 8B-Instruct |
| **BERT** | `BERT` | 768 | Google's BERT-base-uncased |
| **Doc2Vec** | `Doc2Vec` | 64 (configurable) | Traditional document embedding model |
| **ClosedLLM** | `ClosedLLM` | 768 | Uses BERT for encoding with different configuration |

### 2.2 Model Loading and Configuration

Each LLM is loaded from HuggingFace with specific configurations:

```python
# Example: GPT2 initialization
self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
self.gpt2_config.num_hidden_layers = configs.llm_layers
self.gpt2_config.output_attentions = True
self.gpt2_config.output_hidden_states = True

self.llm_model = GPT2Model.from_pretrained(
    'openai-community/gpt2',
    config=self.gpt2_config,
)
self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
```

**Key Configuration Parameters:**
- `llm_layers`: Number of transformer layers to use (default: 6)
- `use_fullmodel`: Whether to use full LLM or just embeddings (0=embeddings only, 1=full model)
- LLM parameters are frozen during training: `param.requires_grad = False`

## 3. Mathematical Details

### 3.1 Independent Modality Processing

Given input time series data $X_{ts} \in \mathbb{R}^{B \times T \times D}$ (batch size × time steps × features) and associated text data $X_{text}$ as a collection of strings:

1. **Time Series Processing**:
   
   The time series model $f_{ts}$ produces a forecast:
   
   $$\hat{Y}_{ts} = f_{ts}(X_{ts}) \in \mathbb{R}^{B \times P \times D}$$
   
   where $P$ is the prediction length.

2. **Text Processing**:
   
   Each text string is tokenized and processed by a language model $f_{llm}$ to obtain token embeddings:
   
   $$E_{text} = f_{llm}(X_{text}) \in \mathbb{R}^{B \times L \times D_{llm}}$$
   
   where $L$ is the sequence length and $D_{llm}$ is the LLM embedding dimension.
   
   These embeddings are then processed through an MLP to match the expected dimensionality:
   
   $$E_{proj} = \text{MLP}(E_{text}) \in \mathbb{R}^{B \times L \times D_{text}}$$

### 3.2 Text Embedding Pooling

Depending on the `pool_type` parameter, token-level embeddings are pooled into a single vector:

1. **Average Pooling**:
   
   $$E_{pooled} = \frac{1}{L} \sum_{i=1}^{L} E_{proj}[:, i, :] \in \mathbb{R}^{B \times D_{text}}$$

2. **Max Pooling**:
   
   $$E_{pooled} = \max_{i \in \{1,\ldots,L\}} E_{proj}[:, i, :] \in \mathbb{R}^{B \times D_{text}}$$

3. **Min Pooling**:
   
   $$E_{pooled} = \min_{i \in \{1,\ldots,L\}} E_{proj}[:, i, :] \in \mathbb{R}^{B \times D_{text}}$$

4. **Attention-based Pooling**:
   
   The time series output is used to compute attention weights:
   
   $$\text{Attention Scores} = (E_{proj})^T \cdot \hat{Y}_{ts} \in \mathbb{R}^{B \times L \times P}$$
   
   $$\text{Attention Weights} = \text{softmax}(\text{Attention Scores}) \in \mathbb{R}^{B \times L \times P}$$
   
   $$E_{pooled} = \sum_{i=1}^{L} E_{proj}[:, i, :] \cdot \text{Attention Weights}[:, i] \in \mathbb{R}^{B \times D_{text}}$$

The pooled embedding is then normalized:

$$E_{norm} = \frac{E_{pooled} - \mu(E_{pooled})}{\sqrt{\sigma^2(E_{pooled}) + \epsilon}}$$

where $\mu$ and $\sigma^2$ are the mean and variance operations respectively.

### 3.3 Final Fusion

The final prediction is a weighted combination of the time series prediction and the text-derived prediction:

$$\hat{Y} = (1 - w) \cdot \hat{Y}_{ts} + w \cdot (E_{norm} + Y_{prior})$$

where:
- $w$ is the `prompt_weight` parameter (default: 0.01)
- $Y_{prior}$ is a prior estimate based on historical data

## 4. Implementation Details

### 4.1 Code Location

The fusion mechanism is primarily implemented in the `exp_long_term_forecasting.py` file within the `Exp_Long_Term_Forecast` class. Key sections include:

- **Initialization**: Lines ~50-75 define the fusion parameters and MLP components
- **Text Processing**: The core text processing logic is duplicated in the `train`, `vali`, and `test` methods
- **Pooling Implementation**: Different pooling methods are implemented around lines ~550-580 in the training loop

### 4.2 MLP Architecture for LLM-to-Time Series Mapping

The library uses **two MLP networks** to map LLM outputs to time series predictions:

#### 4.2.1 MLP Class Definition

```python
class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)  
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  
                x = F.relu(x)
                x = self.dropout(x)  
        return x
```

#### 4.2.2 Two-Stage Mapping Process

**First MLP (`self.mlp`)** - Dimensionality reduction:
```python
# Maps from LLM dimension to text embedding dimension
mlp_sizes = [self.d_llm, int(self.d_llm/8), self.text_embedding_dim]
self.mlp = MLP(mlp_sizes, dropout_rate=0.3)
```

**Second MLP (`self.mlp_proj`)** - Projection to prediction length:
```python
# Projects to final prediction dimension
mlp_sizes2 = [self.text_embedding_dim + self.args.pred_len, self.args.pred_len]
self.mlp_proj = MLP(mlp_sizes2, dropout_rate=0.3)
```

#### 4.2.3 Architecture Flow

The mapping process follows this sequence:

1. **LLM Embedding**: $D_{llm}$ dimensions (e.g., 768 for GPT2, 4096 for LLaMA)
2. **First MLP**: $D_{llm} \rightarrow D_{llm}/8 \rightarrow \text{pred\_len}$
3. **Pooling**: Reduces sequence length to single vector
4. **Normalization**: Applies mean-variance normalization
5. **Fusion**: Weighted combination with time series predictions

### 4.3 Training Process

During training, the fusion process follows these steps:

1. **Data Loading**: Time series data and corresponding text are loaded in batches
   ```python
   batch_text = train_data.get_text(index)
   ```

2. **Text Embedding**: Text is tokenized and embedded through the language model
   ```python
   prompt = [f"<|start_prompt|Make predictions about the future based on the following information: {text_info}<|<end_prompt>|>" for text_info in batch_text]
   prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids
   prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(self.device))
   ```

3. **MLP Processing**: Embeddings are processed through the first MLP
   ```python
   if self.use_fullmodel:
       prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
   else:
       prompt_emb = prompt_embeddings 
   prompt_emb = self.mlp(prompt_emb)
   ```

4. **Time Series Processing**: The standard time series model generates a prediction
   ```python
   outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
   ```

5. **Pooling**: Text embeddings are pooled according to the specified method
   ```python
   # Example for attention pooling
   if self.pool_type == "attention":
       outputs_reshaped = outputs
       outputs_norm = F.normalize(outputs_reshaped, p=2, dim=1)
       prompt_emb_norm = F.normalize(prompt_emb, p=2, dim=2)
       attention_scores = torch.bmm(prompt_emb_norm, outputs_norm)
       attention_weights = F.softmax(attention_scores, dim=1)
       weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)
       prompt_emb = weighted_prompt_emb.unsqueeze(-1)
   ```

6. **Fusion**: The predictions are combined
   ```python
   prompt_y = norm(prompt_emb) + prior_y
   outputs = (1-self.prompt_weight)*outputs + self.prompt_weight*prompt_y
   ```

### 4.4 Configuration Parameters

The fusion mechanism can be configured through several parameters in `run.py`:

- `--llm_model`: Type of language model (GPT2, LLAMA2, LLAMA3, BERT, etc.)
- `--llm_dim`: LLM embedding dimension (automatically set based on model choice)
- `--llm_layers`: Number of transformer layers to use (default: 6)
- `--pool_type`: Pooling strategy for token embeddings (avg, min, max, attention)
- `--prompt_weight`: Weight for the text-based predictions (default: 0.01)
- `--use_fullmodel`: Whether to use the full LLM or just embeddings (0=embeddings only, 1=full model)
- `--text_emb`: Text embedding dimension (set to pred_len by default)

### 4.5 Optimizer Configuration

The library uses separate optimizers for different components:

```python
# Time series model optimizer
model_optim = self._select_optimizer()

# MLP optimizer (for first MLP)
model_optim_mlp = self._select_optimizer_mlp()

# Projection MLP optimizer (for second MLP) 
model_optim_proj = self._select_optimizer_proj()
```

This allows for independent learning rates:
- `--learning_rate`: Main model learning rate (default: 0.0001)
- `--learning_rate2`: MLP learning rate (default: 0.01)
- `--learning_rate3`: Projection MLP learning rate (default: 0.001)

## 5. Example Usage

To run an experiment with specific fusion settings, use the script:

```bash
bash ./scripts/week_health.sh 0 1 0 --llm_model GPT2 --pool_type attention --prompt_weight 0.05
```

This will:
1. Run on the health dataset using GPU 0
2. Use GPT2 as the language model
3. Apply attention-based pooling for text embeddings
4. Apply a 0.05 weight to text predictions in the final fusion

## 6. Limitations and Future Improvements

The current fusion approach has several limitations:

1. The fixed `prompt_weight` doesn't adapt based on text quality or relevance
2. Text processing is independent of the time series context
3. No fine-tuning of the language models for time series forecasting
4. The two-stage MLP approach may introduce information bottlenecks

Potential improvements include:
- Learning adaptive fusion weights based on data quality
- Designing tailored cross-modal attention mechanisms
- Implementing true multimodal encoders that jointly process both modalities
- Fine-tuning language models specifically for forecasting tasks
- End-to-end training of the entire pipeline including LLM parameters

As noted in the README, this library represents a first step toward multimodal time series forecasting and is intended as a foundation for more sophisticated approaches. 