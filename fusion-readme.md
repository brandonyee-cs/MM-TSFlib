# Multimodal Fusion in MM-TSFlib

This document provides a detailed explanation of how MM-TSFlib performs multimodal fusion between time series forecasting models and language models. Unlike many multimodal architectures that use deep integrated fusion, MM-TSFlib employs a late fusion approach at the prediction level, allowing for modular design and easier benchmarking across different foundation models.

## 1. Overview

MM-TSFlib's multimodal fusion pipeline consists of these key steps:

1. Time series data is processed through a time series forecasting model
2. Text data is processed through a language model
3. Text embeddings are pooled to create a fixed-length representation
4. The predictions from both modalities are combined using a weighted sum

The approach is intentionally simple to serve as a baseline for multimodal time series forecasting.

## 2. Mathematical Details

### 2.1 Independent Modality Processing

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

### 2.2 Text Embedding Pooling

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

### 2.3 Final Fusion

The final prediction is a weighted combination of the time series prediction and the text-derived prediction:

$$\hat{Y} = (1 - w) \cdot \hat{Y}_{ts} + w \cdot (E_{norm} + Y_{prior})$$

where:
- $w$ is the `prompt_weight` parameter (default: 0.01)
- $Y_{prior}$ is a prior estimate based on historical data

## 3. Implementation Details

### 3.1 Code Location

The fusion mechanism is primarily implemented in the `exp_long_term_forecasting.py` file within the `Exp_Long_Term_Forecast` class. Key sections include:

- **Initialization**: Lines ~50-75 define the fusion parameters and MLP components
- **Text Processing**: The core text processing logic is duplicated in the `train`, `vali`, and `test` methods
- **Pooling Implementation**: Different pooling methods are implemented around lines ~550-580 in the training loop

### 3.2 Training Process

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

3. **Time Series Processing**: The standard time series model generates a prediction
   ```python
   outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
   ```

4. **Pooling**: Text embeddings are pooled according to the specified method
   ```python
   # Example for attention pooling
   outputs_norm = F.normalize(outputs_reshaped, p=2, dim=1)
   prompt_emb_norm = F.normalize(prompt_emb, p=2, dim=2)
   attention_scores = torch.bmm(prompt_emb_norm, outputs_norm)
   attention_weights = F.softmax(attention_scores, dim=1)
   weighted_prompt_emb = torch.sum(prompt_emb * attention_weights, dim=1)
   prompt_emb = weighted_prompt_emb.unsqueeze(-1)
   ```

5. **Fusion**: The predictions are combined
   ```python
   prompt_y = norm(prompt_emb) + prior_y
   outputs = (1-self.prompt_weight)*outputs + self.prompt_weight*prompt_y
   ```

### 3.3 Configuration Parameters

The fusion mechanism can be configured through several parameters in `run.py`:

- `--llm_model`: Type of language model (GPT2, LLAMA2, LLAMA3, BERT, etc.)
- `--pool_type`: Pooling strategy for token embeddings (avg, min, max, attention)
- `--prompt_weight`: Weight for the text-based predictions (default: 0.01)
- `--use_fullmodel`: Whether to use the full LLM or just embeddings (0=embeddings only, 1=full model)

## 4. Example Usage

To run an experiment with specific fusion settings, use the script:

```bash
bash ./scripts/week_health.sh 0 1 0 --llm_model GPT2 --pool_type attention --prompt_weight 0.05
```

This will:
1. Run on the health dataset using GPU 0
2. Use GPT2 as the language model
3. Apply attention-based pooling for text embeddings
4. Apply a 0.05 weight to text predictions in the final fusion

## 5. Limitations and Future Improvements

The current fusion approach has several limitations:

1. The fixed `prompt_weight` doesn't adapt based on text quality or relevance
2. Text processing is independent of the time series context
3. No fine-tuning of the language models for time series forecasting

Potential improvements include:
- Learning adaptive fusion weights based on data quality
- Designing tailored cross-modal attention mechanisms
- Implementing true multimodal encoders that jointly process both modalities
- Fine-tuning language models specifically for forecasting tasks

As noted in the README, this library represents a first step toward multimodal time series forecasting and is intended as a foundation for more sophisticated approaches. 