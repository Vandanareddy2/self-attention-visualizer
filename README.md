# Self-Attention Visualizer

A from-scratch implementation of the self-attention mechanism using Python and NumPy, designed to expose the internal computations that power transformer-based models.

This project reconstructs the attention pipeline end-to-end and provides a visual interpretation of how tokens interact within a sequence.

---

## Overview

Self-attention is a core component of modern NLP architectures such as GPT and BERT. It enables each token in a sequence to dynamically attend to all other tokens and build context-aware representations.

This implementation follows the complete processing pipeline:

Text → Tokenization → Vocabulary → Encoding → Embeddings → Self-Attention → Visualization

---

## Core Implementation

The project implements scaled dot-product attention:

Attention(Q, K, V) = softmax((QKᵀ) / √d) V

Where:

- Q (Query) captures what each token is attending to  
- K (Key) represents the features of each token  
- V (Value) contains the information being aggregated  

The resulting attention matrix has shape:

(sequence_length × sequence_length)

Each row defines how a token distributes its attention across the sequence.

---

## System Components

- Tokenization and vocabulary construction  
- Token-to-index encoding  
- Embedding matrix generation  
- Linear projections for Query, Key, and Value  
- Scaled dot-product attention computation  
- Row-wise softmax normalization  
- Attention-weighted aggregation  
- Heatmap-based visualization  

---

## Visualization

Attention weights are rendered as a heatmap:

- Rows correspond to query tokens  
- Columns correspond to key tokens  
- Color intensity reflects attention magnitude  

This provides a direct view into how contextual relationships are formed across tokens.

---

## Key Characteristics

- Fully implemented using NumPy (no deep learning frameworks)  
- Explicit matrix operations for transparency  
- Shape-consistent transformations across the pipeline  
- Deterministic execution for reproducibility  

---

## Observations

- Attention is directional:  
  attention(i → j) ≠ attention(j → i)

- Each token independently computes its distribution over all tokens

- With randomly initialized parameters, attention patterns reflect the mechanism rather than learned linguistic structure

---

## Project Structure

self-attention-visualizer/

data/  
  spam.csv  

main.py  
tokenizer.py  
vocab.py  
encoder.py  
embeddings.py  
attention.py  
visualize.py  

---

## Running the Project

Install dependencies:

pip install numpy matplotlib

Run:

python main.py

---

## Extensions

- Multi-head attention  
- Positional encoding  
- Integration with trained embedding spaces  
- End-to-end transformer architecture  

---

## Applications

Understanding the mechanics of self-attention is fundamental for:

- Transformer model design  
- Model interpretability  
- Sequence modeling systems  
- Advanced NLP workflows  

---

## Author

Self-attention implementation and visualization built with a focus on clarity, correctness, and internal transparency.