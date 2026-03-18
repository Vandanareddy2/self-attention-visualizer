# Self-Attention Visualizer (From Scratch)

This project is a step-by-step implementation of **Self-Attention**, the core building block of Transformers, built entirely from scratch using **Python and NumPy**.

The goal is to understand how attention works internally without using high-level libraries like HuggingFace or PyTorch.

---

## 🚀 Current Status

This project currently implements the pipeline up to:

Text → Tokens → Token IDs → Embeddings → Self-Attention → Visualization

⚠️ Note: This is a **learning-focused implementation**. The model is not trained.

---

## 📁 Project Structure

self-attention-visualizer/

data/  
&nbsp;&nbsp;&nbsp;&nbsp;spam.csv  

main.py  
tokenizer.py  
vocab.py  
encoder.py  
embeddings.py  
attention.py  
visualize.py  

---

## 🧠 Core Concept: Self-Attention

Self-attention allows each word in a sentence to look at every other word and decide how important they are.

Core formula:

Attention(Q, K, V) = softmax((QKᵀ) / √d) V

Where:

- Q (Query) → what a word is looking for  
- K (Key) → what a word contains  
- V (Value) → actual information of the word  

---

## 🔄 Pipeline (Implemented)

### 1. Tokenization

Convert sentence into tokens:

"Free entry now" → ['free', 'entry', 'now']

---

### 2. Vocabulary

Map words to integers:

{'free': 0, 'entry': 1, 'now': 2}

---

### 3. Encoding

Convert tokens → IDs:

['free','entry','now'] → [0,1,2]

---

### 4. Embeddings

Convert IDs → vectors:

Shape:
(sequence_length × embedding_dim)

Example:
5 words → (5 × 8)

---

### 5. Q, K, V Computation

Q = X @ Wq  
K = X @ Wk  
V = X @ Wv  

Each word is projected into three different representations.

---

### 6. Attention Scores

scores = Q @ Kᵀ

Shape:
(sequence_length × sequence_length)

Each value represents how much one word relates to another.

---

### 7. Scaling

scores / √d

Helps stabilize softmax.

---

### 8. Softmax

Converts scores into probabilities.

Each row sums to 1.

---

### 9. Final Output

output = attention_weights @ V

Each word becomes a weighted combination of all other words.

---

## 📊 Attention Visualization

The attention weights are visualized using a heatmap.

- Rows → Query words  
- Columns → Key words  
- Color intensity → attention strength  

This helps understand how words interact with each other.

---

## 🧪 Example

Input sentence:

"free entry now"

Attention matrix:

        free entry now
free     0.1  0.7  0.2
entry    0.3  0.4  0.3
now      0.5  0.2  0.3

Interpretation:

- "free" focuses more on "entry"
- "now" focuses more on "free"

---

## ⚠️ Important Note

- Weights are randomly initialized  
- No training is performed  
- Attention patterns are not linguistically meaningful  

This project focuses on understanding **how attention works**, not building a production model.

---

## 🛠️ How to Run

1. Install dependencies:

pip install numpy matplotlib

2. Run:

python main.py

---

## 💡 Key Learnings

- Transformers operate on matrices, not text  
- Attention is a combination of matrix multiplications  
- Softmax converts similarity into probability  
- Each word dynamically gathers context from others  

---

## 🚧 Work in Progress / Next Steps

Planned improvements:

- Add a classification layer (Spam vs Ham)
- Train a simple model instead of random weights
- Implement Multi-Head Attention
- Add Positional Encoding
- Use pretrained embeddings (GloVe / Word2Vec)
- Improve tokenizer (handle punctuation, special tokens)
- Add better visualization and examples

---

## 📌 Why This Project Matters

Understanding self-attention is fundamental for:

- Transformers (GPT, BERT)
- Large Language Models
- Modern NLP systems

This project builds that intuition from scratch.

---

## 👨‍💻 Author

Built as a learning project to understand transformers from first principles.