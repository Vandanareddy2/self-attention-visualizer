import numpy as np

# Step 1: Create Q, K, V
def create_qkv(X):
    d_model = X.shape[1]

    np.random.seed(42)  # for consistent results

    Wq = np.random.rand(d_model, d_model)
    Wk = np.random.rand(d_model, d_model)
    Wv = np.random.rand(d_model, d_model)

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    return Q, K, V


# Step 2: Softmax (row-wise)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Step 3: Scaled Dot-Product Attention
def compute_attention(Q, K, V):
    d_k = Q.shape[1]

    # QKᵀ → similarity scores
    scores = Q @ K.T

    # scale
    scaled_scores = scores / np.sqrt(d_k)

    # convert to probabilities
    weights = softmax(scaled_scores)

    # weighted sum of values
    output = weights @ V

    return output, weights