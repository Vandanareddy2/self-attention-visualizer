import numpy as np

def create_embeddings(vocab_size, embedding_dim=8):
    return np.random.rand(vocab_size, embedding_dim)

def embed_tokens(token_ids, embedding_matrix):
    vectors = []

    for idx in token_ids:
        vectors.append(embedding_matrix[idx])

    return np.array(vectors)