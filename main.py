import csv
from tokenizer import tokenize
from vocab import build_vocab
from encoder import encode
from embeddings import create_embeddings, embed_tokens
from attention import create_qkv, compute_attention
from visualize import plot_attention

file_path = "data/spam.csv"
with open(file_path, encoding="latin-1") as f:
    reader = csv.reader(f)
    next(reader)
    tokenized_sentences = []
    for i, row in enumerate(reader):
        label = row[0]
        message = row[1]
        tokens = tokenize(message)
        tokenized_sentences.append(tokens)
        print(label, ":", tokens)
        if i == 100:
            break

    vocab = build_vocab(tokenized_sentences)
    print("Vocabulary size:", len(vocab))
    print(list(vocab.items())[:10])

    example_tokens = tokenized_sentences[0]
    example_ids = encode(example_tokens, vocab)
    print("\nExample tokens:", example_tokens)
    print("Encoded IDs:", example_ids)

    embedding_dim = 8
    embedding_matrix = create_embeddings(len(vocab), embedding_dim)
    print(embedding_matrix)
    embedded_tokens = embed_tokens(example_ids, embedding_matrix)
    print("\nEmbedding shape:", embedded_tokens.shape)
    print(embedded_tokens)

    Q, K, V = create_qkv(embedded_tokens)
    output, attention_weights = compute_attention(Q, K, V)
    print("\nAttention Weights Shape:", attention_weights.shape)
    print("Attention Weights:\n", attention_weights)
    print("\nRow sums:", attention_weights.sum(axis=1))
    print(output)

    plot_attention(attention_weights, example_tokens)