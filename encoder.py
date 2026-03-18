def encode(tokens, vocab):
    ids = []

    for token in tokens:
        ids.append(vocab[token])

    return ids