def build_vocab(tokenized_sentences):
    vocab = {}
    index = 0

    for tokens in tokenized_sentences:
        for token in tokens:
            if token not in vocab:
                vocab[token] = index
                index += 1

    return vocab