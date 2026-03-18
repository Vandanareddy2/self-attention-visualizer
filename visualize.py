import matplotlib.pyplot as plt
import numpy as np

def plot_attention(attention_weights, tokens):
    fig, ax = plt.subplots()

    cax = ax.matshow(attention_weights, cmap='viridis')

    plt.colorbar(cax)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))

    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)

    plt.xlabel("Keys (Words)")
    plt.ylabel("Queries (Words)")

    plt.title("Self-Attention Heatmap")

    plt.show()