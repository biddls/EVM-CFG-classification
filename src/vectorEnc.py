import torch
import tokeniser


if __name__ == "__main__":
    loader = tokeniser.CFG_Loader()
    for cfg in loader:
        tokens = tokeniser.Tokeniser.preProcessing(cfg)
        vectors = tokeniser.Tokeniser.tokenise(tokens)
        
"""
Things to try:
Doc2Vec
LSTM Autoencoder
TF-IDF
Attention
Simple average
"""