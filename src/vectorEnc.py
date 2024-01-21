import tokeniser


if __name__ == "__main__":
    loader = tokeniser.CFG_Loader()
    data = list()
    for cfg in loader:
        tokens = tokeniser.Tokeniser.preProcessing(cfg)
        vectors = tokeniser.Tokeniser.tokenise(tokens)

        for vector in vectors:
            data.extend(vector)



"""
Things to try:
Doc2Vec
LSTM Autoencoder
TF-IDF
Attention
Simple average
"""