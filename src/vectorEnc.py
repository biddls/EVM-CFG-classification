import tokeniser
from vectorEncoding.LSTM_Autoenc import LSTM_AutoEnc_Training
from tqdm import tqdm

if __name__ == "__main__":
    loader = tokeniser.CFG_Loader()
    data = list()
    count = 0
    for cfg in tqdm(loader):
        tokens = tokeniser.Tokeniser.preProcessing(cfg)
        vectors = tokeniser.Tokeniser.tokenise(tokens)

        data.extend(vectors)
        count += 1
        if count == 10:
            break

    trainer = LSTM_AutoEnc_Training(data, 100)
    trainer.trainEnc(10)
    out = trainer.model.getVectors(data)



"""
Things to try:
Doc2Vec
LSTM Autoencoder
TF-IDF
Attention
Simple average
"""