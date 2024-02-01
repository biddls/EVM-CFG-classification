import tokeniser
from vectorEncoding.LSTM_Autoenc import LSTM_AutoEnc_Training
from vectorEncoding.TF_IDF import TF_IDF
from vectorEncoding.averagingVectors import Average
from tqdm import tqdm

if __name__ == "__main__":
    # todo: keep the data in the index based status until its about to be loaded onto the GPU
    # todo: mby cache it in that state also...
    loader = tokeniser.CFG_Loader()
    data = list()
    count = 0
    cfgs = list()
    CFGs = 50
    for cfg in tqdm(loader):
        try:
            cfgs.append(cfg.addr)
            tokens = tokeniser.Tokeniser.preProcessing(cfg)
            vectors, counts = tokeniser.Tokeniser.tokenise(tokens)

            data.extend(vectors)
            count += 1
            if count == CFGs:
                break

        except KeyboardInterrupt:
            break

    # LSTM autoencoder
    trainer = LSTM_AutoEnc_Training(data, 150, count)
    trainer.trainEnc(200)

    # TF-IDF
    # tfidf = TF_IDF(data)
    # tfIdfVectors = tfidf()
    
    # Average
    # _average = Average(data)
    # averageVectors = _average()



"""
Things to try:
Doc2Vec
LSTM Autoencoder
TF-IDF
Attention
Simple average
"""