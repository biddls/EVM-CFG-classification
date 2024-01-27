import tokeniser
from vectorEncoding.LSTM_Autoenc import LSTM_AutoEnc_Training
from vectorEncoding.TF_IDF import TF_IDF
from vectorEncoding.averagingVectors import Average
from tqdm import tqdm

if __name__ == "__main__":
    loader = tokeniser.CFG_Loader()
    data = list()
    count = 0
    cfgs = list()
    for cfg in tqdm(loader):
        cfgs.append(cfg.addr)
        tokens = tokeniser.Tokeniser.preProcessing(cfg)
        try:
            vectors = tokeniser.Tokeniser.tokenise(tokens)
        except KeyError as e:
            print(cfg.addr)
            raise e

        data.extend(vectors)
        count += 1
        if count == 10:
            break
    print(len(cfgs))
    print(len(data))
    # LSTM autoencoder
    # trainer = LSTM_AutoEnc_Training(data, 100)
    # trainer.trainEnc(10)
    # out = trainer.model.getVectors(data)

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