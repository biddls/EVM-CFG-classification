from timeit import timeit
import tokeniser
from vectorEncoding.LSTM_Autoenc import LSTM_AutoEnc_Training
from vectorEncoding.TF_IDF import TF_IDF
from vectorEncoding.averagingVectors import Average
from tqdm import tqdm
from collections import Counter
from enum import Enum

if __name__ == "__main__":
    # todo: keep the data in the index based status until its about to be loaded onto the GPU
    # todo: mby cache it in that state also...
    loader = tokeniser.CFG_Loader(exclusionList="./src/vectorEncoding/cache/conts/*.txt")
    count = 0
    cfgs = list()
    
    # set to 0 to do all the data
    max_cfgs = 0
    counts: Counter[tuple[int | tuple[int, int]]] = Counter()
    countToggle = Enum("enableCounting", {"Counting": True, "notCounting": False}).Counting
    loader = tqdm(loader)

    for cfg in loader:
        # print(cfg)
        try:
            cfgs.append(cfg.addr)
            tokens = tokeniser.Tokeniser.preProcessing(cfg)
            temp_counts = tokeniser.Tokeniser.tokenise(tokens, counting=countToggle.value)

            if countToggle == countToggle.Counting:
                # adds temp_counts to counts adding the values together
                counts.update(temp_counts)
            else:
                cfg.addTokens(tokens)

            count += 1
            if count == max_cfgs:
                break

        except KeyboardInterrupt:
            break

    # LSTM autoencoder
    trainer = LSTM_AutoEnc_Training(counts, 150, count)
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