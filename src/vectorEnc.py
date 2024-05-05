import tokeniser
from vectorEncoding.LSTM_Autoenc import LSTM_AutoEnc_Training
from vectorEncoding.TF_IDF import TF_IDF
from vectorEncoding.averagingVectors import Average
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from CFG_reader import CFG_Reader
from graphComp.graphLoading import graphLoader
from graphComp.graphCompression import graphCompression
from graphComp.graphLabeling import graphLabeling
from icecream import ic
from vectorEncoding.tailRemoval import shrinkCounts


def main(
    graphName: str,
    lstmWidth: int = 172,
    unlabelled_CFGs: int = 0
) -> None:
    # todo: add in the culling of less frequent tokens
    cfgs: list[CFG_Reader] = list()

    counts: Counter[tuple[int | tuple[int, int]]] = Counter()

    # get the list of the CFG addrs which have labels
    gc = graphLoader(
        CFGs=cfgs,
        pathToTags="./addressTags.csv",
        pathToLabels="./labels.json",
        _tf_idf=None,
        _average=None,
        _lstm=None
    )
    labels = gc.addrLabels
    del gc

    loader = tokeniser.CFG_Loader(exclusionList="./src/vectorEncoding/cache/conts/*.txt")
    loader = tqdm(loader, desc="Loading and encoding CFGs", ncols=0)

    # loads in and pre processes the CFGs
    for cfg in loader:
        try:
            # still loads in the cfg upto a quantity of count even if it doesnt have a label
            if (cfg.addr not in labels):
                if unlabelled_CFGs != 0:
                    unlabelled_CFGs -= 1
                else:
                    continue
            cfg.load()
            cfgs.append(cfg)
            tokens = tokeniser.Tokeniser.preProcessing(cfg)
            temp_counts = tokeniser.Tokeniser.tokenise(tokens)

            _counts: Counter[tuple[int | tuple[int, int]]] = Counter(temp_counts)
            counts.update(_counts)

            cfg.gen_indexes(temp_counts, counts)

        except KeyboardInterrupt:
            break

    # remove the long tail of vectors that are not used frequently
    cfgs, counts = shrinkCounts(counts, cfgs, length=500)

    print(f"Compression ratio of: {100 * (1 - (len(counts) / sum(list(counts.values())))):.2f}%")

    tfIdfVectors = TF_IDF(counts)()
    ic(tfIdfVectors.shape)

    averageVectors = Average(counts)()
    ic(averageVectors.shape)

    LSTM_Encodings = LSTM_AutoEnc_Training(counts, lstmWidth, unlabelled_CFGs).getEncodings()
    ic(LSTM_Encodings.shape)

    cfgs = graphCompression(
        CFGs=cfgs,
        pathToTags="./addressTags.csv",
        pathToLabels="./labels.json",
        _lstm=LSTM_Encodings,
        _counts=counts,
    ).compress()

    # graph labeling
    gc = graphLabeling(
        CFGs=cfgs,
        pathToTags="./addressTags.csv",
        pathToLabels="./labels.json",
        # _tf_idf=tfIdfVectors,
        _counts=counts,
        # _average=averageVectors,
        _lstm=LSTM_Encodings,
        graphName=graphName
    )

    # gc.getGraphLabels()

    propagatedLabels = gc.propagateLabels()


if __name__ == "__main__":
    # for i in [1, 2 ,4 , 8, 16, 32, 64, 128, 172, 256, 512]: # includes the width of the other vectors
    for i in [32]: # includes the width of the other vectors
        main(
            f"matrix_of_confusion_matrix_shrunk_LSTM_Width {i}.png",
            # tf_idf = True,
            # average = True,
            # lstm=True,
            lstmWidth=i,
            unlabelled_CFGs=1000
        )
