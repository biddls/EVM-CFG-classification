from random import shuffle
import tokeniser
# from vectorEncoding.LSTM_Autoenc import LSTM_AutoEnc_Training
# from vectorEncoding.TF_IDF import TF_IDF
from vectorEncoding.averagingVectors import Average
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from CFG_reader import CFG_Reader
from graphComp.graphLoading import graphLoader
from graphComp.graphCompression import graphCompression
from graphComp.graphLabeling import graphLabeling
# from icecream import ic
# from vectorEncoding.tailRemoval import shrinkCounts
from sklearn.metrics import ConfusionMatrixDisplay


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
    # cfgs, counts = shrinkCounts(counts, cfgs, length=500)

    # print(f"Compression ratio of: {100 * (1 - (len(counts) / sum(list(counts.values())))):.2f}%")

    # tfIdfVectors = TF_IDF(counts)()
    # ic(tfIdfVectors.shape)

    averageVectors = Average(counts)()
    # ic(averageVectors.shape)

    # LSTM_Encodings = LSTM_AutoEnc_Training(counts, lstmWidth, unlabelled_CFGs).getEncodings(prog=False)
    # ic(LSTM_Encodings.shape)

    totalConfusionMatrix = np.zeros((3, 3))
    for _ in tqdm(range(0, 100), ncols=0):
        shuffle(cfgs)

        cfgs = graphLoader(
            cfgs,
            "./addressTags.csv",
            "./labels.json",
        ).CFGs

        cfgLabels = [x.label for x in cfgs]
        nftIndex = list()
        erc20Index = list()
        defiIndex = list()
        for i, label in enumerate(cfgLabels):
            if label == "nft" and len(nftIndex) < 10:
                nftIndex.append(i)
                cfgs[i].label = "unknown"
            elif label == "erc20" and len(erc20Index) < 10:
                erc20Index.append(i)
                cfgs[i].label = "unknown"
            elif label == "defi" and len(defiIndex) < 10:
                defiIndex.append(i)
                cfgs[i].label = "unknown"

        # graph compression
        cfgs, importanceTable = graphCompression(
            CFGs=cfgs,
            pathToTags="./addressTags.csv",
            pathToLabels="./labels.json",
            _lstm=averageVectors,
            _counts=counts,
        ).compress(compress=False)

        # graph labeling
        gc = graphLabeling(
            CFGs=cfgs,
            pathToTags="./addressTags.csv",
            pathToLabels="./labels.json",
            # _tf_idf=tfIdfVectors,
            _counts=counts,
            # _average=averageVectors,
            # _lstm=LSTM_Encodings,
            graphName=graphName
        )

        # gc.getGraphLabels()

        # gc.propagateLabelsV1()

        confMatrix = gc.getGrapgLabelsV2(nftIndex, erc20Index, defiIndex, importanceTable)
        totalConfusionMatrix += confMatrix

    disp = ConfusionMatrixDisplay(
        totalConfusionMatrix,
        display_labels=["nft", "erc20", "defi"])
    # plt.ticklabel_format(scilimits=(-5, 8))
    disp.plot(values_format='.0f')
    plt.savefig('total_confusion_matrix.png')
    # gc.propagateLabelsV2(importanceTable)


if __name__ == "__main__":
    i = 172  # includes the width of the other vectors
    main(
        f"matrix_of_confusion_matrix_shrunk_LSTM_Width {i}.png",
        lstmWidth=i,
        unlabelled_CFGs=0)
