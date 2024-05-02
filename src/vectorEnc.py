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
from graphComp.graphClassification import graphCompression
from graphComp.graphLabeling import graphLabeling
from icecream import ic


def main(
    tf_idf: bool = False,
    average: bool = False,
    lstm: bool = False,
    max_cfgs: int = 0 # set to 0 to do all the data
) -> None:
    count = 0
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
            if cfg.addr not in labels:
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

    print(f"Compression ratio of: {100 * (1 - (len(counts) / sum(list(counts.values())))):.2f}%")

    # Getting vectors
    # TF-IDF
    tfIdfVectors = None
    if tf_idf:
        tfIdfVectors = TF_IDF(counts)()
        ic(tfIdfVectors.shape)

    # Average
    averageVectors = None
    if average:
        averageVectors = Average(counts)()
        ic(averageVectors.shape)

    # LSTM autoencoder
    LSTMEncodings = None
    if lstm:
        # # Creating a single set of axes
        # _, ax = plt.subplots()  # Adjust figsize as needed

        # # Plotting all columns on the same plot
        # for i in [1, 2 ,4 , 8, 16, 32, 64, 128, 172, 256, 512]: # includes the width of the other vectors
        #     trainer = LSTM_AutoEnc_Training(counts, i, count)
        #     finalLosses = trainer.trainEnc(3, checkpoints=True, progress=False)
        #     # np.savetxt(f"./src/vectorEncoding/cache/checkpointsLSTM_Autoenc/width{i}/losses.txt", finalLosses, delimiter=" ")
        #     ax.plot(range(len(finalLosses)), finalLosses, label=f'Hidden Dim width: {i}')

        # # Adding labels and legend
        # ax.set_title('Multiple Columns on the Same Plot')
        # ax.set_xlabel('Data Points')
        # ax.set_ylabel('Values')
        # ax.legend()

        # # Display the plot
        # plt.show()

        trainer = LSTM_AutoEnc_Training(counts, 172, count)
        LSTMEncodings = trainer.getEncodings()
        ic(LSTMEncodings.shape)

    shapes = [
        tfIdfVectors.shape if tfIdfVectors is not None else None,
        averageVectors.shape if averageVectors is not None else None,
        LSTMEncodings.shape if LSTMEncodings is not None else None
    ]

    shapes = [shape for shape in shapes if shape is not None]

    if len(shapes) > 1:
        if all([shapes[0] == shape for shape in shapes]):
            ic("All shapes are equal")
        else:
            raise ValueError(f"Shapes are not equal: {shapes}")

    cfgs = graphCompression(
        CFGs=cfgs,
        pathToTags="./addressTags.csv",
        pathToLabels="./labels.json",
        _tf_idf=tfIdfVectors,
        _counts=counts,
    ).compress()

    for cfg in cfgs:
        ic(cfg.addr, cfg.nodeCount(), cfg.edgeCount())

    # exit(0)

    # graph labeling
    gc = graphLabeling(
        CFGs=cfgs,
        pathToTags="./addressTags.csv",
        pathToLabels="./labels.json",
        _tf_idf=tfIdfVectors,
        _counts=counts,
        _average=averageVectors,
        _lstm=LSTMEncodings,
        graphName="matrix_of_confusion_matrix_shrunk.png"
    )

    gc.getGraphLabels()

"""
Things to try:
    [ ] - Doc2Vec # will do if more time is available
    [X] - LSTM Autoencoder
    [x] - TF-IDF
    [~] - Attention # Dont think this is going to work
    [X] - Simple average
"""

if __name__ == "__main__":
    main(
        tf_idf = True,
        average = True,
        lstm=True,
        # max_cfgs = 50
    )
