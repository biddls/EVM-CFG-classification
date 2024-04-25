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
from graphComp.graphClassification import graphLabelingSecondTry
from graphComp.graphLabeling import graphLabelingFirstTry


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
    loader = tqdm(loader, desc="Loading and encoding CFGs")

    # loads in and pre processes the CFGs
    for cfg in loader:
        # print(cfg)
        try:
            if cfg.addr not in labels:
                continue
            cfg.load()
            # count += 1
            cfgs.append(cfg)
            tokens = tokeniser.Tokeniser.preProcessing(cfg)
            # print(f"{len(tokens) = }")
            temp_counts = tokeniser.Tokeniser.tokenise(tokens)
            # print(f"{len(temp_counts) = }")

            _counts: Counter[tuple[int | tuple[int, int]]] = Counter(temp_counts)
            counts.update(_counts)

            cfg.gen_indexes(temp_counts, counts)
            # else:
            #     if count == max_cfgs:
            #         break

        except KeyboardInterrupt:
            break

    print(f"Compression ratio of: {100 * (1 - (len(counts) / sum(list(counts.values())))):.2f}%")

    # # plots a histogram of the lengths of the CFGs
    # lenghts = [len(cfg) for cfg in cfgs]
    # # plot as histogram
    # plt.hist(lenghts, bins=50)
    # plt.xlabel("Length of CFG")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of CFG Lengths")
    # plt.axvline(x=float(np.mean(lenghts)), color="orange", label=f"Mean: {np.mean(lenghts):.2f}")
    # plt.legend(loc='upper right')
    # plt.savefig("histogram.png")
    # plt.close()

    # # plots the frequency distribution of the tokens
    # data = sorted(list(counts.values()), reverse=True)
    # print(f"{len(data) = }")
    # data = np.array(data)
    # data = data[data > 10]
    # cumulative = np.cumsum(data)
    # _x = np.arange(len(data))
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    # # plot frequency
    # ax.set_yscale('log')
    # lns1 = ax.plot(_x, data, color='blue', label='Frequency')

    # # plot cumulative frequency
    # ax_bis = ax.twinx()
    # lns2 = ax_bis.plot(_x, cumulative/cumulative[-1], color='red', label='Cumulative Frequency')

    # # plt.xlabel("Token")
    # plt.ylabel("Frequency")
    # plt.title("Frequency Distribution of Tokens")
    # lns = lns1 + lns2
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, loc='center right')
    # plt.savefig("FrequencyDistributionMoreThan10.png")
    # plt.close()

    # # plots the frequency distribution of the tokens
    # data = sorted(list(counts.values()), reverse=True)
    # data = np.array(data)
    # cumulative = np.cumsum(data)
    # _x = np.arange(len(data))
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    # # plot frequency
    # ax.set_yscale('log')
    # lns1 = ax.plot(_x, data, color='blue', label='Frequency')

    # # plot cumulative frequency
    # ax_bis = ax.twinx()
    # lns2 = ax_bis.plot(_x, cumulative/cumulative[-1], color='red', label='Cumulative Frequency')

    # # plt.xlabel("Token")
    # plt.ylabel("Frequency")
    # plt.title("Frequency Distribution of Tokens")
    # lns = lns1 + lns2
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, loc='center right')
    # plt.savefig("FrequencyDistribution.png")
    # plt.close()

    # exit(0)

    # Getting vectors
    # TF-IDF
    tfIdfVectors = None
    if tf_idf:
        tfIdfVectors = TF_IDF(counts)()
        print(f"{tfIdfVectors.shape = }")

    # Average
    averageVectors = None
    if average:
        averageVectors = Average(counts)()
        print(f"{averageVectors.shape = }")

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
        print(f"{LSTMEncodings.shape = }")

    # exit(0)

        # trainer = LSTM_AutoEnc_Training(counts, 172, count)
        # trainer.trainEnc(3, checkpoints=True, progress=True)
        # LSTMEncodings = trainer.getEncodings()
        # print(f"{LSTMEncodings.shape = }")

    shapes = [
        tfIdfVectors.shape if tfIdfVectors is not None else None,
        averageVectors.shape if averageVectors is not None else None,
        LSTMEncodings.shape if LSTMEncodings is not None else None
    ]

    shapes = [shape for shape in shapes if shape is not None]

    if len(shapes) > 1:
        if all([shapes[0] == shape for shape in shapes]):
            print("All shapes are equal")
        else:
            raise ValueError(f"Shapes are not equal: {shapes}")

    # chart the losses from finalLosses
    # plt.plot(finalLosses)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Losses over Epochs for LSTM Autoencoder")
    # plt.show()

    # graph labeling
    gc = graphLabelingSecondTry(
        CFGs=cfgs,
        pathToTags="./addressTags.csv",
        pathToLabels="./labels.json",
        _tf_idf=tfIdfVectors,
        _average=averageVectors,
        _lstm=LSTMEncodings
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
        # tf_idf = True,
        average = True,
        # lstm=True,
        max_cfgs = 50
    )
