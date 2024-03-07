import tokeniser
from vectorEncoding.LSTM_Autoenc import LSTM_AutoEnc_Training
from vectorEncoding.TF_IDF import TF_IDF
from vectorEncoding.averagingVectors import Average
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from CFG_reader import CFG_Reader
from graphComp.GraphEditDist import graphClassification


def main(
    tf_idf: bool = False,
    average: bool = False,
    lstm: bool = False,
    max_cfgs: int = 0 # set to 0 to do all the data
):
    count = 0
    cfgs: list[CFG_Reader] = list()

    counts: Counter[tuple[int | tuple[int, int]]] = Counter()

    loader = tokeniser.CFG_Loader(exclusionList="./src/vectorEncoding/cache/conts/*.txt")
    loader = tqdm(loader, desc="Loading and encoding CFGs")

    # loads in and pre processes the CFGs
    for cfg in loader:
        # print(cfg)
        try:
            cfgs.append(cfg)
            tokens = tokeniser.Tokeniser.preProcessing(cfg)
            # print(f"{len(tokens) = }")
            temp_counts = tokeniser.Tokeniser.tokenise(tokens)
            # print(f"{len(temp_counts) = }")

            _counts: Counter[tuple[int | tuple[int, int]]] = Counter(temp_counts)
            counts.update(_counts)

            cfg.gen_indexes(temp_counts, counts)

            count += 1
            if count == max_cfgs:
                break

        except KeyboardInterrupt:
            break

    print(f"Compression ratio of: {100 * (1 - (len(counts) / sum(list(counts.values())))):.2f}%")

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
        # Creating a single set of axes
        _, ax = plt.subplots()  # Adjust figsize as needed

        # Plotting all columns on the same plot
        for i in [1, 2 ,4 , 8, 16, 32, 64, 128, 256, 512]:
            trainer = LSTM_AutoEnc_Training(counts, i, count)
            finalLosses = trainer.trainEnc(200, checkpoints=True, progress=False)
            np.savetxt(f"./src/vectorEncoding/cache/checkpointsLSTM_Autoenc/width{i}/losses.txt", finalLosses, delimiter=" ")
            ax.plot(range(len(finalLosses)), finalLosses, label=f'Hidden Dim width: {i}')

        # Adding labels and legend
        ax.set_title('Multiple Columns on the Same Plot')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Values')
        ax.legend()

        # Display the plot
        plt.show()

        LSTMEncodings = trainer.getEncodings()
        print(f"{LSTMEncodings.shape = }")

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
    gc = graphClassification(
        CFGs=cfgs,
        pathToTags="./addressTags.csv",
        pathToLabels="./labels.json",
        _tf_idf=tfIdfVectors,
        _average=averageVectors,
        _lstm=LSTMEncodings
    )

    gc.getGraphLabels()

    return tfIdfVectors, averageVectors, LSTMEncodings

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
        max_cfgs=5
    )
