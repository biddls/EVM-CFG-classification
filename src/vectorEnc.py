import tokeniser
from vectorEncoding.LSTM_Autoenc import LSTM_AutoEnc_Training
from vectorEncoding.TF_IDF import TF_IDF
from vectorEncoding.averagingVectors import Average
from tqdm import tqdm
from collections import Counter
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    loader = tokeniser.CFG_Loader(exclusionList="./src/vectorEncoding/cache/conts/*.txt")
    count = 0
    cfgs = list()

    # set to 0 to do all the data
    max_cfgs = 0
    counts: Counter[tuple[int | tuple[int, int]]] = Counter()
    countToggle = Enum("enableCounting", {"Counting": True, "notCounting": False}).Counting
    loader = tqdm(loader, desc="Loading and encoding CFGs")
    data: dict[str, list[tuple[int | tuple[int, int]]]] = {}

    for cfg in loader:
        # print(cfg)
        try:
            cfgs.append(cfg.addr)
            tokens = tokeniser.Tokeniser.preProcessing(cfg)
            temp_counts = tokeniser.Tokeniser.tokenise(tokens, counting=countToggle.value)

            if countToggle == countToggle.Counting:
                if isinstance(counts, Counter):
                    # adds temp_counts to counts adding the values together
                    counts.update(temp_counts)
            elif countToggle == countToggle.notCounting:
                raise NotImplementedError("Not implemented yet")
                # cfg.addTokens(tokens)

            count += 1
            if count == max_cfgs:
                break

        except KeyboardInterrupt:
            break

    print(f"Compression ratio of: {100 * (1 - (len(counts) / sum(list(counts.values())))):.2f}%")


    # TF-IDF
    tfIdfVectors = TF_IDF(counts)()
    print(f"{tfIdfVectors.shape = }")

    # Average
    # raise NotImplementedError("Averaging is Not implemented yet")
    averageVectors = Average(counts)()
    print(f"{averageVectors.shape = }")

    # LSTM autoencoder
    # Creating a single set of axes
    fig, ax = plt.subplots()  # Adjust figsize as needed

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
        tfIdfVectors.shape,
        averageVectors.shape,
        LSTMEncodings.shape
    ]

    if all([shapes[0][0] == shape[0] for shape in shapes]):
        print("All shapes are equal length")

    # chart the losses from finalLosses
    # plt.plot(finalLosses)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Losses over Epochs for LSTM Autoencoder")
    # plt.show()


"""
Things to try:
    [ ] - Doc2Vec # will do if more time is available
    [X] - LSTM Autoencoder
    [x] - TF-IDF
    [~] - Attention # Dont think this is going to work
    [X] - Simple average
"""
