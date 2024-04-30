from tqdm import tqdm
from itertools import permutations
from math import factorial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np
from numpy import typing as npt
from CFG_reader import CFG_Reader
from graphComp.graphLoading import graphLoader

class graphCompression(graphLoader):
    def __post_init__(self):
        # generates a IDF for the labels for every node to find the importance of the labels
        # for cfg in self.CFGs:
        #     print(cfg.label)

        length = self.average.shape[0]
        importance = np.zeros((3, length))
        
        counts: dict[str, int] = {"defi": 0, "nft": 0, "erc20": 0}

        # for every term search every CFG
        for cfg in tqdm(self.CFGs):
            indexes: list[int] = [x[1] for x in cfg.graph.nodes(data='extIndex')] # type: ignore
            indexes = list(set(indexes))
            if cfg.label == "defi":
                index = 0
                counts["defi"] += 1
            elif cfg.label == "nft":
                index = 1
                counts["nft"] += 1
            else:
                # ERC-20
                index = 2
                counts["erc20"] += 1

            importance[index][indexes] += 1

        # calculate the co-occurance of the labels
        importance = importance / np.array([[counts["defi"], counts["nft"], counts["erc20"]]]).T
        print(len(self.CFGs))
        print(importance.shape)
        print(np.max(importance))
        print(np.min(importance))
        print(np.mean(importance))

        freq = self.counts.values()
        print(freq)
        print(f"{len(freq) = }")

    def compress(self) -> list[CFG_Reader]:
        return self.CFGs