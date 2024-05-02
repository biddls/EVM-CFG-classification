from tqdm import tqdm
from itertools import permutations
from math import factorial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np
from numpy import typing as npt
from CFG_reader import CFG_Reader
from graphComp.graphLoading import graphLoader
import networkx as nx
from icecream import ic


class graphCompression(graphLoader):
    def compress(self) -> list[CFG_Reader]:
        length = self.tf_idf.shape[0]
        importance = np.zeros((3, length))
        
        counts: list[int] = [0, 0, 0]
        label: dict[str, int] = {"defi": 0, "nft": 1, "erc20": 2}

        # for every term search every CFG
        for cfg in self.CFGs:
            indexes: list[int] = [x[1] for x in cfg.graph.nodes(data='extIndex')] # type: ignore
            indexes = list(set(indexes))
            contractType = label[cfg.label]
            counts[contractType] += 1
            importance[contractType][indexes] += 1

        # calculate the co-occurance of the labels
        # normalises for the varying distribution of 
        importance = importance / np.array([counts]).T
        # normalises for the number of occourances
        importance = importance / np.array(list(self.counts.values()))
        # ic(len(self.CFGs))
        # ic(importance.shape)
        # ic(np.max(importance))
        # ic(np.min(importance))
        # ic(np.mean(importance))

        # find the indexes that have the most dominance
        dominance = np.max(importance, axis=0) / np.sum(importance, axis=0)
        # plt.hist(dominance)
        # plt.show()
        # ic(dominance.shape)
        # ic(np.max(dominance))
        # ic(np.min(dominance))
        # ic(np.mean(dominance))

        # Get start and end nodes
        for i, cfg in enumerate(self.CFGs):
            nodes: dict[int, int] = {k: v for k, v in cfg.graph.nodes(data='extIndex')} # type: ignore
            endNodeIndex = list(cfg.graph.nodes)[-1]
            _label = label[cfg.label]
            func = lambda x, _, __: 1/importance[_label][nodes[x]]

            subgraph = nx.DiGraph()
            while True:
                try:
                    path = nx.shortest_path(cfg.graph, source=0, target=endNodeIndex, weight=func)
                except nx.exception.NetworkXNoPath:
                    break
                subgraph.add_nodes_from(path)
                path = path[1:-1]
                cfg.graph.remove_nodes_from(path)

            self.CFGs[i].graph = subgraph

        return self.CFGs

"""
return {
    n for n, d in G.in_degree() if d == 0
}
return {
    n for n, d in G.out_degree() if d == 0
}
"""
