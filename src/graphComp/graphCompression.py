# from tqdm import tqdm
# from itertools import permutations
# from math import factorial
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from matplotlib import pyplot as plt
import numpy as np
from numpy import typing as npt
from CFG_reader import CFG_Reader
from graphComp.graphLoading import graphLoader
import networkx as nx
# from icecream import ic


class graphCompression(graphLoader):
    def compress(self, compress=True) -> tuple[list[CFG_Reader], npt.NDArray[np.float_]]:
        length = self.lstm.shape[0]
        importance: npt.NDArray[np.float_] = np.zeros((3, length))

        counts: list[int] = [0, 0, 0]
        label: dict[str, int] = {"defi": 0, "nft": 1, "erc20": 2}

        # for every term search every CFG
        for cfg in self.CFGs:
            if cfg.label == "unknown":
                continue
            indexes: list[int] = [x[1] for x in cfg.graph.nodes(data='extIndex')]  # type: ignore
            indexes = [x for x in indexes if x != -1]
            indexes = list(set(indexes))
            contractType = label[cfg.label]
            counts[contractType] += 1
            importance[contractType][indexes] += 1

        # calculate the co-occurance of the labels
        # normalises for the varying distribution of
        np_counts = np.array([counts]).T
        importance = importance / np_counts
        # normalises for the number of occourances
        importance = importance / np.array(list(self.counts.values()))
        # importance = importance / np.max(importance, axis=0)
        if not compress:
            return self.CFGs, importance

        # Get start and end nodes
        for i, cfg in enumerate(self.CFGs):
            if cfg.label == "unknown":
                continue
            nodes: dict[int, int] = {k: v for k, v in cfg.graph.nodes(data='extIndex')}  # type: ignore
            # nodes = {k: v for k, v in nodes.items() if v != -1}
            endNodeIndex = list(cfg.graph.nodes)[-1]
            _label = label[cfg.label]
            func = lambda x, _, __: (1 / importance[_label, nodes[x]]) + np.sum(importance[:, nodes[x]])

            # find the shortest path from the start to the end node
            # this is the path that has the most impactful nodes
            subGraph = set()
            tempGraph = cfg.graph.copy()
            while True:
                try:
                    path = nx.shortest_path(
                        tempGraph,
                        source=0,
                        target=endNodeIndex,
                        weight=func)
                except nx.exception.NetworkXNoPath:
                    break
                subGraph.update(path)
                tempGraph.remove_nodes_from(path[1:-1])
                break

            # remove all nodes from cfg that aren't in subgraph
            cfg.graph.remove_nodes_from(set(cfg.graph.nodes) - subGraph)
            self.CFGs[i].graph = cfg.graph

        return self.CFGs, importance


"""
return {
    n for n, d in G.in_degree() if d == 0
}
return {
    n for n, d in G.out_degree() if d == 0
}
"""
