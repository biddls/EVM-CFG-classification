import networkx as nx
import numpy as np
from numpy import typing as npt
# from matplotlib import pyplot as plt
from CFG_reader import CFG_Reader
import pandas as pd
import json
from tqdm import tqdm
from graphComp.nodeDict import CFG_Node
from difflib import SequenceMatcher
from collections import Counter
from numpy import dot
from numpy.linalg import norm
from functools import lru_cache
import sys
sys.setrecursionlimit(1500)


class graphClassification:
    """
    Class for labeling graph
    """
    CFGs: list[CFG_Reader]
    pathToTypes: str
    pathToLabels: str
    addrLabels: dict[str, str]
    
    tf_idf: npt.NDArray
    tf_idfVectors: npt.NDArray[np.bool_]
    average: npt.NDArray
    averageVectors: npt.NDArray[np.bool_]
    lstm: npt.NDArray
    lstmVectors: npt.NDArray[np.bool_]

    def __init__(
        self,
        CFGs: list[CFG_Reader],
        pathToTags: str,
        pathToLabels: str,
        _tf_idf: npt.NDArray | None = None,
        _average: npt.NDArray | None = None,
        _lstm: npt.NDArray | None = None,
    ) -> None:

        # Load in all data
        self.CFGs = CFGs
        self.pathToTypes = pathToTags
        self.pathToLabels = pathToLabels
        if _tf_idf is not None:
            self.tf_idf = _tf_idf
            self.tf_idfVectors = self.cosine_similarity_np(_tf_idf)
        if _average is not None:
            self.average = _average
            self.averageVectors = self.cosine_similarity_np(_average)
        if _lstm is not None:
            self.lstm = _lstm
            self.lstmVectors = self.cosine_similarity_np(_lstm)

        # load in classes and labels
        self.loadClasses()

    def loadClasses(self):
        """
        Load classes from file
        """
        # Load labels from Json
        with open(self.pathToLabels, "r") as f:
            tempDict: dict[str, list[str]] = json.load(f)

        # invert mapping
        labels: dict[str, str] = {
            subV: k
            for k, v in tempDict.items()
            for subV in v}

        # load CSV
        df = pd.read_csv(self.pathToTypes)
        df.drop(["source"], inplace=True, axis=1)
        df = df.dropna(subset=["tag"])

        # print(df.head())
        # print(f"Shape of df: {df.shape}")

        # convert it to an easier to use format
        tags: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            if row['address'] in tags:
                tags[row['address']].append(row['tag'])
            else:
                tags[row['address']] = [row['tag']]

        # print(f"Number of unique addresses with tags: {len(tags)}")

        # count = 0
        # for _tags in tags.values():
        #     for _tag in _tags:
        #         if _tag in labels.keys():
        #             count += 1
        #             break

        # print(f"Number of unique addresses with tags in labels: {count}")

        # label the address using the tags
        addrLabels: dict[str, str] = {}
        # label the tags
        for addr, _tags in tqdm(tags.items()):
            # Get the tags
            _tags = list(set(_tags))
            _tags = list(
                filter(
                    lambda x: x != 'Source Code',
                    _tags
                )
            )

            if len(_tags) == 0:
                continue

            tempLabels = []
            for tag in _tags:
                try:
                    tempLabels.append(labels[tag])
                except KeyError:
                    pass

            # remove "other"
            tempLabels = list(filter(lambda x: x != "other", tempLabels))
            defi = tempLabels.count("defi")
            nft = tempLabels.count("nft")
            erc20 = tempLabels.count("erc20")

            # get the most common label
            if defi > nft and defi > erc20:
                addrLabels[addr] = "defi"
            elif nft > defi and nft > erc20:
                addrLabels[addr] = "nft"
            elif erc20 > defi and erc20 > nft:
                addrLabels[addr] = "erc20"
            else:
                continue

        self.addrLabels = addrLabels

    def nodeSimilarity(self, node1: CFG_Node, node2: CFG_Node) -> bool:
        """
        Get the similarity between two nodes
        """

        node1Index = node1["index"]
        node2Index = node2["index"]
        return self.getSimilarity(node1Index, node2Index)

    @lru_cache(maxsize=None)
    def getSimilarity(self, node1Index: int, node2Index: int) -> bool:
        # looks up the vectors for each node
        node1Vector = self.tf_idf[node1Index]
        node2Vector = self.tf_idf[node2Index]

        cos_sim = dot(node1Vector, node2Vector)/(norm(node1Vector)*norm(node2Vector))

        # print(f"cosine similarity: {cos_sim}")
        if cos_sim > 0.8:
            return True
        else:
            return False

    def getGraphSimilarity(self, CFG_1: CFG_Reader, CFG_2: CFG_Reader):
        """
        Get the similarity between two graphs
        """
        print(f"Size of graph1: {len(CFG_1.graph.nodes)}")
        print(f"Size of graph2: {len(CFG_2.graph.nodes)}")
        maxNodeCombi = len(CFG_1.graph.nodes) * len(CFG_2.graph.nodes)
        print(f"Max number of node combinations: {maxNodeCombi}")

        # # ! this wont work as its too slow
        # dist = nx.optimize_graph_edit_distance(
        #     graph1.graph,
        #     graph2.graph,
        #     # node_match=lambda x, y: True)
        #     node_match=self.nodeSimilarity)

        # for temp in dist:
        #     print(temp)
        #     minDist = temp

        # print(f"Graph edit distance: {minDist}")

        # Doing a simple comparison of the nodes
        # Get the nodes
        nodes1: map[CFG_Node] = map(lambda x: CFG_1.graph.nodes[x], list(CFG_1.graph.nodes))
        nodes2: map[CFG_Node] = map(lambda x: CFG_2.graph.nodes[x], list(CFG_2.graph.nodes))

        # get only the external indexes from the nodes using map
        # print(nodes1[0])
        nodeIndexesTEMP1 = list(map(lambda x: x["index"], nodes1))
        nodeIndexesTEMP2 = list(map(lambda x: x["index"], nodes2))

        # get the similarity between the nodes using tf-idf
        indexMatrix = np.ix_(nodeIndexesTEMP1, nodeIndexesTEMP2)
        result_matrix: npt.NDArray[np.bool_] = self.tf_idfVectors.T[indexMatrix]
        print(result_matrix.shape)

        exit(0)

    def getGraphLabels(self):
        """
        generates the labels for the graphs
        """
        cfg1 = self.CFGs[0]
        cfg2 = self.CFGs[1]
        self.getGraphSimilarity(cfg1, cfg2)

    def cosine_similarity_np(self, matrix: npt.NDArray[np.float64], cutoff: float = 0.8) -> npt.NDArray[np.bool_]:
        dot_product = np.dot(matrix, matrix.T)
        norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
        cosine_similarities = dot_product / (norm_matrix * norm_matrix.T)
        # apply cutoff
        cosine_similarities = cosine_similarities > cutoff
        return cosine_similarities
