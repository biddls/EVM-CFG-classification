import networkx as nx
# import numpy as np
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
import sys
sys.setrecursionlimit(1500)


class graphClassification:
    """
    Class for labeling graph
    """
    CFGs: list[CFG_Reader]
    pathToTypes: str
    pathToLabels: str
    tf_idf: npt.NDArray
    average: npt.NDArray
    lstm: npt.NDArray
    addrLabels: dict[str, str]

    def __init__(
        self,
        CFGs: list[CFG_Reader],
        pathToTags: str,
        pathToLabels: str,
        tf_idf: npt.NDArray | None = None,
        average: npt.NDArray | None = None,
        lstm: npt.NDArray | None = None,
    ) -> None:

        # Load in all data
        self.CFGs = CFGs
        self.pathToTypes = pathToTags
        self.pathToLabels = pathToLabels
        if tf_idf is not None:
            self.tf_idf = tf_idf
        if average is not None:
            self.average = average
        if lstm is not None:
            self.lstm = lstm

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

        # looks up the vectors for each node
        node1Vector = self.tf_idf[node1Index]
        node2Vector = self.tf_idf[node2Index]

        cos_sim = dot(node1Vector, node2Vector)/(norm(node1Vector)*norm(node2Vector))

        # print(f"cosine similarity: {cos_sim}")
        if cos_sim > 0.8:
            return True
        else:
            return False

    def getGraphSimilarity(self, graph1: CFG_Reader, graph2: CFG_Reader):
        """
        Get the similarity between two graphs
        """
        print(f"Size of graph1: {len(graph1.graph.nodes)}")
        print(f"Size of graph2: {len(graph2.graph.nodes)}")

        dist = nx.optimize_graph_edit_distance(
            graph1.graph,
            graph2.graph,
            node_match=self.nodeSimilarity)

        for temp in dist:
            print(temp)
            minDist = temp

        print(f"Graph edit distance: {minDist}")

        exit(0)

    def getGraphLabels(self):
        """
        generates the labels for the graphs
        """
        cfg1 = self.CFGs[0]
        cfg2 = self.CFGs[1]
        self.getGraphSimilarity(cfg1, cfg2)