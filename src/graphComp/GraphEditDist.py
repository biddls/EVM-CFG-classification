# import networkx as nx
import numpy as np
from numpy import typing as npt
from CFG_reader import CFG_Reader
import pandas as pd
import json
from tqdm import tqdm
# from graphComp.nodeDict import CFG_Node
# from difflib import SequenceMatcher
# from collections import Counter
from itertools import permutations
# from numpy import dot
# from numpy.linalg import norm
from math import factorial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import euclidean_distances


class graphClassification:
    """
    Class for labeling graph
    """
    CFGs: list[CFG_Reader]
    pathToTypes: str
    pathToLabels: str
    addrLabels: dict[str, str]
    
    tf_idf: npt.NDArray
    tf_idfVectors: npt.NDArray[np.float_]
    average: npt.NDArray
    averageVectors: npt.NDArray[np.float_]
    lstm: npt.NDArray
    lstmVectors: npt.NDArray[np.float_]

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

        # set the classes to the CFGs
        self.setClassesToCFGs()

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
        for addr, _tags in tags.items():
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
        # print(f"Number of addresses with labels: {len(addrLabels)}")

    def setClassesToCFGs(self):
        """
        Takes the labels found for the different classes
        and writes them to the CFGs stored
        """
        for CFG in self.CFGs:
            addr = CFG.addr
            try:
                CFG.label = self.addrLabels[addr]
            except KeyError:
                CFG.label = "unknown"

    # def nodeSimilarity(self, node1: CFG_Node, node2: CFG_Node) -> bool:
    #     """
    #     Get the similarity between two nodes
    #     """

    #     node1Index = node1["index"]
    #     node2Index = node2["index"]
    #     return self.getSimilarity(node1Index, node2Index)

    # def getSimilarity(self, node1Index: int, node2Index: int) -> bool:
    #     # looks up the vectors for each node
    #     node1Vector = self.tf_idf[node1Index]
    #     node2Vector = self.tf_idf[node2Index]

    #     cos_sim = dot(node1Vector, node2Vector)/(norm(node1Vector)*norm(node2Vector))

    #     # print(f"cosine similarity: {cos_sim}")
    #     if cos_sim > 0.8:
    #         return True
    #     else:
    #         return False

    def getGraphSimilarity(self, CFG_1: CFG_Reader, CFG_2: CFG_Reader) -> float:
        """
        Get the similarity between two graphs
        """
        # Get the nodes
        nodes1: list[int] = [x[1] for x in CFG_1.graph.nodes(data='extIndex')] # type: ignore
        nodes2: list[int] = [x[1] for x in CFG_2.graph.nodes(data='extIndex')] # type: ignore

        # get the similarity between the nodes using tf-idf
        indexMatrix = np.ix_(nodes1, nodes2)
        result_matrix: npt.NDArray[np.float_] = self.lstmVectors.T[indexMatrix]

        return float(np.average(result_matrix))

    def getGraphLabels(self):
        """
        generates the labels for the graphs
        """
        pairs = permutations(range(len(self.CFGs)), 2)
        lenPairs = int(factorial(len(self.CFGs))/factorial(len(self.CFGs)-2))
        pairs = tqdm(pairs, total=lenPairs, smoothing=0)

        similarityMatrix = np.identity(len(self.CFGs))

        # todo: table is mirrored so we only need to calculate half of it
        for pair1, pair2 in pairs:
            graph1 = self.CFGs[pair1]
            graph2 = self.CFGs[pair2]
            similarity = self.getGraphSimilarity(graph1, graph2)
            similarityMatrix[pair1, pair2] = similarity

        from matplotlib import pyplot as plt

        plt.imshow(similarityMatrix)
        plt.colorbar
        plt.savefig("temp.png")

        # get labels for the graphs and their corresponding addresses
        # for cfg in self.CFGs:
        #     print(f"address: {cfg.addr}, label: {cfg.label}")

        labels = [cfg.label for cfg in self.CFGs]

        defiIndces = [i for i, x in enumerate(labels) if x == "defi"]
        nftIndces = [i for i, x in enumerate(labels) if x == "nft"]
        erc20Indces = [i for i, x in enumerate(labels) if x == "erc20"]
        print(f"defi: {len(defiIndces)}, nft: {len(nftIndces)}, erc20: {len(erc20Indces)}")

        defiTestIndces = defiIndces[:int(len(defiIndces)/5)]
        defiTrainIndces = defiIndces[int(len(defiIndces)/5):]

        nftTestIndces = nftIndces[:int(len(nftIndces)/5)]
        nftTrainIndces = nftIndces[int(len(nftIndces)/5):]
        
        erc20TestIndces = erc20Indces[:int(len(erc20Indces)/5)]
        erc20TrainIndces = erc20Indces[int(len(erc20Indces)/5):]

        testIndces = defiTestIndces + nftTestIndces + erc20TestIndces
        trueLabels = [self.CFGs[i].label for i in testIndces]
        predLabels = list()

        for i in testIndces:
            # if cfg.label != "unknown":
            #     continue
            # print(similarityMatrix[i, defiTrainIndces].shape)
            # print(similarityMatrix[i, nftTrainIndces].shape)
            # print(similarityMatrix[i, erc20TrainIndces].shape)
            # print(np.max(similarityMatrix[i, defiTrainIndces]))
            # print(np.max(similarityMatrix[i, nftTrainIndces]))
            # print(np.max(similarityMatrix[i, erc20TrainIndces]))
            # exit(0)
            defiSim  = np.average(similarityMatrix[i, defiTrainIndces])
            nftSim  = np.average(similarityMatrix[i, nftTrainIndces])
            erc20Sim  = np.average(similarityMatrix[i, erc20TrainIndces])


            # cfg = self.CFGs[i]

            # write the maximum to the cfg
            if defiSim > nftSim and defiSim > erc20Sim:
                predLabels.append("defi")
                print(f"defi: {defiSim}")
            elif nftSim > defiSim and nftSim > erc20Sim:
                predLabels.append("nft")
                print(f"nft: {nftSim}")
            elif erc20Sim > defiSim and erc20Sim > nftSim:
                predLabels.append("erc20")
                print(f"erc20: {erc20Sim}")

        confMatrix = confusion_matrix(trueLabels, predLabels, labels=["defi", "nft", "erc20"])
        print(confMatrix)

        # plot the confusion matrix
        # fig, ax = plt.subplots()
        # cax = ax.matshow(confMatrix, cmap=plt.cm.Blues) # type: ignore

        # # Add labels
        # # ax.set_xlabel(["defi", "nft", "erc20"])
        # # ax.set_ylabel(["defi", "nft", "erc20"])

        # # Add color bar
        # fig.colorbar(cax)

        # # Save the plot
        # plt.savefig('confusion_matrix.png')
        
        disp = ConfusionMatrixDisplay(confMatrix , display_labels=["defi", "nft", "erc20"])

        disp.plot()
        plt.savefig('confusion_matrix.png')

    def cosine_similarity_np(self, matrix: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        dot_product = np.dot(matrix, matrix.T)
        norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
        cosine_similarities = dot_product / (norm_matrix * norm_matrix.T)
        # apply cutoff
        # cosine_similarities = cosine_similarities > cutoff
        return cosine_similarities

    def euclideanDistance(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the Euclidean distance between points in a matrix.

        Parameters:
        - matrix: A numpy array of shape (n, m) containing n points with m dimensions.

        Returns:
        - distances: A numpy array of shape (n, n) containing the pairwise Euclidean distances between points.
        """
        # Calculate squared distances
        # print(matrix)
        dist = euclidean_distances(matrix, squared=True) / matrix.shape[1]
        return dist
    
    def dotProduct(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the dot product between points in a matrix.

        Parameters:
        - matrix: A numpy array of shape (n, m) containing n points with m dimensions.

        Returns:
        - distances: A numpy array of shape (n, n) containing the pairwise dot products between points.
        """
        # Calculate squared distances
        dist = np.dot(matrix, matrix.T)
        print(matrix.shape)
        exit(0)
        return dist / matrix.shape[1]
