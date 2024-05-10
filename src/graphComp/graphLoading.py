from collections import Counter
import numpy as np
from numpy import typing as npt
from CFG_reader import CFG_Reader
import pandas as pd
import json
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from matplotlib import pyplot as plt
from icecream import ic


class graphLoader:
    """
    Class for labeling graph
    """
    CFGs: list[CFG_Reader]
    pathToTypes: str
    pathToLabels: str
    addrLabels: dict[str, str]
    counts: Counter[tuple[int | tuple[int, int]]]

    tf_idf: npt.NDArray
    average: npt.NDArray
    lstm: npt.NDArray

    def __init__(
        self,
        CFGs: list[CFG_Reader],
        pathToTags: str,
        pathToLabels: str,
        _counts: Counter[tuple[int | tuple[int, int]]] | None = None,
        _tf_idf: npt.NDArray | None = None,
        _average: npt.NDArray | None = None,
        _lstm: npt.NDArray | None = None,
        graphName: str = "graph.png"
    ) -> None:

        # Load in all data
        self.CFGs = CFGs
        self.pathToTypes = pathToTags
        self.pathToLabels = pathToLabels
        self.graphName = graphName
        if _counts is not None:
            self.counts = _counts
        
        if _tf_idf is not None:
            self.tf_idf = _tf_idf
        if _average is not None:
            self.average = _average
        if _lstm is not None:
            self.lstm = _lstm

        # if _tf_idf is not None:
        #     counter = tqdm(range(3), ncols=0, desc="Calculating TF-IDF similarities")  
        #     self.tf_idf = _tf_idf
        #     self.tf_idfVectors_cosine_similarity_np = cosine_similarity_np(_tf_idf)
        #     counter.update(1)
        #     self.tf_idfVectors_euclideanDistance = euclideanDistance(_tf_idf)
        #     counter.update(1)
        #     self.tf_idfVectors_dotProduct = dotProduct(_tf_idf)
        #     counter.update(1)
        # if _average is not None:
        #     counter = tqdm(range(3), ncols=0, desc="Calculating Average similarities")  
        #     self.average = _average
        #     self.averageVectors_cosine_similarity_np = cosine_similarity_np(_average)
        #     counter.update(1)
        #     self.averageVectors_euclideanDistance = euclideanDistance(_average)
        #     counter.update(1)
        #     self.averageVectors_dotProduct = dotProduct(_average)
        #     counter.update(1)
        # if _lstm is not None:
        #     counter = tqdm(range(3), ncols=0, desc="Calculating LSTM similarities")  
        #     self.lstm = _lstm
        #     self.lstmVectors_cosine_similarity_np = cosine_similarity_np(_lstm)
        #     counter.update(1)
        #     self.lstmVectors_euclideanDistance = euclideanDistance(_lstm)
        #     counter.update(1)
        #     self.lstmVectors_dotProduct = dotProduct(_lstm)
        #     counter.update(1)

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
    
    def _cosine_similarity_np(
        self,
        matrix1: npt.NDArray[np.int_],
        matrix2: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float_]:
        matrix1 = matrix1.squeeze()
        matrix2 = matrix2.squeeze()
        dot_product: npt.NDArray[np.float_] = np.dot(matrix1, matrix2.T)
        norm_matrix1: npt.NDArray[np.int_] = np.linalg.norm(matrix1, axis=1, keepdims=True)
        norm_matrix2: npt.NDArray[np.int_] = np.linalg.norm(matrix2, axis=1, keepdims=True)
        cosine_similarities = dot_product / (norm_matrix1 * norm_matrix2.T)
        return cosine_similarities

    def _euclideanDistance(
        self,
        matrix1: npt.NDArray[np.int_],
        matrix2: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float_]:
        matrix1 = matrix1.squeeze()
        matrix2 = matrix2.squeeze()
        return euclidean_distances(matrix1, matrix2) / matrix1.shape[1]

    def _dotProduct(
        self,
        matrix1: npt.NDArray[np.int_],
        matrix2: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float_]:
        matrix1 = matrix1.squeeze()
        matrix2 = matrix2.squeeze()
        return np.dot(matrix1, matrix2.T) / matrix1.shape[1]
