import numpy as np
from numpy import typing as npt
from CFG_reader import CFG_Reader
import pandas as pd
import json
from sklearn.metrics.pairwise import euclidean_distances


class graphLoader:
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
            self.tf_idfVectors_cosine_similarity_np = cosine_similarity_np(_tf_idf)
            self.tf_idfVectors_euclideanDistance = euclideanDistance(_tf_idf)
            self.tf_idfVectors_dotProduct = dotProduct(_tf_idf)
        if _average is not None:
            self.average = _average
            self.averageVectors_cosine_similarity_np = cosine_similarity_np(_average)
            self.averageVectors_euclideanDistance = euclideanDistance(_average)
            self.averageVectors_dotProduct = dotProduct(_average)
        if _lstm is not None:
            self.lstm = _lstm
            self.lstmVectors_cosine_similarity_np = cosine_similarity_np(_lstm)
            self.lstmVectors_euclideanDistance = euclideanDistance(_lstm)
            self.lstmVectors_dotProduct = dotProduct(_lstm)

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

def cosine_similarity_np(matrix: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    dot_product = np.dot(matrix, matrix.T)
    norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
    cosine_similarities = dot_product / (norm_matrix * norm_matrix.T)
    # apply cutoff
    # cosine_similarities = cosine_similarities > cutoff
    return cosine_similarities

def euclideanDistance(matrix: np.ndarray) -> np.ndarray:
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
    # print(matrix.shape)
    # exit(0)
    return dist

def dotProduct(matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the dot product between points in a matrix.

        Parameters:
        - matrix: A numpy array of shape (n, m) containing n points with m dimensions.

        Returns:
        - distances: A numpy array of shape (n, n) containing the pairwise dot products between points.
        """
        # Calculate squared distances
        dist = np.dot(matrix, matrix.T)
        # print(matrix.shape)
        # exit(0)
        return dist / matrix.shape[1]
