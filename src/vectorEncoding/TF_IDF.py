from collections import Counter
from typing import Any
import numpy as np
import numpy.typing as npt
from tokeniser import Tokeniser
from tqdm import tqdm


class TF_IDF:
    data: list[npt.NDArray[np.bool_]]
    counts : list[int]
    width: int


    def __init__(self, data: Counter[tuple[int | tuple[int, int]]]):
        """
        Shape of the data:
        first level is the list of cfgs (documents)
        The second level is the list of nodes 
        The third level is the vector representation of the node
        """

        temp = list(data.keys())
        vectorise = lambda x: Tokeniser.vectoriseNode(x)
        temp = list(map(vectorise, list(temp)))
        self.data = temp
        self.counts = list(data.values())
        self.width = self.data[0].shape[1]

    def __call__(self, *args: Any, **kwds: Any) -> npt.NDArray[np.float64]:
        tf = self.__termFrequency()
        idf = self.__inverseDocumentFrequency()
        return tf * idf

    def __termFrequency(self) -> npt.NDArray[np.float64]:
        """
        Calculates the term frequency of the data
        for each CFG (document) it calculates the term frequency of each token (word)
        data: [CFG[Tokens[vector]]
        """

        # flatten the data
        # calculate the term frequency
        # [number of documents, number of tokens in each word]
        tf = np.zeros((len(self.data), self.width))
        # for each document
        for i, document in enumerate(self.data):
            length = len(document)
            document = np.array(document)
            # sums the values down each column 
            # divides by the number of words in the document
            tf[i] = document.sum(axis=0) / length
        return tf

    def __inverseDocumentFrequency(self) -> npt.NDArray[np.float64]:
        """
        Calculates the inverse document frequency of the data
        """
        # calculate the inverse document frequency
        idf = np.zeros(self.width)
        # calculate the number of documents that contain the token
        for node, counts in zip(self.data, self.counts):
            temp = np.clip(np.sum(node, axis=0), None, 1)
            # the " * counts" is the number of documents that are the same as the current document
            idf += temp * counts

        # divide by the number of documents &
        # take the log of the result
        idf += 1
        idf: npt.NDArray[np.float64] = np.log(len(self.data) / idf)
        return idf