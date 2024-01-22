from typing import Any
import numpy as np
import numpy.typing as npt


class TF_IDF:
    def __init__(self, data: list[npt.NDArray[np.bool_]]):        
        self.data = data

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        tf = self._termFrequency(self.data)
        idf = self._inverseDocumentFrequency(self.data)
        return tf * idf

    def _termFrequency(self, data: list[npt.NDArray[np.bool_]]) -> npt.NDArray[np.float64]:
        """
        Calculates the term frequency of the data
        """
        # calculate the term frequency
        tf = np.zeros((len(data), data[0].shape[1]))
        # for each document
        for i, document in enumerate(data):
            # sums the values down each column
            # divides by the number of vectors
            tf[i] = document.sum(axis=0) / document.shape[0]
        return tf
    
    def _inverseDocumentFrequency(self, data: list[npt.NDArray[np.bool_]]) -> npt.NDArray[np.float64]:
        """
        Calculates the inverse document frequency of the data
        """
        # calculate the inverse document frequency
        idf = np.zeros((len(data), data[0].shape[1]))
        # for each column
        for i in range(data[0].shape[1]):
            # calculate the number of documents that contain the token
            for doc in data:
                idf[:, i] += np.count_nonzero(doc[:, i])
        # divide by the number of documents
        # take the log of the result
        idf: npt.NDArray[np.float64] = np.log(len(data) / (1+idf))
        # add 1 to each value
        idf += 1
        return idf