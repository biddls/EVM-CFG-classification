from collections import Counter
import numpy as np
import numpy.typing as npt
from tokeniser import Tokeniser


class Average:
    data: list[npt.NDArray[np.bool_]]
    width: int

    def __init__(self, data: Counter[tuple[int | tuple[int, int]]]):
        """
        Shape of the data:
        first level is the list of cfgs (documents)
        The second level is the list of nodes
        The third level is the vector representation of the node
        """
        temp = list(data.keys())
        temp = list(map(Tokeniser.vectoriseNode, list(temp)))  # type: ignore
        self.data = temp
        self.width = self.data[0].shape[1]

    def __call__(self, *args, **kwargs) -> npt.NDArray[np.float64]:
        return self._average()

    def _average(self) -> npt.NDArray[np.float64]:
        """
        Calculates the average of the data
        """
        average = np.zeros((len(self.data), self.width))
        for i, document in enumerate(self.data):
            average[i] = document.sum(axis=0) / document.shape[0]
        return average
