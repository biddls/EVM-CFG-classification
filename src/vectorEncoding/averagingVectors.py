import numpy as np
import numpy.typing as npt

class Average:
    def __init__(self, data: list[npt.NDArray[np.bool_]]):
        self.data = data
    
    def __call__(self, *args, **kwargs):
        return self._average(self.data)
    
    def _average(self, data: list[npt.NDArray[np.bool_]]) -> npt.NDArray[np.float64]:
        """
        Calculates the average of the data
        """
        average = np.zeros((len(data), data[0].shape[1]))
        for i, document in enumerate(data):
            average[i] = document.sum(axis=0) / document.shape[0]
        return average