from tqdm import tqdm
from itertools import permutations, combinations
from math import factorial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np
from numpy import typing as npt
from graphComp.graphLoading import graphLoader

class graphLabeling(graphLoader):
    def getGraphSimilarity(self, nodes1: list[int], nodes2: list[int]) -> npt.NDArray[np.float_]:
        """
        Get the similarity between two graphs
        """
        # get the similarity between the nodes
        indexMatrix = np.ix_(nodes1, nodes2)

        result_matrix_tf_idfVectors_cosine_similarity_np = self.tf_idfVectors_cosine_similarity_np.T[indexMatrix]
        result_matrix_tf_idfVectors_euclideanDistance = self.tf_idfVectors_euclideanDistance.T[indexMatrix]
        result_matrix_tf_idfVectors_dotProduct = self.tf_idfVectors_dotProduct.T[indexMatrix]
        result_matrix_averageVectors_cosine_similarity_np = self.averageVectors_cosine_similarity_np.T[indexMatrix]
        result_matrix_averageVectors_euclideanDistance = self.averageVectors_euclideanDistance.T[indexMatrix]
        result_matrix_averageVectors_dotProduct = self.averageVectors_dotProduct.T[indexMatrix]
        result_matrix_lstmVectors_cosine_similarity_np = self.lstmVectors_cosine_similarity_np.T[indexMatrix]
        result_matrix_lstmVectors_euclideanDistance = self.lstmVectors_euclideanDistance.T[indexMatrix]
        result_matrix_lstmVectors_dotProduct = self.lstmVectors_dotProduct.T[indexMatrix]

        result_tf_idfVectors_cosine_similarity_np = float(np.average(result_matrix_tf_idfVectors_cosine_similarity_np))
        result_tf_idfVectors_euclideanDistance = float(np.average(result_matrix_tf_idfVectors_euclideanDistance))
        result_tf_idfVectors_dotProduct = float(np.average(result_matrix_tf_idfVectors_dotProduct))
        result_averageVectors_cosine_similarity_np = float(np.average(result_matrix_averageVectors_cosine_similarity_np))
        result_averageVectors_euclideanDistance = float(np.average(result_matrix_averageVectors_euclideanDistance))
        result_averageVectors_dotProduct = float(np.average(result_matrix_averageVectors_dotProduct))
        result_lstmVectors_cosine_similarity_np = float(np.average(result_matrix_lstmVectors_cosine_similarity_np))
        result_lstmVectors_euclideanDistance = float(np.average(result_matrix_lstmVectors_euclideanDistance))
        result_lstmVectors_dotProduct = float(np.average(result_matrix_lstmVectors_dotProduct))

        return np.array([
            result_tf_idfVectors_cosine_similarity_np,
            result_tf_idfVectors_euclideanDistance,
            result_tf_idfVectors_dotProduct,
            result_averageVectors_cosine_similarity_np,
            result_averageVectors_euclideanDistance,
            result_averageVectors_dotProduct,
            result_lstmVectors_cosine_similarity_np,
            result_lstmVectors_euclideanDistance,
            result_lstmVectors_dotProduct
        ])

    def getGraphLabels(self):
        """
        generates the labels for the graphs
        """
        pairs = combinations(range(len(self.CFGs)), 2)
        lenPairs = int(factorial(len(self.CFGs))/factorial(len(self.CFGs)-2))
        pairs = tqdm(pairs, desc="Generating the similiarity matrix", total=lenPairs, smoothing=0, ncols=0)

        similarityMatrix = np.identity(len(self.CFGs))
        similarityMatrix = np.tile(similarityMatrix[:, :, np.newaxis], (1, 1, 9))

        for pair1, pair2 in pairs:
            graph1 = self.CFGs[pair1]
            graph2 = self.CFGs[pair2]
            nodes1: list[int] = [x[1] for x in graph1.graph.nodes(data='extIndex')] # type: ignore
            nodes2: list[int] = [x[1] for x in graph2.graph.nodes(data='extIndex')] # type: ignore
            simliaties = self.getGraphSimilarity(nodes1, nodes2)
            similarityMatrix[pair1, pair2] = simliaties

        similarityMatrix += np.rot90(np.fliplr(similarityMatrix))

        labels = [cfg.label for cfg in self.CFGs]

        defiIndces = [i for i, x in enumerate(labels) if x == "defi"]
        nftIndces = [i for i, x in enumerate(labels) if x == "nft"]
        erc20Indces = [i for i, x in enumerate(labels) if x == "erc20"]

        Indces = defiIndces + nftIndces + erc20Indces
        trueLabels = [self.CFGs[i].label for i in Indces]

        labels = [
            "tf-idf | cosine similarity",
            "tf-idf | euclidean distance",
            "tf-idf | dotproduct",
            "average | cosine similarity",
            "average | euclideanDistance",
            "average | dotProduct",
            "lstm | cosine similarity",
            "lstm | euclideanDistance",
            "lstm | dotProduct"
        ]

        # 3 X 3 plot
        fig, axis = plt.subplots(3, 3, figsize=(15, 15))
        axisIndex = list()
        for x in range(3):
            for y in range(3):
                axisIndex.append([x, y])

        for d, label in enumerate(labels):

            predLabels = list()
            for i in Indces:
                defiSim  = np.average(similarityMatrix[i, defiIndces, d])
                nftSim  = np.average(similarityMatrix[i, nftIndces, d])
                erc20Sim  = np.average(similarityMatrix[i, erc20Indces, d])

                # write the maximum to the cfg
                if defiSim > nftSim and defiSim > erc20Sim:
                    predLabels.append("defi")
                    # print(f"defi: {defiSim}")
                elif nftSim > defiSim and nftSim > erc20Sim:
                    predLabels.append("nft")
                    # print(f"nft: {nftSim}")
                elif erc20Sim > defiSim and erc20Sim > nftSim:
                    predLabels.append("erc20")
                    # print(f"erc20: {erc20Sim}")
            # print(f"{len(trueLabels) = }, {len(predLabels) = }")
            confMatrix = confusion_matrix(trueLabels, predLabels, labels=["defi", "nft", "erc20"])
            # print(confMatrix)

            disp = ConfusionMatrixDisplay(confMatrix, display_labels=["defi", "nft", "erc20"])
            # if d == 8:
            #     continue
            x, y = axisIndex[d]
            disp.plot(ax=axis[x, y], colorbar=False)
            axis[x, y].set_title(label)

        fig.savefig(f'./matrix_of_confusion_matrix.png')

        # make global prediction
        predLabels = list()
        for i in Indces:
            defiSim  = np.average(similarityMatrix[i, defiIndces])
            nftSim  = np.average(similarityMatrix[i, nftIndces])
            erc20Sim  = np.average(similarityMatrix[i, erc20Indces])

            # write the maximum to the cfg
            if defiSim > nftSim and defiSim > erc20Sim:
                predLabels.append("defi")
                # print(f"defi: {defiSim}")
            elif nftSim > defiSim and nftSim > erc20Sim:
                predLabels.append("nft")
                # print(f"nft: {nftSim}")
            elif erc20Sim > defiSim and erc20Sim > nftSim:
                predLabels.append("erc20")
                # print(f"erc20: {erc20Sim}")

        confMatrix = confusion_matrix(trueLabels, predLabels, labels=["defi", "nft", "erc20"])

        disp = ConfusionMatrixDisplay(confMatrix, display_labels=["defi", "nft", "erc20"])

        disp.plot()
        plt.savefig(f'confusion_matrix.png')
