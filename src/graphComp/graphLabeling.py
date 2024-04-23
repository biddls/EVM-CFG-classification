from tqdm import tqdm
from itertools import permutations
from math import factorial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np
from numpy import typing as npt
from graphComp.graphLoading import graphLoader

class graphLabelingFirstTry(graphLoader):
    def getGraphSimilarity(self, nodes1: list[int], nodes2: list[int]) -> npt.NDArray[np.float_]:
        """
        Get the similarity between two graphs
        """
        # get the similarity between the nodes using tf-idf
        indexMatrix = np.ix_(nodes1, nodes2)
        print(f"{self.tf_idfVectors_cosine_similarity_np.T.shape = }")
        print(f"{indexMatrix[0].shape = }, {indexMatrix[1].shape = }")
        print(f"{np.max(indexMatrix[0]) = }, {np.max(indexMatrix[1]) = }")
        # exit(0)
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

        # labels = [
        #     "result_matrix_tf_idfVectors_cosine_similarity_np",
        #     "result_matrix_tf_idfVectors_euclideanDistance",
        #     "result_matrix_tf_idfVectors_dotProduct",
        #     "result_matrix_averageVectors_cosine_similarity_np",
        #     "result_matrix_averageVectors_euclideanDistance",
        #     "result_matrix_averageVectors_dotProduct",
        #     "result_matrix_lstmVectors_cosine_similarity_np",
        #     "result_matrix_lstmVectors_euclideanDistance",
        #     "result_matrix_lstmVectors_dotProduct"
        # ]
        # if len(CFG_1.graph.nodes) > 500 and len(CFG_2.graph.nodes) > 500:
        #     for label in labels:
        #         print(f"Calculating {label}")
        #         plt.imshow(eval(label))
        #         # print(f"{similarityMatrix[:, :, d].shape = }")
        #         plt.colorbar()
        #         plt.savefig(f'./similarity_matrix/{CFG_1.addr}-{CFG_2.addr}-{label}_similarity_matrix.png')
        #         plt.close()
        #     exit(0)

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
        pairs = permutations(range(len(self.CFGs)), 2)
        lenPairs = int(factorial(len(self.CFGs))/factorial(len(self.CFGs)-2))
        pairs = tqdm(pairs, desc="Generating the similiarity matrix", total=lenPairs, smoothing=0)

        similarityMatrix = np.identity(len(self.CFGs))
        similarityMatrix = np.tile(similarityMatrix[:, :, np.newaxis], (1, 1, 9))

        # todo: table is mirrored so we only need to calculate half of it
        for pair1, pair2 in pairs:
            graph1 = self.CFGs[pair1]
            graph2 = self.CFGs[pair2]
            nodes1: list[int] = [x[1] for x in graph1.graph.nodes(data='extIndex')] # type: ignore
            nodes2: list[int] = [x[1] for x in graph2.graph.nodes(data='extIndex')] # type: ignore
            print(f"\n{max(max(nodes1), max(nodes2)) = }")
            print(f"{len(nodes1) = }, {len(nodes2) = }")
            simliaties = self.getGraphSimilarity(nodes1, nodes2)
            similarityMatrix[pair1, pair2] = simliaties

        labels = [cfg.label for cfg in self.CFGs]

        defiIndces = [i for i, x in enumerate(labels) if x == "defi"]
        nftIndces = [i for i, x in enumerate(labels) if x == "nft"]
        erc20Indces = [i for i, x in enumerate(labels) if x == "erc20"]
        # print(f"defi: {len(defiIndces)}, nft: {len(nftIndces)}, erc20: {len(erc20Indces)}")

        defiTestIndces = defiIndces[:int(len(defiIndces)/5)]
        defiTrainIndces = defiIndces[int(len(defiIndces)/5):]

        nftTestIndces = nftIndces[:int(len(nftIndces)/5)]
        nftTrainIndces = nftIndces[int(len(nftIndces)/5):]
        
        erc20TestIndces = erc20Indces[:int(len(erc20Indces)/5)]
        erc20TrainIndces = erc20Indces[int(len(erc20Indces)/5):]

        testIndces = defiTestIndces + nftTestIndces + erc20TestIndces
        trueLabels = [self.CFGs[i].label for i in testIndces]

        labels = [
            "tf_idfVectors_cosine_similarity_np",
            "tf_idfVectors_euclideanDistance",
            "tf_idfVectors_dotProduct",
            "averageVectors_cosine_similarity_np",
            "averageVectors_euclideanDistance",
            "averageVectors_dotProduct",
            "lstmVectors_cosine_similarity_np",
            "lstmVectors_euclideanDistance",
            "lstmVectors_dotProduct"
        ]

        for d, label in enumerate(labels):
            print(f"Calculating {label}")

            plt.imshow(similarityMatrix[:, :, d])
            # print(f"{similarityMatrix[:, :, d].shape = }")
            plt.colorbar()
            plt.savefig(f'./similarity_matrix/{label}_similarity_matrix.png')
            plt.close()

            predLabels = list()
            for i in testIndces:
                defiSim  = np.average(similarityMatrix[i, defiTrainIndces, d])
                nftSim  = np.average(similarityMatrix[i, nftTrainIndces, d])
                erc20Sim  = np.average(similarityMatrix[i, erc20TrainIndces, d])

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
            print(confMatrix)

            disp = ConfusionMatrixDisplay(confMatrix, display_labels=["defi", "nft", "erc20"])

            disp.plot()
            plt.savefig(f'./confusion_matrix/{label}_confusion_matrix.png')
            plt.close()

        # make global prediction
        predLabels = list()
        for i in testIndces:
            defiSim  = np.average(similarityMatrix[i, defiTrainIndces])
            nftSim  = np.average(similarityMatrix[i, nftTrainIndces])
            erc20Sim  = np.average(similarityMatrix[i, erc20TrainIndces])

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
        print("total confusion matrix:")
        print(confMatrix)

        disp = ConfusionMatrixDisplay(confMatrix, display_labels=["defi", "nft", "erc20"])

        disp.plot()
        plt.savefig(f'confusion_matrix.png')
