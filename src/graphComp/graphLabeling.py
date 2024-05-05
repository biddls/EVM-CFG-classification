from tqdm import tqdm
from itertools import permutations, combinations, product
from math import factorial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np
from numpy import typing as npt
from graphComp.graphLoading import graphLoader
from icecream import ic
from typing import Literal
#todo: add label propogation


class graphLabeling(graphLoader):
    def getGraphSimilarity(
        self,
        nodes1: list[int],
        nodes2: list[int],
        row: Literal["tf_idf", "average", "lstm", ""],
        column: Literal["cosine_similarity_np", "euclideanDistance", "dotProduct", ""]
    ) -> npt.NDArray[np.float_]:
        """
        Get the similarity between two graphs
        """
        if row == "" and column != "":
            raise ValueError("You cannot spesify a column without a row")

        # get the similarity between the nodes
        indexMatrix = np.ix_(nodes1, nodes2)
        if row == "" and column == "":
            results = np.zeros((3, 3))
            results[0,0] = self.getSimmilarityMatrix(self.tf_idfVectors_cosine_similarity_np, indexMatrix)
            results[0,1] = self.getSimmilarityMatrix(self.tf_idfVectors_euclideanDistance, indexMatrix)
            results[0,2] = self.getSimmilarityMatrix(self.tf_idfVectors_dotProduct, indexMatrix)
            results[1,0] = self.getSimmilarityMatrix(self.averageVectors_cosine_similarity_np, indexMatrix)
            results[1,1] = self.getSimmilarityMatrix(self.averageVectors_euclideanDistance, indexMatrix)
            results[1,2] = self.getSimmilarityMatrix(self.averageVectors_dotProduct, indexMatrix)
            results[2,0] = self.getSimmilarityMatrix(self.lstmVectors_cosine_similarity_np, indexMatrix)
            results[2,1] = self.getSimmilarityMatrix(self.lstmVectors_euclideanDistance, indexMatrix)
            results[2,2] = self.getSimmilarityMatrix(self.lstmVectors_dotProduct, indexMatrix)
        elif column == "":
            results = np.zeros((1, 3))
            for i, _col in enumerate(["cosine_similarity_np", "euclideanDistance", "dotProduct"]):
                results[0, i] = self.getSimmilarityMatrix(eval(f"self.{row}Vectors_{_col}"), indexMatrix)
        else:
            results = np.zeros((1, 1))
            results[0, 0] = self.getSimmilarityMatrix(eval(f"self.{row}Vectors_{column}"), indexMatrix)
        return results.flatten()

    def getSimmilarityMatrix(
        self,
        similarityMatrix: npt.NDArray[np.float_],
        pairs: tuple[npt.NDArray[np.int_], ...]
    ) -> float:
        return float(np.average(similarityMatrix.T[pairs]))

    def getGraphLabels(self):
        """
        generates the labels for the graphs
        """
        pairs = combinations(range(len(self.CFGs)), 2)
        lenPairs = int(factorial(len(self.CFGs))/factorial(len(self.CFGs)-2))
        pairs = tqdm(pairs, desc="Generating the similiarity matrix", total=int(lenPairs/2), ncols=0)

        similarityMatrix = np.identity(len(self.CFGs))
        similarityMatrix = np.tile(similarityMatrix[:, :, np.newaxis], (1, 1, 9))

        for pair1, pair2 in pairs:
            graph1 = self.CFGs[pair1]
            graph2 = self.CFGs[pair2]
            nodes1: list[int] = [x[1] for x in graph1.graph.nodes(data='extIndex')] # type: ignore
            nodes2: list[int] = [x[1] for x in graph2.graph.nodes(data='extIndex')] # type: ignore
            # for i, node in enumerate(nodes1):
            #     if not isinstance(node, int):
            #         ic(node, 1, "Nodes1")
            # for i, node in enumerate(nodes2):
            #     if not isinstance(node, int):
            #         ic(node, 1, "Nodes2")
            simliaties = self.getGraphSimilarity(nodes1, nodes2, "", "")
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

        predLookup = {0: "defi", 1: "nft", 2: "erc20"}
        for d, label in enumerate(labels):

            predLabels = list()
            for i in Indces:
                defiSim  = np.average(similarityMatrix[i, defiIndces, d])
                nftSim  = np.average(similarityMatrix[i, nftIndces, d])
                erc20Sim  = np.average(similarityMatrix[i, erc20Indces, d])
                pred = np.argmax([defiSim, nftSim, erc20Sim])

                # write the maximum to the cfg
                predLabels.append(predLookup[pred])
            confMatrix = confusion_matrix(trueLabels, predLabels, labels=["defi", "nft", "erc20"])

            disp = ConfusionMatrixDisplay(confMatrix, display_labels=["defi", "nft", "erc20"])
            x, y = axisIndex[d]
            disp.plot(ax=axis[x, y], colorbar=False)
            axis[x, y].set_title(label)

        fig.savefig(f'./{self.graphName}')
        plt.close()

        # make global prediction
        predLabels = list()
        for i in Indces:
            defiSim  = np.average(similarityMatrix[i, defiIndces])
            nftSim  = np.average(similarityMatrix[i, nftIndces])
            erc20Sim  = np.average(similarityMatrix[i, erc20Indces])

            pred = np.argmax([defiSim, nftSim, erc20Sim])
            # write the maximum to the cfg
            predLabels.append(predLookup[pred])

        confMatrix = confusion_matrix(trueLabels, predLabels, labels=["defi", "nft", "erc20"])

        disp = ConfusionMatrixDisplay(confMatrix, display_labels=["defi", "nft", "erc20"])

        disp.plot()
        plt.savefig(f'confusion_matrix.png')

    def propagateLabels(self):
        """
        Propogates the labels
        """
        # for cfg in self.CFGs:
        #     ic(cfg.label)
        ic(len(self.CFGs))
        nftIndex = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "nft"]
        erc20Index = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "erc20"]
        defiIndex = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "defi"]
        unlabeledIndex = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "unknown"]
        ic(len(nftIndex), len(erc20Index), len(defiIndex), len(unlabeledIndex))
        # unlabeledIndex = tqdm(unlabeledIndex, desc="Propogating labels", ncols=0)
        for cfg in unlabeledIndex:
            graph1 = self.CFGs[cfg]
            nodes1: list[int] = [x[1] for x in graph1.graph.nodes(data='extIndex')] # type: ignore
            
            # get the similarity between the nodes for the NFTs
            nftSimliaties = list()
            for nft in nftIndex:
                graph2 = self.CFGs[nft]
                nodes2: list[int] = [x[1] for x in graph2.graph.nodes(data='extIndex')] # type: ignore
                simliaties = self.getGraphSimilarity(nodes1, nodes2, "lstm", "cosine_similarity_np")
                nftSimliaties.append(simliaties)
            
            # get the similarity between the nodes for the ERC20s
            erc20Simliaties = list()
            for erc20 in erc20Index:
                graph2 = self.CFGs[erc20]
                nodes2: list[int] = [x[1] for x in graph2.graph.nodes(data='extIndex')] # type: ignore
                simliaties = self.getGraphSimilarity(nodes1, nodes2, "lstm", "cosine_similarity_np")
                erc20Simliaties.append(simliaties)

            # get the similarity between the nodes for the defi
            defiSimliaties = list()
            for defi in defiIndex:
                graph2 = self.CFGs[defi]
                nodes2: list[int] = [x[1] for x in graph2.graph.nodes(data='extIndex')] # type: ignore
                simliaties = self.getGraphSimilarity(nodes1, nodes2, "lstm", "cosine_similarity_np")
                defiSimliaties.append(simliaties)

            nftSimliaties = nftSimliaties or [0]
            erc20Simliaties = erc20Simliaties or [0]
            defiSimliaties = defiSimliaties or [0]

            preds = np.array([
                np.average(nftSimliaties),
                np.average(erc20Simliaties),
                np.average(defiSimliaties)
            ])

            pred = np.argmax(preds)
            ic(pred, preds)
            self.CFGs[cfg].label = ["nft", "erc20", "defi"][pred]

        nftIndex = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "nft"]
        erc20Index = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "erc20"]
        defiIndex = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "defi"]
        ic(len(nftIndex), len(erc20Index), len(defiIndex))
