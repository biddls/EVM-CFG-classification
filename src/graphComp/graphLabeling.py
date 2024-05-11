from tqdm import tqdm
from itertools import permutations, combinations, product
from math import factorial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np
from numpy import typing as npt
from graphComp.graphLoading import graphLoader
from icecream import ic
from typing import Literal, Callable
from collections import Counter



class graphLabeling(graphLoader):
    def getGraphSimilarity(
        self,
        nodes1: list[int],
        nodes2: list[int],
        row: Literal["tf_idf", "average", "lstm", ""],
        col: Literal["cosine_similarity_np", "euclideanDistance", "dotProduct", ""]
    ) -> npt.NDArray[np.float_]:
        """
        Get the similarity between two graphs
        """
        # if row == "" and col != "":
        #     raise ValueError("You cannot spesify a column without a row")

        # get the similarity between the nodes
        if row == "" and col == "":
            results = np.zeros((3, 3))
            temp1tf_idf = self.tf_idf[nodes1]
            temp2 = self.tf_idf[nodes2]
            results[0,0] = self.getSimmilarityMatrix(self._cosine_similarity_np, (temp1tf_idf, temp2))
            result = self.getSimmilarityMatrix(self._euclideanDistance, (temp1tf_idf, temp2))
            results[0,1] = 1/result
            results[0,2] = self.getSimmilarityMatrix(self._dotProduct, (temp1tf_idf, temp2))
            temp1average = self.average[nodes1]
            temp2 = self.average[nodes2]            
            results[1,0] = self.getSimmilarityMatrix(self._cosine_similarity_np, (temp1average, temp2))
            result = self.getSimmilarityMatrix(self._euclideanDistance, (temp1average, temp2))
            results[1,1] = 1/result
            results[1,2] = self.getSimmilarityMatrix(self._dotProduct, (temp1average, temp2))
            temp1lstm = self.lstm[nodes1]
            temp2 = self.lstm[nodes2]
            results[2,0] = self.getSimmilarityMatrix(self._cosine_similarity_np, (temp1lstm, temp2))
            result = self.getSimmilarityMatrix(self._euclideanDistance, (temp1lstm, temp2))
            results[2,1] = 1/result
            results[2,2] = self.getSimmilarityMatrix(self._dotProduct, (temp1lstm, temp2))
        elif col == "":
            results = np.zeros((1, 3))
            temp1 = eval(f"self.{row}")[nodes1]
            temp2 = eval(f"self.{row}")[nodes2]
            for i, _col in enumerate(["cosine_similarity_np", "euclideanDistance", "dotProduct"]):
                result  = self.getSimmilarityMatrix(eval(f"self._{_col}"), (temp1, temp2))
                if _col == "euclideanDistance":
                    result = 1/result
                results[0, i] = result
        elif row == "":
            results = np.zeros((3, 1))
            evalFunc = eval(f"self._{col}")
            for i, _row in enumerate(["tf_idf", "average", "lstm"]):
                temp1 = eval(f"self.{_row}")[nodes1]
                temp2 = eval(f"self.{_row}")[nodes2]
                result = self.getSimmilarityMatrix(evalFunc, (temp1, temp2))
                if col == "euclideanDistance":
                    result = 1/result
                results[i, 0] = result
        else:
            results = np.zeros((1, 1))
            temp1 = eval(f"self.{row}")[nodes1]
            temp2 = eval(f"self.{row}")[nodes2]
            results[0, 0] = self.getSimmilarityMatrix(eval(f"self._{col}"), (temp1, temp2))
        return results.flatten()

    def getSimmilarityMatrix(
        self,
        similarityMatrix: Callable[[npt.NDArray[np.int_], npt.NDArray[np.int_]], npt.NDArray[np.float_]],
        pairs: tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]
    ) -> float:
        return float(np.average(similarityMatrix(*pairs)))

    def getGraphLabelsV1(self):
        """
        generates the labels for the graphs
        """
        labeledCFGs = [cfg for cfg in self.CFGs if cfg.label != "unknown"]
        pairs = combinations(range(len(labeledCFGs)), 2)
        lenPairs = int(factorial(len(labeledCFGs))/factorial(len(labeledCFGs)-2))
        pairs = tqdm(
            pairs,
            desc="Generating the similiarity matrix",
            total=int(lenPairs/2),
            ncols=0,
            mininterval=1)

        similarityMatrix = np.zeros((len(labeledCFGs), len(labeledCFGs)))
        similarityMatrix = np.tile(similarityMatrix[:, :, np.newaxis], (1, 1, 9))

        for pair1, pair2 in pairs:
            graph1 = labeledCFGs[pair1]
            graph2 = labeledCFGs[pair2]
            nodes1: list[int] = [x[1] for x in graph1.graph.nodes(data='extIndex')] # type: ignore
            nodes2: list[int] = [x[1] for x in graph2.graph.nodes(data='extIndex')] # type: ignore
            simliaties = self.getGraphSimilarity(nodes1, nodes2, "", "")
            similarityMatrix[pair1, pair2] = simliaties

        similarityMatrix += np.rot90(np.fliplr(similarityMatrix))

        labels = [cfg.label for cfg in labeledCFGs]
        defiIndces = [i for i, x in enumerate(labels) if x == "defi"]
        nftIndces = [i for i, x in enumerate(labels) if x == "nft"]
        erc20Indces = [i for i, x in enumerate(labels) if x == "erc20"]

        Indces = defiIndces + nftIndces + erc20Indces
        trueLabels = [labeledCFGs[i].label for i in Indces]

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
                axisIndex.append([y, x])

        predLookup = ["defi", "nft", "erc20"]
        for d, label in enumerate(labels):

            predLabels = list()
            for i in Indces:
                defiSim  = np.average(similarityMatrix[i, defiIndces, d])
                nftSim  = np.average(similarityMatrix[i, nftIndces, d])
                erc20Sim  = np.average(similarityMatrix[i, erc20Indces, d])
                pred = np.argmax([defiSim, nftSim, erc20Sim])

                # write the maximum to the cfg
                predLabels.append(predLookup[pred])
            confMatrix = confusion_matrix(trueLabels, predLabels, labels=predLookup)

            disp = ConfusionMatrixDisplay(confMatrix, display_labels=predLookup)
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

    def propagateLabelsV1(self):
        """
        Propogates the labels
        """
        # for cfg in self.CFGs:
        #     ic(cfg.label)
        # ic(len(self.CFGs))
        nftIndex = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "nft"]
        erc20Index = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "erc20"]
        defiIndex = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "defi"]
        unlabeledIndex = [*nftIndex[-10:], *erc20Index[-10:], *defiIndex[-10:]]
        nftIndex = nftIndex[:-10]
        erc20Index = erc20Index[:-10]
        defiIndex = defiIndex[:-10]
        # ic(len(nftIndex), len(erc20Index), len(defiIndex), len(unlabeledIndex))
        # unlabeledIndex = tqdm(unlabeledIndex, desc="Propogating labels", ncols=0)
        for row in ['']:#['tf_idf', 'average', 'lstm']:
            for col in ['cosine_similarity_np', 'euclideanDistance', 'dotProduct']:
                accuracy: list[bool] = list()
                for cfg in unlabeledIndex:
                    graph1 = self.CFGs[cfg]
                    nodes1: list[int] = [x[1] for x in graph1.graph.nodes(data='extIndex')] # type: ignore
                    nodes1 = [x for x in nodes1 if x != -1]
                    # get the similarity between the nodes for the NFTs
                    # row: Literal['tf_idf', 'average', 'lstm', ''] = 'lstm'
                    # col: Literal['cosine_similarity_np', 'euclideanDistance', 'dotProduct', ''] = 'euclideanDistance'

                    nftSimliaties = list()
                    for nft in nftIndex:
                        graph2 = self.CFGs[nft]
                        nodes2: list[int] = [x[1] for x in graph2.graph.nodes(data='extIndex')] # type: ignore
                        nodes2 = [x for x in nodes2 if x != -1]
                        simliaties = self.getGraphSimilarity(nodes1, nodes2, row, col) # type: ignore
                        nftSimliaties.append(simliaties)

                    # get the similarity between the nodes for the ERC20s
                    erc20Simliaties = list()
                    for erc20 in erc20Index:
                        graph2 = self.CFGs[erc20]
                        nodes2: list[int] = [x[1] for x in graph2.graph.nodes(data='extIndex')] # type: ignore
                        nodes2 = [x for x in nodes2 if x != -1]
                        simliaties = self.getGraphSimilarity(nodes1, nodes2, row, col) # type: ignore
                        erc20Simliaties.append(simliaties)

                    # get the similarity between the nodes for the defi
                    defiSimliaties = list()
                    for defi in defiIndex:
                        graph2 = self.CFGs[defi]
                        nodes2: list[int] = [x[1] for x in graph2.graph.nodes(data='extIndex')] # type: ignore
                        nodes2 = [x for x in nodes2 if x != -1]
                        simliaties = self.getGraphSimilarity(nodes1, nodes2, row, col) # type: ignore
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
                    # ic(pred, preds)
                    label = ["nft", "erc20", "defi"][pred]
                    accuracy.append(label == self.CFGs[cfg].label)
                    # self.CFGs[cfg].label = label

                # nftIndex = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "nft"]
                # erc20Index = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "erc20"]
                # defiIndex = [i for i, cfg in enumerate(self.CFGs) if cfg.label == "defi"]
                ic(np.average(accuracy), row, col)
                # ic(len(nftIndex), len(erc20Index), len(defiIndex))

    def getGrapgLabelsV2(self, nftIndex, erc20Index, defiIndex, importanceTable):
        """
        Propogates the labels
        """
        labels: tuple[str, str, str] = ("defi", "nft", "erc20")
        indexs = defiIndex + nftIndex + erc20Index
        trueLabels = list()
        predLabels = list()

        for i, cfg in enumerate(indexs):
            graph = self.CFGs[cfg]
            nodes: list[int] = [x[1] for x in graph.graph.nodes(data='extIndex')] # type: ignore
            nodes = [x for x in nodes if x != -1]

            similiartyTable = importanceTable[:, nodes]
            similiartyTable = np.sum(similiartyTable, axis=1)

            pred = np.argmax(similiartyTable)
            # # ic(pred, preds)
            predLabel = labels[pred]
            label = labels[i//10]
            # ic(label, label == predLabel)
            indexs[i] = label == predLabel
            trueLabels.append(label)
            predLabels.append(predLabel)

        confMatrix = confusion_matrix(trueLabels, predLabels, labels=labels)

        disp = ConfusionMatrixDisplay(confMatrix, display_labels=labels)

        disp.plot()
        plt.savefig(f'confusion_matrix.png')

        ic(np.average(indexs))
        ic(np.average(indexs[:10]), np.average(indexs[10:20]), np.average(indexs[20:30]))

    def propagateLabelsV2(self, importanceTable):
        """
        Propogates the labels
        """
        labels: tuple[str, str, str] = ("defi", "nft", "erc20")
        predLabels = list()
        predGraphs = [x for x in self.CFGs if x.label == "unknown"]
        for graph in predGraphs:
            nodes: list[int] = [x[1] for x in graph.graph.nodes(data='extIndex')] # type: ignore
            nodes = [x for x in nodes if x != -1]

            similiartyTable = importanceTable[:, nodes]
            similiartyTable = np.sum(similiartyTable, axis=1)

            pred = np.argmax(similiartyTable)
            predLabel = labels[pred]
            predLabels.append(predLabel)

        ic(Counter(predLabels))
        # trueLabels = [cfg.label for cfg in self.CFGs]
        # ic(Counter(trueLabels))
