import numpy as np
from numpy import typing as npt
from CFG_reader import CFG_Reader
import pandas as pd
import json
from tqdm import tqdm
from itertools import permutations
from math import factorial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt



class graphClassification:
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
            self.tf_idfVectors_cosine_similarity_np = self.cosine_similarity_np(_tf_idf)
            self.tf_idfVectors_euclideanDistance = self.euclideanDistance(_tf_idf)
            self.tf_idfVectors_dotProduct = self.dotProduct(_tf_idf)
        if _average is not None:
            self.average = _average
            self.averageVectors_cosine_similarity_np = self.cosine_similarity_np(_average)
            self.averageVectors_euclideanDistance = self.euclideanDistance(_average)
            self.averageVectors_dotProduct = self.dotProduct(_average)
        if _lstm is not None:
            self.lstm = _lstm
            self.lstmVectors_cosine_similarity_np = self.cosine_similarity_np(_lstm)
            self.lstmVectors_euclideanDistance = self.euclideanDistance(_lstm)
            self.lstmVectors_dotProduct = self.dotProduct(_lstm)

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

    def getGraphSimilarity(self, CFG_1: CFG_Reader, CFG_2: CFG_Reader) -> npt.NDArray[np.float_]:
        """
        Get the similarity between two graphs
        """
        # Get the nodes
        nodes1: list[int] = [x[1] for x in CFG_1.graph.nodes(data='extIndex')] # type: ignore
        nodes2: list[int] = [x[1] for x in CFG_2.graph.nodes(data='extIndex')] # type: ignore

        # get the similarity between the nodes using tf-idf
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
            simliaties = self.getGraphSimilarity(graph1, graph2)
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

    def cosine_similarity_np(self, matrix: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        dot_product = np.dot(matrix, matrix.T)
        norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
        cosine_similarities = dot_product / (norm_matrix * norm_matrix.T)
        # apply cutoff
        # cosine_similarities = cosine_similarities > cutoff
        return cosine_similarities

    def euclideanDistance(self, matrix: np.ndarray) -> np.ndarray:
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
    
    def dotProduct(self, matrix: np.ndarray) -> np.ndarray:
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
