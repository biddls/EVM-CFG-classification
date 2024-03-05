import networkx as nx
import numpy as np
from numpy import typing as npt
from matplotlib import pyplot as plt
from CFG_reader import CFG_Reader
import pandas as pd
import json
from tqdm import tqdm
# nx.graph_edit_distance

from difflib import SequenceMatcher

def similar(a: str, b: str) -> float:
    similarity = SequenceMatcher(
        lambda x: x == " ",
        a,
        b
    ).ratio()

    return similarity


class graphClassification:
    """
    Class for labeling graph
    """
    CFGs: list[CFG_Reader]
    pathToTypes: str
    pathToLabels: str
    tf_idf: npt.NDArray
    average: npt.NDArray
    lstm: npt.NDArray
    df: pd.DataFrame

    def __init__(
        self,
        CFGs: list[CFG_Reader],
        pathToTags: str,
        pathToLabels: str,
        tf_idf: npt.NDArray | None = None,
        average: npt.NDArray | None = None,
        lstm: npt.NDArray | None = None,
        ) -> None:
        # Load in all data
        self.CFGs = CFGs
        self.pathToTypes = pathToTags
        self.pathToLabels = pathToLabels
        if tf_idf is not None:
            self.tf_idf = tf_idf
        if average is not None:
            self.average = average
        if lstm is not None:
            self.lstm = lstm

        # load in classes and labels
        self.loadClasses()

    def loadClasses(self):
        """
        Load classes from file
        """
        # Load labels from Json
        with open(self.pathToLabels, "r") as f:
            tempDict: dict[str,list[str]] = json.load(f)

        # reverse the labels
        labels: dict[str, str] = {subV: k for k, v in tempDict.items() for subV in v}

        # load CSV
        df = pd.read_csv(self.pathToTypes)
        df.drop(["source"], inplace=True, axis=1)
        df = df.dropna(subset=["tag"])

        print(df.head())
        print(f"Shape of df: {df.shape}")

        # Process it to get classes
        tags: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            if row['address'] in tags:
                tags[row['address']].append(row['tag'])
            else:
                tags[row['address']] = [row['tag']]

        print(f"Number of unique addresses with tags: {len(tags)}")

        count = 0
        for _tags in tags.values():
            for _tag in _tags:
                if _tag in labels.keys():
                    count += 1
                    break

        print(f"Number of unique addresses with tags in labels: {count}")
        # exit(0)
        addrLabels: dict[str, list[str]] = {}
        # label the tags
        for addr, _tags in tqdm(tags.items()):
            # Get the tags
            _tags = list(set(_tags))
            tags[addr] = _tags

            tempLabels = []
            for tag in _tags:
                try:
                    tempLabels.append(labels[tag])
                except KeyError:
                    ...
                    # similarity = max(list(labels.keys()), key = lambda x: similar(x, tag))
                    # print(f"\n\t## {tag} most similar tag is: {similarity} ##")
                    # print(f"Tag: {tag}")
                    # for i, key in enumerate(tempDict.keys()):
                    #     print(f"{i}: {key}")
                    # choice = input("Enter the index of the chosen label: ")
                    # if choice == "":
                    #     choice = list(tempDict.keys()).index(labels[similarity])
                    # else:
                    #     choice = int(choice)
                    # tempDict[list(tempDict.keys())[choice]].append(tag)
                    # with open(self.pathToLabels, "w") as f:
                    #     json.dump(tempDict, f, indent=4)
                    # exit(0)

            tempLabels = list(set(tempLabels))
            addrLabels[addr] = tempLabels
        
        print(len(addrLabels))

        # Get unique classes
        # tags = df["tag"].unique()
        # print(f"Number of unique tags: {len(tags)}")

        # print(f"Shape of df: {df.shape}")
        # print(df.head())

        # self.df = df
        raise NotImplementedError
