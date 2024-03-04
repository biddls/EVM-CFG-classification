import networkx as nx
import numpy as np
from numpy import typing as npt
from matplotlib import pyplot as plt
from CFG_reader import CFG_Reader
import pandas as pd
# nx.graph_edit_distance


class graphClassification:
    """
    Class for labeling graph
    """
    CFGs: list[CFG_Reader]
    pathToTypes: str
    tf_idf: npt.NDArray
    average: npt.NDArray
    lstm: npt.NDArray
    df: pd.DataFrame

    def __init__(
        self,
        CFGs: list[CFG_Reader],
        pathToTypes: str,
        tf_idf: npt.NDArray | None = None,
        average: npt.NDArray | None = None,
        lstm: npt.NDArray | None = None,
        ) -> None:
        # Load in all data
        self.CFGs = CFGs
        self.pathToTypes = pathToTypes
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
        # load CSV
        df = pd.read_csv(self.pathToTypes)
        print(df.head())
        print(f"Shape of df: {df.shape}")

        # Process it to get classes
        df = df.dropna(subset=["tag"])

        # Get unique classes
        classes = df["tag"].unique()
        print(f"Number of unique classes: {len(classes)}")

        print(f"Shape of df: {df.shape}")
        print(df.head())
        
        self.df = df