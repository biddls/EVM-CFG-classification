import json
from operator import ge
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt


class CFG_Reader:
    parsedOpcodes: list[str] = []
    graph: nx.DiGraph

    # Load JSON data from file
    with open("./src/ControlFlowGraphs/file.json", "r") as json_file:
        data = json.load(json_file)

    def __init__(self, genGraph=False) -> None:
        self.graphGenerated = genGraph
        if genGraph:
            # Create a directed graph
            graph = nx.DiGraph()

            # Add nodes
            for node in tqdm(self.data["runtimeCfg"]["nodes"]):
                graph.add_node(
                    node["offset"],
                    type=node["type"],
                    bytecodeHex=node["bytecodeHex"],
                    parsedOpcodes=node["parsedOpcodes"],
                )

            # Add edges
            for successor in tqdm(self.data["runtimeCfg"]["successors"]):
                from_node = successor["from"]
                to_nodes = successor["to"]
                for to_node in to_nodes:
                    graph.add_edge(from_node, to_node)
        else:
            nodes = self.data["runtimeCfg"]["nodes"]
            self.parsedOpcodes = [node["parsedOpcodes"] for node in nodes]

    def drawGraph(self) -> None:
        if not self.graphGenerated:
            raise Exception("Graph not generated")
        # Draw the graph
        pos = nx.spring_layout(self.graph)  # You can use other layouts as well
        nx.draw(self.graph, pos, node_size=10, node_color="skyblue", arrowsize=10)
        plt.show()
