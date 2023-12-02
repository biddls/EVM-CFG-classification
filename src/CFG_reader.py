import json
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt


class CFG_Reader:
    parsedOpcodes: list[list[str]] = []
    graph: nx.DiGraph | None = None

    def __init__(self, genGraph: bool = False) -> None:
        # Load JSON data from file
        with open("./src/ControlFlowGraphs/file.json", "r") as json_file:
            data = json.load(json_file)

        if genGraph:
            # Create a directed graph
            graph = nx.DiGraph()

            # Add nodes
            for node in tqdm(data["runtimeCfg"]["nodes"]):
                graph.add_node(
                    node["offset"],
                    type=node["type"],
                    bytecodeHex=node["bytecodeHex"],
                    parsedOpcodes=node["parsedOpcodes"],
                )

            # Add edges
            for successor in tqdm(data["runtimeCfg"]["successors"]):
                from_node = successor["from"]
                to_nodes = successor["to"]
                for to_node in to_nodes:
                    graph.add_edge(from_node, to_node)

        nodes = data["runtimeCfg"]["nodes"]
        parsedOpcodes: list[str] = [
            node["parsedOpcodes"].split("\n") for node in nodes
            ]

        for i in range(len(parsedOpcodes)):
            temp = [
                opcode.split(" ", maxsplit=1)[1] for opcode in parsedOpcodes[i]
            ]
            self.parsedOpcodes.append(temp)

    def drawGraph(self) -> None:
        if self.graph is None:
            raise Exception("Graph not generated")
        # Draw the graph
        pos = nx.spring_layout(self.graph)  # You can use other layouts as well
        nx.draw(
            self.graph,
            pos,
            node_size=10,
            node_color="skyblue",
            arrowsize=10)
        plt.show()


if __name__ == "__main__":
    cfg = CFG_Reader()
    print(cfg.parsedOpcodes[0])
