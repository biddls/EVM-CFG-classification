import json
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load JSON data from file
with open('file.json', 'r') as json_file:
    data = json.load(json_file)

# Create a directed graph
graph = nx.DiGraph()

# Add nodes
for node in tqdm(data['runtimeCfg']['nodes']):
    graph.add_node(node['offset'], type=node['type'], bytecodeHex=node['bytecodeHex'], parsedOpcodes=node['parsedOpcodes'])

# Add edges
for successor in tqdm(data['runtimeCfg']['successors']):
    from_node = successor['from']
    to_nodes = successor['to']
    for to_node in to_nodes:
        graph.add_edge(from_node, to_node)

# Draw the graph
pos = nx.spring_layout(graph)  # You can use other layouts as well
nx.draw(graph, pos, node_size=10, node_color='skyblue', arrowsize=10)
plt.show()
