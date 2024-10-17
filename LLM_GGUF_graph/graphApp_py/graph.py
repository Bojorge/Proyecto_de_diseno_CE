# graph.py

class Tensor:
    def __init__(self, name, dimensions, data_type, storage_type):
        self.name = name
        self.dimensions = dimensions  # List of integers representing dimensions (e.g., [64, 64])
        self.data_type = data_type    # Data type of the tensor (e.g., float32, int8)
        self.storage_type = storage_type  # Storage type (e.g., CPU, GPU)

    def __repr__(self):
        return f"Tensor(name={self.name}, dimensions={self.dimensions}, data_type={self.data_type}, storage_type={self.storage_type})"


class Node:
    def __init__(self, node_id, operation, inputs, output):
        self.id = node_id                   # Unique ID of the node
        self.operation = operation          # Operation type (e.g., MatMul, Conv2D, etc.)
        self.inputs = inputs                # List of input tensors
        self.output = output                # Output tensor
        self.predecessors = []              # IDs of predecessor nodes
        self.successors = []                # IDs of successor nodes

    def __repr__(self):
        return f"Node(id={self.id}, operation={self.operation}, inputs={self.inputs}, output={self.output})"


class Graph:
    def __init__(self):
        self.nodes = {}          # Dictionary mapping node ID to node object
        self.next_node_id = 0     # Next available node ID

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, from_node_id, to_node_id):
        self.nodes[from_node_id].successors.append(to_node_id)
        self.nodes[to_node_id].predecessors.append(from_node_id)

    def remove_node(self, node_id):
        if node_id in self.nodes:
            del self.nodes[node_id]

        # Remove references to the node in other nodes' successor and predecessor lists
        for node in self.nodes.values():
            node.successors = [succ for succ in node.successors if succ != node_id]
            node.predecessors = [pred for pred in node.predecessors if pred != node_id]

    def get_nodes(self):
        return list(self.nodes.values())

    def print_graph(self):
        for node in self.get_nodes():
            print(f"Node ID: {node.id} | Operation: {node.operation}")
            print(f"Inputs: {', '.join([tensor.name for tensor in node.inputs])}")
            print(f"Output: {node.output.name}")
            print(f"Successors: {node.successors}")
            print(f"Predecessors: {node.predecessors}\n")
