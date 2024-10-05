# extract_graph.py

from graph import Graph, Node, Tensor

def simplify_graph(graph):
    """
    Simplifies the graph by applying optimization rules
    """
    nodes_to_remove = []
    for node in graph.get_nodes():
        if node.operation == "duplicate":
            print(f"Eliminating redundant node with operation: {node.operation}")
            nodes_to_remove.append(node.id)

    for node_id in nodes_to_remove:
        graph.remove_node(node_id)

def list_accelerators(graph):
    """
    Lists compatible accelerators based on the operations in the graph
    """
    print("List of compatible accelerators:")
    for node in graph.get_nodes():
        if node.operation == "MatMul":
            print(f" - Operation: {node.operation} can be accelerated using GPUs or specialized hardware (e.g., TPUs)")

def load_gguf(filename):
    """
    Loads a GGUF file and builds the graph.
    This is a simplified placeholder for demonstration purposes.
    """
    graph = Graph()
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if "Node" in line:
                operation = "MatMul"  # Placeholder, this should be extracted from the file
                inputs = [Tensor("input_placeholder", [4096, 4096], "float32", "CPU")]  # Example placeholder
                output = Tensor("output_placeholder", [4096, 4096], "float32", "CPU")
                node = Node(graph.next_node_id, operation, inputs, output)
                graph.add_node(node)
                graph.next_node_id += 1

    except FileNotFoundError:
        print(f"Error: Unable to open file {filename}")
    
    return graph
