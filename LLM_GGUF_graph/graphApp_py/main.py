# main.py

from graph import Graph, Tensor, Node
from extract_graph import simplify_graph, list_accelerators, load_gguf

def main():
    # Create a simple graph manually
    graph = Graph()

    # Create example tensors
    input_tensor = Tensor("input1", [4096, 4096], "float32", "CPU")
    output_tensor = Tensor("output1", [4096, 4096], "float32", "CPU")

    # Create a MatMul node
    matmul_node = Node(graph.next_node_id, "MatMul", [input_tensor], output_tensor)
    graph.add_node(matmul_node)
    graph.next_node_id += 1

    # Add another node to simulate a more complex graph
    input_tensor2 = Tensor("input2", [4096, 4096], "float32", "CPU")
    output_tensor2 = Tensor("output2", [4096, 4096], "float32", "CPU")
    another_node = Node(graph.next_node_id, "MatMul", [input_tensor2], output_tensor2)
    graph.add_node(another_node)
    graph.next_node_id += 1

    # Print initial graph
    print("Initial Graph:")
    graph.print_graph()

    # Simplify the graph
    simplify_graph(graph)

    # Print simplified graph
    print("\nSimplified Graph:")
    graph.print_graph()

    # List accelerators for the operations in the graph
    list_accelerators(graph)

    # Load a GGUF file (this is a placeholder function for now)
    print("\nLoading graph from GGUF file...")
    graph_from_file = load_gguf("example.gguf")
    print("Graph loaded from GGUF:")
    graph_from_file.print_graph()

if __name__ == "__main__":
    main()
