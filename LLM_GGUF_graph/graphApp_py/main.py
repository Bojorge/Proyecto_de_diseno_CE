# main.py

from graph import Graph, Tensor, Node
from extract_graph import simplify_graph, list_accelerators, load_gguf

def main():
    
    graph_from_file = load_gguf("law-llm.Q2_K.gguf")

if __name__ == "__main__":
    main()
