#include "graph_builder.h"











// Función que construye un grafo a partir de un gguf_context
struct Graph build_graph_from_context(const struct gguf_context* ctx) {
    Graph graph;

    // Asegúrate de que el contexto no sea nulo
    if (ctx == nullptr) {
        std::cerr << "Error: Context is null." << std::endl;
        return graph; // Devuelve un grafo vacío
    }

   
    // Inicializa el grafo
    initGraph(graph);
    

    // Finalmente, devuelve el grafo completo
    return graph;
}

// Función que imprime el grafo
void print_graph(const struct Graph& graph) {
    for (const Node& node : graph.nodes) {
        std::cout << "Node ID: " << node.id << std::endl;
        std::cout << "Operation: " << node.operation << std::endl;

        std::cout << "Input IDs: ";
        for (int input_id : node.input_ids) {
            std::cout << input_id << " ";
        }
        std::cout << std::endl;

        std::cout << "Input Tensor 1 Shape: ";
        for (int dim : node.shape_in1) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "Input Tensor 2 Shape: ";
        for (int dim : node.shape_in2) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "Output Tensor Shape: ";
        for (int dim : node.shape_out) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "-----------------------------" << std::endl;
    }
}
