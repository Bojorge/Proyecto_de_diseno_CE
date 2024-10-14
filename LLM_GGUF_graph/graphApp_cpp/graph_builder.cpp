#include "graph_builder.h"

// Función que construye un grafo a partir de un gguf_context
Graph build_graph_from_context(const struct gguf_context* ctx) {
    Graph graph;

    // Ejemplo de cómo podrías extraer nodos desde el contexto gguf_context
    // Aquí, solo estamos añadiendo nodos hipotéticamente.
    // La lógica de extracción real dependería de cómo se estructuran los datos en gguf_context.

    // Nodo 1: RMS Norm
    Node rms_norm;
    rms_norm.name = "RMS Norm";
    rms_norm.operation = "RMS Norm";
    rms_norm.inputs = {"input"};  // Vector de entrada de tamaño 4096x1
    rms_norm.shape_in = {4096, 1};
    rms_norm.shape_out = {4096, 1};
    graph.add_node(rms_norm);

    // Nodo 2: EW Multiply
    Node ew_multiply;
    ew_multiply.name = "EW Multiply";
    ew_multiply.operation = "Element-Wise Multiply";
    ew_multiply.inputs = {"RMS Norm"};  // Toma la salida del nodo anterior
    ew_multiply.shape_in = {4096, 1};
    ew_multiply.shape_out = {4096, 1};
    graph.add_node(ew_multiply);

    // Y así sucesivamente, añadirías todos los nodos correspondientes a las operaciones en el grafo

    // Finalmente, devuelve el grafo completo
    return graph;
}

// Función que imprime el grafo
void print_graph(const Graph& graph) {
    for (const auto& pair : graph.get_nodes()) {
        const Node& node = pair.second;
        std::cout << "Node: " << node.name << "\n";
        std::cout << "Operation: " << node.operation << "\n";
        std::cout << "Inputs: ";
        for (const auto& input : node.inputs) {
            std::cout << input << " ";
        }
        std::cout << "\nOutputs: ";
        for (const auto& output : node.outputs) {
            std::cout << output << " ";
        }
        std::cout << "\n";
        std::cout << "Input Shape: [" << node.shape_in[0] << ", " << node.shape_in[1] << "]\n";
        std::cout << "Output Shape: [" << node.shape_out[0] << ", " << node.shape_out[1] << "]\n";
        std::cout << "------------------------------------\n";
    }
}
