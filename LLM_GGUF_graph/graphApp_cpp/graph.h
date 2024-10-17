#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>
#include "structures.h" // Asegúrate de que este archivo esté correctamente incluido

// Definición de un nodo en el grafo (una operación)
struct Node {
    int id;                                    // Identificador único del nodo
    std::vector<int> input_ids;                // IDs de los nodos de entrada
    struct gguf_tensor_info input1_tensor;     // Tensor de entrada 1
    struct gguf_tensor_info input2_tensor;     // Tensor de entrada 2 (puede ser vacío)
    struct gguf_tensor_info output_tensor;     // Tensor de salida
    std::vector<std::string> inputs;           // Nombres de los nodos de entrada
    std::vector<std::string> outputs;          // Nombres de los nodos de salida
    std::string operation;                     // Tipo de operación (Matrix Multiply, EW Multiply, Softmax, etc)
    std::vector<int> shape_in1;                // Forma del tensor de entrada 1
    std::vector<int> shape_in2;                // Forma del tensor de entrada 2
    std::vector<int> shape_out;                // Forma del tensor de salida
};

// Estructura para representar el grafo
struct Graph {
    std::vector<Node> nodes; // Contenedor para almacenar los nodos
    int next_node_id;        // ID del siguiente nodo a agregar
};

// Inicializa el grafo
void initGraph(Graph& graph);

// Función para agregar un nodo al grafo
void addNode(Graph& graph, const std::vector<int>& in_ids, 
             const struct gguf_tensor_info& in1, 
             const struct gguf_tensor_info& in2, 
             const struct gguf_tensor_info& out, 
             const std::string& operation);

// Función para obtener todos los nodos
const std::vector<Node>& getNodes(const Graph& graph);

#endif // GRAPH_H
