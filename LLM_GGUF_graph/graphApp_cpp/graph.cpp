#include "graph.h"

// Inicializa el grafo
void initGraph(Graph& graph) {
    graph.next_node_id = 1; // Inicializa el ID en 1, 0 es para entradas que no se toman
}

// Función para agregar un nodo al grafo
void addNode(Graph& graph, const std::vector<int>& in_ids, 
             const struct gguf_tensor_info& in1, 
             const struct gguf_tensor_info& in2, 
             const struct gguf_tensor_info& out, 
             const std::string& operation) {
    // Crear un nuevo nodo
    int node_id = graph.next_node_id++; // Asignar el ID actual y luego incrementarlo

    // Inicializar el nuevo nodo como un struct
    Node new_node;
    new_node.id = node_id;
    new_node.input_ids = in_ids; 
    new_node.input1_tensor = in1;
    new_node.input2_tensor = in2;
    new_node.output_tensor = out;
    new_node.operation = operation;

    // Establecer las formas a partir de los tensores de entrada
    if (in1.n_dims == 1) {
        new_node.shape_in1.push_back(in1.ne[0]); // Para 1D, solo un tamaño
    } 
    else if (in1.n_dims == 2) {
        new_node.shape_in1.push_back(in1.ne[0]); // Filas
        new_node.shape_in1.push_back(in1.ne[1]); // Columnas
    }
    
    if (in2.n_dims == 1) {
        new_node.shape_in2.push_back(in2.ne[0]); // Para 1D, solo un tamaño
    } 
    else if (in2.n_dims == 2) {
        new_node.shape_in2.push_back(in2.ne[0]); // Filas
        new_node.shape_in2.push_back(in2.ne[1]); // Columnas
    }

    // Establecer la forma del tensor de salida
    if (out.n_dims == 1) {
        new_node.shape_out.push_back(out.ne[0]); // Para 1D, solo un tamaño
    } 
    else if (out.n_dims == 2) {
        new_node.shape_out.push_back(out.ne[0]); // Filas
        new_node.shape_out.push_back(out.ne[1]); // Columnas
    }

    // Agregar el nodo al grafo
    graph.nodes.push_back(new_node);
}

// Función para obtener todos los nodos
const std::vector<Node>& getNodes(const Graph& graph) {
    return graph.nodes;
}
