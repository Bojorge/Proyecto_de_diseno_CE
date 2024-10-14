#include "graph.h"

// Implementación de la función para añadir un nodo al grafo
void Graph::add_node(const Node& node) {
    nodes[node.name] = node;
}

// Implementación de la función para obtener todos los nodos
const std::unordered_map<std::string, Node>& Graph::get_nodes() const {
    return nodes;
}
