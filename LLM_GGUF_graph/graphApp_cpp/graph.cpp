#include "graph.h"

// Función para crear un tensor
Tensor createTensor(const std::string& name, const std::vector<int>& dimensions, const std::string& data_type, const std::string& storage_type) {
    return {name, dimensions, data_type, storage_type};
}

// Función para crear un nodo
Node createNode(Graph& graph, const std::string& operation, const std::vector<Tensor>& inputs, const Tensor& output) {
    Node node;
    node.id = graph.nextNodeID++;
    node.operation = operation;
    node.inputs = inputs;
    node.output = output;
    return node;
}

// Implementación para añadir un nodo al grafo
void addNode(Graph& graph, const Node& node) {
    graph.nodes[node.id] = node;  // Añadir el nodo al mapa con su ID como clave
}

// Implementación para añadir una arista entre dos nodos
void addEdge(Graph& graph, int fromNodeID, int toNodeID) {
    graph.nodes[fromNodeID].successors.push_back(toNodeID);
    graph.nodes[toNodeID].predecessors.push_back(fromNodeID);
}

// Implementación para eliminar un nodo
void removeNode(Graph& graph, int nodeID) {
    // Eliminar el nodo del mapa
    graph.nodes.erase(nodeID);
    
    // Eliminar el nodo de los sucesores y predecesores de otros nodos
    for (auto& pair : graph.nodes) {
        Node& node = pair.second;
        node.successors.erase(std::remove(node.successors.begin(), node.successors.end(), nodeID), node.successors.end());
        node.predecessors.erase(std::remove(node.predecessors.begin(), node.predecessors.end(), nodeID), node.predecessors.end());
    }
}

// Implementación para obtener todos los nodos
std::vector<Node> getNodes(const Graph& graph) {
    std::vector<Node> nodeList;
    for (const auto& pair : graph.nodes) {
        nodeList.push_back(pair.second);
    }
    return nodeList;
}

// Implementación para imprimir el grafo
void printGraph(const Graph& graph) {
    for (const auto& pair : graph.nodes) {
        const Node& node = pair.second;
        std::cout << "Node ID: " << node.id << " | Operation: " << node.operation << std::endl;
        std::cout << "Inputs: ";
        for (const auto& tensor : node.inputs) {
            std::cout << tensor.name << " (";
            for (size_t i = 0; i < tensor.dimensions.size(); ++i) {
                std::cout << tensor.dimensions[i];
                if (i != tensor.dimensions.size() - 1) std::cout << "x";
            }
            std::cout << ", " << tensor.data_type << ")";
        }
        std::cout << "\nOutput: " << node.output.name << " (";
        for (size_t i = 0; i < node.output.dimensions.size(); ++i) {
            std::cout << node.output.dimensions[i];
            if (i != node.output.dimensions.size() - 1) std::cout << "x";
        }
        std::cout << ", " << node.output.data_type << ")\n";

        std::cout << "Successors: ";
        for (int succ : node.successors) {
            std::cout << succ << " ";
        }
        std::cout << "\nPredecessors: ";
        for (int pred : node.predecessors) {
            std::cout << pred << " ";
        }
        std::cout << "\n\n";
    }
}
