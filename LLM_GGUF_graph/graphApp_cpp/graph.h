#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>
#include <unordered_map>

// Definición de un nodo en el grafo (una operación)
struct Node {
    std::string name;                  // Nombre de la operación (ej. Matrix Multiply, Softmax)
    std::vector<std::string> inputs;   // Nombres de los nodos de entrada
    std::vector<std::string> outputs;  // Nombres de los nodos de salida
    std::string operation;             // Tipo de operación (Matrix Multiply, EW Multiply, Softmax, etc)
    std::vector<int> shape_in;         // Forma del tensor de entrada
    std::vector<int> shape_out;        // Forma del tensor de salida
};

// Grafo que contiene nodos y sus conexiones
class Graph {
public:
    // Añadir un nodo al grafo
    void add_node(const Node& node);
    
    // Obtener todos los nodos del grafo (para visualización u otros usos)
    const std::unordered_map<std::string, Node>& get_nodes() const;

private:
    std::unordered_map<std::string, Node> nodes;  // Mapa de nodos, indexados por nombre
};

#endif // GRAPH_H
