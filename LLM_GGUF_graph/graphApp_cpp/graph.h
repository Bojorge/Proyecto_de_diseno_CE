#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

// Representa un tensor en el grafo
struct Tensor {
    std::string name;                // Nombre del tensor
    std::vector<int> dimensions;     // Dimensiones del tensor
    std::string data_type;           // Tipo de dato (e.g., float32, int8)
    std::string storage_type;        // Tipo de almacenamiento (e.g., CPU, GPU, etc.)
};

// Representa un nodo (operación) en el grafo
struct Node {
    int id;                          // Identificador único del nodo
    std::string operation;           // Tipo de operación (MatMul, Conv2D, etc.)
    std::vector<Tensor> inputs;      // Tensores de entrada
    Tensor output;                   // Tensor de salida
    std::vector<int> predecessors;   // Nodos predecesores (IDs de nodos)
    std::vector<int> successors;     // Nodos sucesores (IDs de nodos)
};

// Grafo
struct Graph {
    std::unordered_map<int, Node> nodes;  // Mapa de nodos (ID como clave)
    int nextNodeID = 0;                   // Para llevar un seguimiento del próximo ID disponible
};

// Funciones para manejar el grafo
Tensor createTensor(const std::string& name, const std::vector<int>& dimensions, const std::string& data_type, const std::string& storage_type);
Node createNode(Graph& graph, const std::string& operation, const std::vector<Tensor>& inputs, const Tensor& output);

void addNode(Graph& graph, const Node& node);    // Añadir un nodo al grafo
void addEdge(Graph& graph, int fromNodeID, int toNodeID);  // Conectar dos nodos (añadir una arista)
void removeNode(Graph& graph, int nodeID);  // Eliminar un nodo del grafo
std::vector<Node> getNodes(const Graph& graph);  // Obtener todos los nodos
void printGraph(const Graph& graph);  // Imprimir el grafo

#endif // GRAPH_H
