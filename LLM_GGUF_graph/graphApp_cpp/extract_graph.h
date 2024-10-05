#ifndef EXTRACT_GRAPH_H
#define EXTRACT_GRAPH_H

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "graph.h"

// Función para simplificar el grafo aplicando reglas de optimización
void simplifyGraph(Graph& graph);

// Función para listar los aceleradores compatibles basados en las operaciones del grafo
void listAccelerators(const Graph& graph);

// Función para leer un archivo GGUF y construir el grafo
Graph loadGGUF(const std::string& filename);

#endif // EXTRACT_GRAPH_H
