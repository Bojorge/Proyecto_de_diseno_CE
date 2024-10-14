#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include "graph.h" 

// Función para crear un archivo .dot a partir del grafo
void create_dot_file(const Graph& graph);

// Función para mostrar el archivo .dot generado
void show_dot_file();

#endif // GRAPH_UTILS_H
