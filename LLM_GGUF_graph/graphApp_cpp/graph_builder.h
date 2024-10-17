#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include <iostream>
#include "graph.h"

// Declaración de la función que construye un grafo a partir de un gguf_context
Graph build_graph_from_context(const struct gguf_context* ctx);

// Declaración de la función para imprimir el contenido del grafo
void print_graph(const Graph& graph);

#endif // GRAPH_BUILDER_H
