#include "extract_graph.h"

// Simplifica el grafo aplicando reglas de optimización
void simplifyGraph(Graph& graph) {
    for (const auto& node : getNodes(graph)) { // Cambié graph.getNodes() a getNodes(graph)
        if (node.operation == "duplicate") {
            std::cout << "Eliminando nodo redundante con operación: " << node.operation << std::endl;
            removeNode(graph, node.id); // Implementar en Graph una función que elimine nodos
        }
        // Agrega más reglas de simplificación según el contexto
    }
}

// Lista los aceleradores compatibles basados en las operaciones del grafo
void listAccelerators(const Graph& graph) {
    std::cout << "List of compatible accelerators:" << std::endl;
    
    // Basado en las operaciones del grafo, identificamos los aceleradores compatibles
    for (const auto& node : getNodes(graph)) {
        if (node.operation == "MatMul") {
            std::cout << " - Operation: " << node.operation << " could be accelerated using GPUs or specialized hardware (TPUs)" << std::endl;
        }
        // Añadir más condiciones según la naturaleza de las operaciones y aceleradores
    }
}

Graph loadGGUF(const std::string& filename) {
    Graph graph;

    std::streampos start = 0;     // Inicio de lectura
    std::streampos end = 10000;    // Fin de lectura (n bytes)

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
        return graph;  // Retornar un grafo vacío en caso de error
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    
    std::cout << "\n >>> Tamaño del archivo:  " << fileSize << "  [bytes]" << std::endl;
    if (end > fileSize) {
        end = fileSize;
    }

    // Mover al inicio de lectura
    file.seekg(start);

    // Calcular el número de bytes a leer
    std::streampos bytesToRead = end - start;

    // Verificar que el rango de lectura es válido
    if (bytesToRead <= 0) {
        std::cerr << "Error: El rango de lectura no es válido." << std::endl;
        return graph;  // Retornar un grafo vacío
    }

    // Leer datos de operaciones
    std::string operationsData(bytesToRead, '\0');
    file.read(&operationsData[0], bytesToRead);

    // Verificar si la lectura fue exitosa
    if (!file) {
        std::cerr << "Error: Fallo en la lectura del archivo." << std::endl;
        return graph;  // Retornar un grafo vacío
    }

    std::cout << "\n >>> Datos de operaciones: \n" << operationsData << std::endl;

    file.close();
    return graph;
}