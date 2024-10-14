#include "graph_to_file.h"

//sudo apt-get install graphviz
//sudo apt install --reinstall libstdc++6




void create_dot_file(const Graph& graph) {
    std::ofstream file("graph.dot");
    
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo graph.dot" << std::endl;
        return;
    }

    file << "digraph G {\n"; // Inicia el grafo

    // Itera sobre cada nodo en el grafo
    for (const auto& node : graph.nodes) {
        // Escapa el nombre de la operación
        std::string operation_label = node.operation;
        std::replace(operation_label.begin(), operation_label.end(), '\"', '\''); // Reemplaza comillas dobles

        // Agrega el nodo con su operación como etiqueta
        file << "    " << node.id << " [label=\"" << operation_label << "\"];\n";
        
        // Agrega las conexiones a los nodos de entrada (máximo 2 entradas)
        for (const auto& input_id : node.input_ids) {
            file << "    " << input_id << " -> " << node.id << ";\n"; // Conectar cada entrada al nodo
        }

        // Solo puede haber una salida
        if (node.output_tensor.name.n > 0) {
            std::string tensorName(node.output_tensor.name.data, node.output_tensor.name.n);
            // Escapa el nombre del tensor
            std::replace(tensorName.begin(), tensorName.end(), '\"', '\''); // Reemplaza comillas dobles
            // Actualiza la representación del nodo para incluir la salida
            file << "    " << node.id << " [label=\"" << operation_label << "\\n" << tensorName << "\"];\n";
        }
    }
    file << "}\n"; // Cierra el grafo
    file.close();
    std::cout << "Archivo graph.dot generado." << std::endl;
}


void show_dot_file() {
    // Genera una imagen PNG del archivo dot
    int result = std::system("dot -Tpng graph.dot -o graph.png");
    if (result != 0) {
        std::cerr << "Error: No se pudo generar la imagen PNG." << std::endl;
        return;
    }

    result = std::system("xdg-open graph.png");
    if (result != 0) {
        std::cerr << "Error: No se pudo abrir la imagen PNG." << std::endl;
    }
}
