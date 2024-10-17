#include "graph_to_file.h"

//sudo apt-get install graphviz


void create_dot_file(const Graph& graph) {
    // Abrir el archivo .dot
    std::ofstream file("graph_output.dot");
    if (!file.is_open()) {
        std::cerr << "Error al abrir el archivo .dot para escritura." << std::endl;
        return;
    }

    // Configuración inicial del archivo DOT
    file << "digraph G {\n";
    file << "    rankdir=LR;\n";  // Colocar los nodos de izquierda a derecha
    file << "    node [shape=box, style=rounded];\n";  // Hacer los nodos como cuadros
    file << "    edge [arrowhead=open, penwidth=2.0];\n";  // Flechas con cabezas abiertas y líneas más gruesas

    // Recorrer los nodos del grafo
    for (const auto& node : graph.nodes) {
        // Convertir gguf_str a std::string
        std::string input1_name_str(node.input1_tensor.name.data, node.input1_tensor.name.n);
        std::string input2_name_str(node.input2_tensor.name.data, node.input2_tensor.name.n);
        std::string output_name_str(node.output_tensor.name.data, node.output_tensor.name.n);

        // Crear los nodos en DOT
        file << "    " << node.id << " [label=\"" << node.operation << "\"];\n";

        // Si hay un tensor de entrada (input1_tensor)
        if (!input1_name_str.empty()) {
            // Nodo de entrada a la izquierda, si no está conectado a otro nodo
            std::string input1_id = "tensor_" + input1_name_str;
            file << "    " << input1_id << " [label=\"" << input1_name_str << " 1D (" << node.input1_tensor.ne[0] << ")\", shape=none];\n";
            file << "    " << input1_id << " -> " << node.id 
                 << " [label=\"1D (" << node.input1_tensor.ne[0] << ")\"];\n";
        }

        // Si hay un tensor de entrada 2 (input2_tensor), conectarlo también
        if (!input2_name_str.empty()) {
            std::string input2_id = "tensor_" + input2_name_str;
            file << "    " << input2_id << " [label=\"" << input2_name_str << " 1D (" << node.input2_tensor.ne[0] << ")\", shape=none];\n";
            file << "    " << input2_id << " -> " << node.id 
                 << " [label=\"1D (" << node.input2_tensor.ne[0] << ")\"];\n";
        }

        // Conectar el nodo al tensor de salida
        if (!output_name_str.empty()) {
            std::string output_id = "tensor_" + output_name_str;
            file << "    " << node.id << " -> " << output_id 
                 << " [label=\"1D (" << node.output_tensor.ne[0] << ")\"];\n";
            file << "    " << output_id << " [label=\"" << output_name_str << " 1D (" << node.output_tensor.ne[0] << ")\", shape=none];\n";
        }
    }

    // Finalizar el archivo DOT
    file << "}\n";

    file.close();
    std::cout << "Archivo .dot generado con éxito." << std::endl;
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
