#include "mock_graph.h"
#include <iostream>

// Función para crear un grafo de ejemplo con 15 nodos
struct Graph create_mock_graph() {
    struct Graph graph;
    // Inicializar el grafo
    initGraph(graph);

    // Crear tensores de ejemplo
    std::string input1_str = "input1";
    std::string input2_str = "input2";
    std::string output_str = "output";

    // Asignar dinámicamente la memoria para la cadena y copiar los datos
    char* tensor_input_name1 = new char[input1_str.size() + 1]; // +1 para el carácter nulo
    std::strcpy(tensor_input_name1, input1_str.c_str());

    char* tensor_input_name2 = new char[input2_str.size() + 1]; // +1 para el carácter nulo
    std::strcpy(tensor_input_name2, input2_str.c_str());

    char* tensor_output_name = new char[output_str.size() + 1]; // +1 para el carácter nulo
    std::strcpy(tensor_output_name, output_str.c_str());

    
    // Llenar los tensores de entrada y salida
    gguf_tensor_info input_tensor1 = {
        {input1_str.size(), tensor_input_name1}, // Nombre
        1, // n_dims
        {4096}, // ne
        GGML_TYPE_F32, // type
        0, // offset
        nullptr, // data (aún no asignado)
        4096 * sizeof(float) // size
    };

    gguf_tensor_info output_tensor_1 = {
        {output_str.size(), tensor_output_name}, // Nombre
        1, // n_dims
        {4096}, // ne
        GGML_TYPE_F32, // type
        0, // offset
        nullptr, // data (aún no asignado)
        4096 * sizeof(float) // size
    };

    gguf_tensor_info output_tensor_2 = {
        {output_str.size(), tensor_output_name}, // Nombre
        1, // n_dims
        {4096}, // ne
        GGML_TYPE_F32, // type
        0, // offset
        nullptr, // data (aún no asignado)
        4096 * sizeof(float) // size
    };


    

    addNode(graph, {}, input_tensor1, {}, output_tensor_1, "RMS NORM");
    addNode(graph, {}, input_tensor1, {}, output_tensor_2, "EW ADDITION");
    
    

    return graph;
}

void print_graph(struct Graph& graph) {
    const std::vector<Node>& nodes = getNodes(graph);

    for (const auto& node : nodes) {
        std::cout << "Node ID: " << node.id << "\n";
        std::cout << "Operation: " << node.operation << "\n";
        std::cout << "Input's id: ";
        for (const auto& input : node.input_ids) {
            std::cout << input << "  ";
        }
        //std::cout << "\nOutput tensor name: " << node.output_tensor.name.data << "\n";
        std::string tensorName(node.output_tensor.name.data, node.output_tensor.name.n);
    
        std::cout << "\nOutput tensor name: " << tensorName << "\n";

        
        std::cout << "Shapes:\n";
        std::cout << "  Input1 Shape: ";
        for (const auto& dim : node.shape_in1) {
            std::cout << dim << " ";
        }
        std::cout << "\n  Input2 Shape: ";
        for (const auto& dim : node.shape_in2) {
            std::cout << dim << " ";
        }
        std::cout << "\n  Output Shape: ";
        for (const auto& dim : node.shape_out) {
            std::cout << dim << " ";
        }
        std::cout << "\n\n";
    }
}