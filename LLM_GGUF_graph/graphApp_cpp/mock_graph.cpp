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
    
    // Llenar los tensores de entrada y salida
    gguf_tensor_info input_tensor1 = {
        {input1_str.size(), const_cast<char*>(input1_str.c_str())}, // Nombre
        2, // n_dims
        {64, 64}, // ne
        GGML_TYPE_F32, // type
        0, // offset
        nullptr, // data (aún no asignado)
        64 * 64 * sizeof(float) // size
    };

    gguf_tensor_info input_tensor2 = {
        {input2_str.size(), const_cast<char*>(input2_str.c_str())}, // Nombre
        2, // n_dims
        {64, 64}, // ne
        GGML_TYPE_F32, // type
        0, // offset
        nullptr, // data (aún no asignado)
        64 * 64 * sizeof(float) // size
    };

    gguf_tensor_info output_tensor = {
        {output_str.size(), const_cast<char*>(output_str.c_str())}, // Nombre
        2, // n_dims
        {64, 64}, // ne
        GGML_TYPE_F32, // type
        0, // offset
        nullptr, // data (aún no asignado)
        64 * 64 * sizeof(float) // size
    };

    // Agregar el primer nodo con las entradas iniciales y el tensor de salida
    addNode(graph, {}, input_tensor1, input_tensor2, output_tensor, "Matrix Multiply");

    // Crear nodos adicionales con configuraciones variadas de entradas
    for (int i = 1; i < 15; ++i) {
        std::string new_input_str = "tensor" + std::to_string(i);
        std::string new_output_str = "out" + std::to_string(i);

        gguf_tensor_info new_input_tensor = {
            {new_input_str.size(), const_cast<char*>(new_input_str.c_str())}, // Nombre
            2, // n_dims
            {64, 64}, // ne
            GGML_TYPE_F32, // type
            0, // offset
            nullptr, // data (aún no asignado)
            64 * 64 * sizeof(float) // size
        };

        gguf_tensor_info new_output_tensor = {
            {new_output_str.size(), const_cast<char*>(new_output_str.c_str())}, // Nombre
            2, // n_dims
            {64, 64}, // ne
            GGML_TYPE_F32, // type
            0, // offset
            nullptr, // data (aún no asignado)
            64 * 64 * sizeof(float) // size
        };

        if (i % 2 == 0) {
            // Si el índice es par, el nodo tiene una sola entrada
            addNode(graph, {i - 1}, new_input_tensor, {}, new_output_tensor, "EW Multiply");
        } else {
            // Si el índice es impar, el nodo tiene dos entradas
            addNode(graph, {i - 1, i - 2}, new_input_tensor, new_input_tensor, new_output_tensor, "Add");
        }
    }

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