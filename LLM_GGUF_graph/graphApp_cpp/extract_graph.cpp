#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include "extract_graph.h"

void printMetadataAsTable(const std::string& metadata) {
    // Lista de prefijos de claves que deseamos extraer
    const char* keyPrefixes[] = {"llama.", "general.", "tokenizer."};
    const int numPrefixes = 3; 

    size_t position = 0; // Para manejar índices
    size_t startPosition = 0;
    size_t endPosition = 0;
    std::string textLine;

    while (position < metadata.size()) {
        bool foundKey = false; // Para determinar si se encontró un key

        for (int i = 0; i < numPrefixes; i++) {
            startPosition = metadata.find(keyPrefixes[i], position);
            if (startPosition != std::string::npos) {
                foundKey = true; // Se encontró un key

                // Buscar el siguiente key
                endPosition = std::string::npos; // Resetear endPosition
                for (int j = 0; j < numPrefixes; j++) {
                    endPosition = metadata.find(keyPrefixes[j], startPosition + strlen(keyPrefixes[i]));
                    if (endPosition != std::string::npos) {
                        break; // Salir si se encontró el siguiente key
                    }
                }

                // Si no se encontró un siguiente key, ajustar endPosition al final de la cadena
                if (endPosition == std::string::npos) {
                    endPosition = metadata.size();
                }

                // Tomar la línea de texto desde startPosition hasta endPosition
                textLine = metadata.substr(startPosition, endPosition - startPosition);
                std::cout << textLine << std::endl;

                position = endPosition-1; // Continuar desde el final del key encontrado
                break; // Salir del bucle de prefijos si se encontró uno
            }
        }

        // Si no se encontró un key, avanzar la posición
        if (!foundKey) {
            position++;
        }
    }
}


Graph loadGGUF(const std::string& filename) {
    Graph graph;
    // Variables para los encabezados
    uint32_t gguf_magic_number;
    uint32_t gguf_version;
    uint64_t tensor_count;
    uint64_t kv_count;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
        return graph;  // Retornar un grafo vacío en caso de error
    }

    // Leer el tamaño del archivo completo
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    std::cout << "\n >>> Tamaño del archivo:  " << fileSize << "  [bytes]" << std::endl;

    // Ir al principio del archivo
    file.seekg(0, std::ios::beg);

    // Leer los primeros datos: GGUF magic number, versión, tensor_count, metadata_kv_count
    file.read(reinterpret_cast<char*>(&gguf_magic_number), sizeof(gguf_magic_number));
    file.read(reinterpret_cast<char*>(&gguf_version), sizeof(gguf_version));
    file.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
    file.read(reinterpret_cast<char*>(&kv_count), sizeof(kv_count));

    // El punto actual en el archivo es donde empieza la metadata
    std::streampos start = file.tellg();

    // Determinar cuántos bytes leer para la metadata
    std::streampos metadataBytesToRead = 5000;  // cantidad de metadata a leer
    std::streampos end = start + metadataBytesToRead;

    // Verificar que el rango de lectura es válido
    if (end > fileSize) {
        end = fileSize;
    }

    std::streampos bytesToRead = end - start;
    if (bytesToRead <= 0) {
        std::cerr << "Error: El rango de lectura no es válido." << std::endl;
        return graph;  // Retornar un grafo vacío
    }

    // Leer la metadata
    std::string metadataData(bytesToRead, '\0');
    file.read(&metadataData[0], bytesToRead);

    // Verificar si la lectura fue exitosa
    if (!file) {
        std::cerr << "Error: Fallo en la lectura del archivo." << std::endl;
        return graph;  // Retornar un grafo vacío
    }

    std::cout << "\n > gguf_magic_number: " << gguf_magic_number << std::endl;
    std::cout << " > gguf_version: " << gguf_version << std::endl;
    std::cout << " > tensor_count: " << tensor_count << std::endl;
    std::cout << " > kv_count: " << kv_count << std::endl;

    // Imprimir metadata como tabla
    printMetadataAsTable(metadataData);

    file.close();
    return graph;
}
