#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <nlohmann/json.hpp> // Para JSON
#include <yaml-cpp/yaml.h> // Para YAML
#include <fmt/format.h> // Para formateo de texto

// Define las estructuras necesarias para manejar GGUF
struct GGUFMetadataValue {
    // Agrega los campos necesarios para representar el valor de la metadata
};

struct GGUFMetadata {
    std::string key;
    // Suponiendo que tienes un tipo de valor
    GGUFMetadataValue value;
};

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t tensor_type; // Cambia el tipo según tu implementación
    uint64_t offset;
};

struct GGUFFile {
    std::vector<GGUFMetadata> metadata;
    std::vector<GGUFTensorInfo> tensors;
    // Puedes agregar más campos según sea necesario
};

// Funciones para leer GGUF
GGUFFile read_gguf_file(const std::string& filename) {
    GGUFFile file;
    // Implementa la lógica para leer y analizar el archivo GGUF
    // Asegúrate de llenar 'file.metadata' y 'file.tensors' con datos leídos
    return file;
}

// Función para construir tabla de metadata
std::string build_metadata_table(const GGUFFile& read_file) {
    std::ostringstream oss;
    oss << fmt::format("{:<5} {:<20} {:<10} {}\n", "#", "Key", "Type", "Value");
    for (size_t i = 0; i < read_file.metadata.size(); ++i) {
        const auto& metadata = read_file.metadata[i];
        oss << fmt::format("{:<5} {:<20} {:<10} {}\n", i + 1, metadata.key, "TODO", "TODO"); // Reemplaza "TODO" con la representación del valor
    }
    return oss.str();
}

// Función para construir tabla de información de tensores
std::string build_tensor_info_table(const GGUFFile& read_file) {
    std::ostringstream oss;
    oss << fmt::format("{:<5} {:<20} {:<10} {:<20} {}\n", "#", "Name", "Type", "Dimensions", "Offset");
    for (size_t i = 0; i < read_file.tensors.size(); ++i) {
        const auto& tensor = read_file.tensors[i];
        oss << fmt::format("{:<5} {:<20} {:<10} {:<20} {}\n", i + 1, tensor.name, tensor.tensor_type, "TODO", tensor.offset); // Reemplaza "TODO" con la representación de dimensiones
    }
    return oss.str();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <ruta_al_archivo_gguf> [--json | --yaml | --table]" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    GGUFFile read_file = read_gguf_file(filename);

    // Comprobación de errores
    if (read_file.metadata.empty() && read_file.tensors.empty()) {
        std::cerr << "Error: No se pudieron leer datos del archivo GGUF." << std::endl;
        return 1;
    }

    // Salida de datos
    if (argc > 2) {
        std::string format = argv[2];
        if (format == "--json") {
            nlohmann::json json_output = {{"metadata", read_file.metadata}, {"tensors", read_file.tensors}};
            std::cout << json_output.dump(4) << std::endl; // Formato bonito
        } else if (format == "--yaml") {
            YAML::Emitter out;
            out << YAML::BeginMap;
            out << YAML::Key << "metadata" << YAML::Value << read_file.metadata;
            out << YAML::Key << "tensors" << YAML::Value << read_file.tensors;
            out << YAML::EndMap;
            std::cout << out.str() << std::endl;
        } else {
            std::cout << "Tabla de Metadata:\n" << build_metadata_table(read_file) << std::endl;
            std::cout << "Información de Tensors:\n" << build_tensor_info_table(read_file) << std::endl;
        }
    } else {
        std::cout << "Tabla de Metadata:\n" << build_metadata_table(read_file) << std::endl;
        std::cout << "Información de Tensors:\n" << build_tensor_info_table(read_file) << std::endl;
    }

    return 0;
}
