#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <fstream> 
#include <iostream> 

// Enum que representa los diferentes tipos de valores de metadatos en el archivo GGUF.
enum class GGUfMetadataValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Uint64,
    Int64,
    Float64,
    Bool,
    String,
    Array
};

// Estructura que representa un valor de metadato.
struct GGUFMetadataValue {
    // Aquí puedes definir los constructores y destructores apropiados según el tipo.
    // Se pueden incluir campos para almacenar diferentes tipos de valores.
};

// Estructura que representa un valor de metadato que es un arreglo.
struct GGUFMetadataArrayValue {
    GGUfMetadataValueType value_type; // Tipo de los elementos en el arreglo.
    uint64_t len; // Longitud del arreglo.
    std::vector<GGUFMetadataValue> value; // Valores del arreglo.
};

// Estructura que representa un metadato con su clave, tipo y valor.
struct GGUFMetadata {
    std::string key; // Clave del metadato.
    GGUfMetadataValueType value_type; // Tipo de valor.
    GGUFMetadataValue value; // Valor del metadato.
};

// Estructura que representa la cabecera del archivo GGUF.
struct GGUFHeader {
    uint32_t version; // Versión del archivo.
    uint64_t tensor_count; // Cantidad de tensores en el archivo.
    std::vector<GGUFMetadata> metadata; // Lista de metadatos.
};

// Estructura que contiene información sobre un tensor.
struct GGUFTensorInfo {
    std::string name; // Nombre del tensor.
    std::vector<uint64_t> dimensions; // Dimensiones del tensor.
    GGUfMetadataValueType tensor_type; // Tipo de dato del tensor.
    uint64_t offset; // Offset del tensor en el archivo.
};

// Estructura que representa un archivo GGUF completo.
struct GGUFFile {
    GGUFHeader header; // Cabecera del archivo.
    std::vector<GGUFTensorInfo> tensors; // Lista de tensores.
};

// Función que lee una cadena de un buffer de datos.
std::string gguf_string(const uint8_t* &data) {
    uint64_t len; // Longitud de la cadena.
    std::memcpy(&len, data, sizeof(len)); // Copiar la longitud de la cadena desde el buffer.
    data += sizeof(len); // Mover el puntero de datos.
    std::string result(reinterpret_cast<const char*>(data), len); // Crear la cadena a partir del buffer.
    data += len; // Mover el puntero de datos.
    return result; // Devolver la cadena.
}

// Función que verifica si el buffer de datos contiene la cadena mágica "GGUF".
bool magic(const uint8_t* &data) {
    const char magic_str[] = "GGUF"; // Cadena mágica para identificar el formato.
    if (std::memcmp(data, magic_str, sizeof(magic_str) - 1) == 0) { // Comparar con la cadena mágica.
        data += sizeof(magic_str) - 1; // Mover el puntero de datos.
        return true; // La cadena mágica coincide.
    }
    return false; // La cadena mágica no coincide.
}

// Función que determina el tipo de valor de metadato a partir del buffer de datos.
GGUfMetadataValueType gguf_metadata_value_type(const uint8_t* &data) {
    uint32_t type; // Variable para almacenar el tipo.
    std::memcpy(&type, data, sizeof(type)); // Copiar el tipo desde el buffer.
    data += sizeof(type); // Mover el puntero de datos.
    return static_cast<GGUfMetadataValueType>(type); // Devolver el tipo como GGUfMetadataValueType.
}

// Función que lee un valor de metadato del buffer de datos.
GGUFMetadataValue gguf_metadata_value(GGUfMetadataValueType value_type, const uint8_t* &data) {
    GGUFMetadataValue value; // Variable para almacenar el valor.
    switch (value_type) {
        case GGUfMetadataValueType::Uint8: {
            uint8_t v;
            std::memcpy(&v, data, sizeof(v)); // Leer valor uint8.
            data += sizeof(v); // Mover puntero.
            value = GGUFMetadataValue(/* inicializar con v */); // Inicializar el valor.
            break;
        }
        case GGUfMetadataValueType::Int8: {
            int8_t v;
            std::memcpy(&v, data, sizeof(v)); // Leer valor int8.
            data += sizeof(v); // Mover puntero.
            value = GGUFMetadataValue(/* inicializar con v */); // Inicializar el valor.
            break;
        }
        case GGUfMetadataValueType::Uint16: {
            uint16_t v;
            std::memcpy(&v, data, sizeof(v)); // Leer valor uint16.
            data += sizeof(v); // Mover puntero.
            value = GGUFMetadataValue(/* inicializar con v */); // Inicializar el valor.
            break;
        }
        case GGUfMetadataValueType::Int16: {
            int16_t v;
            std::memcpy(&v, data, sizeof(v)); // Leer valor int16.
            data += sizeof(v); // Mover puntero.
            value = GGUFMetadataValue(/* inicializar con v */); // Inicializar el valor.
            break;
        }
        case GGUfMetadataValueType::Uint32: {
            uint32_t v;
            std::memcpy(&v, data, sizeof(v)); // Leer valor uint32.
            data += sizeof(v); // Mover puntero.
            value = GGUFMetadataValue(/* inicializar con v */); // Inicializar el valor.
            break;
        }
        case GGUfMetadataValueType::Int32: {
            int32_t v;
            std::memcpy(&v, data, sizeof(v)); // Leer valor int32.
            data += sizeof(v); // Mover puntero.
            value = GGUFMetadataValue(/* inicializar con v */); // Inicializar el valor.
            break;
        }
        case GGUfMetadataValueType::Float32: {
            float v;
            std::memcpy(&v, data, sizeof(v)); // Leer valor float32.
            data += sizeof(v); // Mover puntero.
            value = GGUFMetadataValue(/* inicializar con v */); // Inicializar el valor.
            break;
        }
        // Otros tipos siguen el mismo patrón...
        case GGUfMetadataValueType::Bool: {
            uint8_t b;
            std::memcpy(&b, data, sizeof(b)); // Leer valor booleano.
            data += sizeof(b); // Mover puntero.
            if (b == 0) {
                value = GGUFMetadataValue(/* inicializar con false */); // Inicializar con falso.
            } else if (b == 1) {
                value = GGUFMetadataValue(/* inicializar con true */); // Inicializar con verdadero.
            } else {
                throw std::invalid_argument("invalid bool value"); // Excepción para valor booleano inválido.
            }
            break;
        }
        case GGUfMetadataValueType::String: {
            std::string str = gguf_string(data); // Leer cadena.
            value = GGUFMetadataValue(/* inicializar con str */); // Inicializar el valor.
            break;
        }
        case GGUfMetadataValueType::Array: {
            GGUfMetadataValueType array_type = gguf_metadata_value_type(data); // Leer tipo de arreglo.
            uint64_t len;
            std::memcpy(&len, data, sizeof(len)); // Leer longitud del arreglo.
            data += sizeof(len); // Mover puntero.
            std::vector<GGUFMetadataValue> v; // Vector para almacenar valores del arreglo.
            for (uint64_t i = 0; i < len; ++i) {
                v.push_back(gguf_metadata_value(array_type, data)); // Leer valores del arreglo.
            }
            value = GGUFMetadataValue(/* inicializar con array_type, len, v */); // Inicializar el valor.
            break;
        }
    }
    return value; // Devolver el valor leído.
}

// Función que lee un metadato del buffer de datos.
GGUFMetadata gguf_metadata(const uint8_t* &data) {
    std::string key = gguf_string(data); // Leer clave del metadato.
    GGUfMetadataValueType value_type = gguf_metadata_value_type(data); // Leer tipo de valor.
    GGUFMetadataValue value = gguf_metadata_value(value_type, data); // Leer valor.
    return GGUFMetadata{key, value_type, value}; // Devolver el metadato.
}

// Función que lee la cabecera del archivo GGUF.
GGUFHeader gguf_header(const uint8_t* &data) {
    if (!magic(data)) { // Verificar la cadena mágica.
        throw std::runtime_error("Invalid GGUF file"); // Excepción si el archivo no es válido.
    }

    uint32_t version; // Variable para almacenar la versión.
    uint64_t tensor_count, metadata_count; // Variables para contar tensores y metadatos.

    std::memcpy(&version, data, sizeof(version)); // Leer la versión.
    data += sizeof(version); // Mover puntero.

    std::memcpy(&tensor_count, data, sizeof(tensor_count)); // Leer cantidad de tensores.
    data += sizeof(tensor_count); // Mover puntero.

    std::memcpy(&metadata_count, data, sizeof(metadata_count)); // Leer cantidad de metadatos.
    data += sizeof(metadata_count); // Mover puntero.

    std::vector<GGUFMetadata> metadata; // Vector para almacenar los metadatos.
    for (uint64_t i = 0; i < metadata_count; ++i) {
        metadata.push_back(gguf_metadata(data)); // Leer cada metadato.
    }

    return GGUFHeader{version, tensor_count, metadata}; // Devolver la cabecera.
}

// Función que lee información sobre un tensor del buffer de datos.
GGUFTensorInfo gguf_tensor_info(const uint8_t* &data) {
    std::string name = gguf_string(data); // Leer nombre del tensor.
    uint32_t n_dimensions; // Variable para la cantidad de dimensiones.
    std::memcpy(&n_dimensions, data, sizeof(n_dimensions)); // Leer cantidad de dimensiones.
    data += sizeof(n_dimensions); // Mover puntero.

    std::vector<uint64_t> dimensions(n_dimensions); // Vector para almacenar dimensiones.
    std::memcpy(dimensions.data(), data, n_dimensions * sizeof(uint64_t)); // Leer las dimensiones.
    data += n_dimensions * sizeof(uint64_t); // Mover puntero.

    GGUfMetadataValueType tensor_type = gguf_metadata_value_type(data); // Leer tipo de tensor.

    uint64_t offset; // Variable para el offset.
    std::memcpy(&offset, data, sizeof(offset)); // Leer offset del tensor.
    data += sizeof(offset); // Mover puntero.

    return GGUFTensorInfo{name, dimensions, tensor_type, offset}; // Devolver información del tensor.
}

// Función que lee un archivo GGUF completo desde el buffer de datos.
GGUFFile gguf_file(const uint8_t* &data) {
    GGUFHeader header = gguf_header(data); // Leer la cabecera.
    std::vector<GGUFTensorInfo> tensors; // Vector para almacenar la información de los tensores.
    for (uint64_t i = 0; i < header.tensor_count; ++i) {
        tensors.push_back(gguf_tensor_info(data)); // Leer cada tensor.
    }
    return GGUFFile{header, tensors}; // Devolver el archivo GGUF.
}
