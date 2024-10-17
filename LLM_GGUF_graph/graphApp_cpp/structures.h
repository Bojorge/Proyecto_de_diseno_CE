#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <stdint.h>   // Para tipos como uint8_t, uint32_t, etc.
#include <stddef.h>   // Para el tipo size_t
#include <stdbool.h>  // Para el tipo bool

// Definiciones necesarias para las estructuras
#define GGML_MAX_DIMS 4  // Máximo número de dimensiones
#define GGUF_DEFAULT_ALIGNMENT 32

// Enumeración para los tipos de datos de ggml (tensores)
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_COUNT,
};

// Enumeración para los diferentes tipos de datos que gguf puede almacenar
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,       // Marca el final de la enumeración
};

// Estructura para manejar cadenas de texto en gguf
struct gguf_str {
    uint64_t n;    // Longitud de la cadena
    char * data;   // Datos de la cadena
};

// Unión que representa los posibles valores de gguf
union gguf_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    struct gguf_str str;   // Si el valor es una cadena

    struct {
        enum gguf_type type; // Tipo de los elementos del arreglo
        uint64_t n;          // Cantidad de elementos
        void * data;         // Puntero a los datos del arreglo
    } arr;
};

// Estructura para representar el encabezado de un archivo gguf
struct gguf_header {
    char magic[4];        // Cadena "gguf" indicando el formato
    uint32_t version;     // Versión del formato
    uint64_t n_tensors;   // Número de tensores (GGUFv2)
    uint64_t n_kv;        // Número de pares clave-valor (GGUFv2)
};

// Estructura para representar pares clave-valor (kv) en gguf
struct gguf_kv {
    struct gguf_str key;   // Clave del par
    enum gguf_type type;   // Tipo de dato del valor
    union gguf_value value; // Valor asociado a la clave
};

// Estructura que contiene la información de un tensor en gguf
struct gguf_tensor_info {
    struct gguf_str name;     // Nombre del tensor
    uint32_t n_dims;          // Número de dimensiones del tensor
    uint64_t ne[GGML_MAX_DIMS];  // Número de elementos por dimensión
    enum ggml_type type;      // Tipo de dato del tensor
    uint64_t offset;          // Desplazamiento desde el inicio de los datos (debe ser múltiplo de ALIGNMENT)

    // Para la API de escritura
    const void * data;        // Puntero a los datos del tensor
    size_t size;              // Tamaño de los datos en bytes
};

// Estructura que representa el contexto gguf
struct gguf_context {
    struct gguf_header header;        // Encabezado del archivo gguf
    struct gguf_kv * kv;              // Arreglo de pares clave-valor
    struct gguf_tensor_info * infos;  // Arreglo con la información de los tensores

    size_t alignment;   // Alineación de los datos
    size_t offset;      // Desplazamiento de los datos desde el inicio del archivo
    size_t size;        // Tamaño total de los datos en bytes

    void * data;        // Puntero a la sección de datos
};

#endif // STRUCTURES_H
