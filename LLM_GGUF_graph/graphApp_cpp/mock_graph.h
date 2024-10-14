#ifndef GGUF_H
#define GGUF_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "structures.h"

// Funciones para conversión de tipo a cadena
const char* gguf_type_to_string(enum gguf_type type);
const char* ggml_type_to_string(enum ggml_type type);

// Función para convertir valores a cadenas
const char* gguf_value_to_string(enum gguf_type type, union gguf_value value);

// Función para imprimir el contexto
void gguf_print_context(const struct gguf_context ctx);

// Función para obtener el tamaño de un tipo de ggml
size_t gguf_get_type_size(enum ggml_type type);

// Función para calcular el tamaño de un tensor
size_t gguf_calculate_tensor_size(struct gguf_tensor_info * info);

// Función para crear un contexto de ejemplo/mock
struct gguf_context * create_mock_ctx();

#endif // GGUF_H
