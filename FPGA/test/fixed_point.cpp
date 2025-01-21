#include "fixed_point.h"

// Función para convertir de float a punto fijo (16 bits)
int16_t floatToFixed(float f) {
    return static_cast<int16_t>(f * FIXED_SCALE);
}

// Función para convertir de punto fijo a float
float fixedToFloat(int16_t fixedValue) {
    return static_cast<float>(fixedValue) / FIXED_SCALE;
}

// Función de multiplicación de punto fijo
int16_t fixedMul(int16_t a, int16_t b) {
    // Multiplicamos los valores fijos y luego ajustamos para mantener la escala
    return static_cast<int16_t>((static_cast<int32_t>(a) * b) / FIXED_SCALE);
}


