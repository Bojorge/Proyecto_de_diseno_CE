#ifndef FIXED_POINT_H
#define FIXED_POINT_H

#include <cstdint>
#include <iostream>

// Representación de Punto Fijo

// Formato de punto fijo:
// Generalmente se representa como Valor = Entero × 2^(-número de bits decimales).
// La cantidad de bits totales que decides afecta el rango y la precisión.

// Precisión y Rango:
// Para un formato de punto fijo de n bits, donde se utilizan f bits para la parte fraccionaria:
// Rango de valores representables: De -2^(n-f) a 2^(n-f) - 1.
// Precisión: La menor cantidad que se puede representar es 2^(-f). En el caso específico de usar 16 bits con 4 bits para la parte fraccionaria y 12 bits para la parte entera, 
// Rango aproximado -2048 a 2047.9375 y la menor unidad representable será 0.0625 (error 1=0.0625)

// Definición de la escala para la parte fraccionaria
#define FIXED_SCALE (1 << 4) // 4 bits para la parte fraccionaria (16 = 2^4)

// Funciones para operaciones de punto fijo
int16_t floatToFixed(float f);
float fixedToFloat(int16_t fixedValue);
int16_t fixedMul(int16_t a, int16_t b);

#endif // FIXED_POINT_H
