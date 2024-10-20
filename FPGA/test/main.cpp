#include "fixed_point.h"
#include <bitset>
#include <cstring>


// Función para imprimir el valor binario de un float
void printFloatBinary(float f) {
    uint32_t floatBits;
    // Copiar los bits del float a un entero sin signo
    // Esto se hace para obtener la representación binaria del float en un formato manipulable
    std::memcpy(&floatBits, &f, sizeof(float));
    // Crear un objeto bitset de 32 bits para almacenar la representación binaria
    std::bitset<32> binary(floatBits);
    std::cout << "Valor float en binario: " << binary << std::endl;
    
    // La representación es: [signo | exponente | mantisa]
    
    // Ejemplos de representación binaria de otros números en formato IEEE 754:
    // 0.0 se representa como 00000000000000000000000000000000
    // -1.0 se representa como 10111111100000000000000000000000
    // 2.0 se representa como 01000000000000000000000000000000
    // -2.5 se representa como 11000000001000000000000000000000
    // 0.15625 se representa como 00111111100101000000000000000000
    // 3.14159 se representa como 01000000010010010000111111011011
    // 10.0 se representa como 01000000101000000000000000000000
    // 0.1 se representa como 00111111010011001100110011001101

    // Valores extremos representables:
    // Valor máximo positivo (3.4028235E+38): 01111111111111111111111111111111
    // Valor mínimo positivo (1.17549435E-38): 00000000000000000000000000000001
    // Valor mínimo negativo (-3.4028235E+38): 11111111111111111111111111111111

    // Cualquier número que exceda el valor máximo resultará en un desbordamiento (infinito).
    // Números muy cercanos a cero (pero positivos) se redondearán a cero (subdesbordamiento).
    
    // Para el ejemplo específico de 1.0, esto se verá como 00111111100000000000000000000000
}


// Función para imprimir el valor binario de un entero de punto fijo
void printFixedBinary(int16_t fixedValue) {
    std::bitset<16> binary(fixedValue);
    std::cout << "Valor punto fijo en binario: " << binary << std::endl;
}

// Función para mostrar el valor convertido a float
void printFixedValue(int16_t fixedValue) {
    std::cout << fixedToFloat(fixedValue) << std::endl;
}



int main() {
    // Ejemplos de valores flotantes para verificar sus representaciones binarias
    float examples[] = {
        0.0f,          // Caso base
        1.0f,          // Dentro del rango positivo
        2047.97f,    // Valor máximo representable
        -1.0f,         // Dentro del rango negativo
        -2048.0f,      // Valor mínimo representable
        2.5f,          // Valor representable en el rango
        -2.0f,         // Valor negativo dentro del rango
        0.0625f,       // Menor unidad representable
        1.10f,          // Valor que puede ser redondeado
        3.5f           // Valor representable en el rango
    };

    for (float fa : examples) {
        printFloatBinary(fa);
        std::cout << "------------------" << std::endl;
        int16_t a = floatToFixed(fa);
        printFixedBinary(a);
        std::cout << "------------------" << std::endl;
        std::cout << "Float: " << fa << " -> Punto fijo: " << a << " -> " << std::endl;
        std::cout << "Valor convertido de vuelta a float: " << fixedToFloat(a) << std::endl << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << "------------------" << std::endl;
    }

    return 0;
}

/*
int main() {
    // Ejemplos de valores flotantes para verificar sus representaciones binarias
    float examples[] = {1.0f, 0.0f, -1.0f, 2.0f, -2.5f, 0.15625f, 3.14159f, 10.0f, 0.1f};

    for (float fa : examples) {
        printFloatBinary(fa);
        std::cout << "------------------" << std::endl;
        int16_t a = floatToFixed(fa);
        printFixedBinary(a);
        std::cout << "------------------" << std::endl;
        std::cout << "Float: " << fa << " -> Punto fijo: " << a << " -> " << std::endl;
        std::cout << "Valor convertido de vuelta a float: " << fixedToFloat(a) << std::endl << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << "------------------" << std::endl;
        std::cout << "------------------" << std::endl;
    }

    // Valores extremos
    float maxFloat = 3.4028235E+38f;
    float minFloat = 1.17549435E-38f;
    printFloatBinary(maxFloat);
    printFloatBinary(minFloat);
    std::cout << "------------------" << std::endl;
    printFixedBinary(floatToFixed(maxFloat));
    printFixedBinary(floatToFixed(minFloat));
    std::cout << "------------------" << std::endl;
    std::cout << "Valor máximo: " << maxFloat << " -> Punto fijo: " << floatToFixed(maxFloat) << std::endl;
    std::cout << "Valor mínimo: " << minFloat << " -> Punto fijo: " << floatToFixed(minFloat) << std::endl;

    return 0;
}
*/


