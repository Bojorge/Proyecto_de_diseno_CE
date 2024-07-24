#include <iostream>
extern "C" {
    #include "funciones_c.h"
}

void funcion_en_cpp() {
    std::cout << "Función en C++" << std::endl;
}

int main() {
    funcion_en_c();
    funcion_en_cpp();
    return 0;
}
