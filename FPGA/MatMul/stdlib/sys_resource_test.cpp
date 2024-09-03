#include <iostream>
#include <sys/resource.h>

void printCPUUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    // Extraer el tiempo de CPU en modo usuario
    long seconds = usage.ru_utime.tv_sec;
    long microseconds = usage.ru_utime.tv_usec;

    // Convertir microsegundos a segundos
    double microsecondsInSeconds = microseconds / 1e6;

    // Calcular el tiempo total en segundos
    double totalTimeInSeconds = seconds + microsecondsInSeconds;

    // Mostrar los resultados
    std::cout << "Tiempo de CPU en modo usuario:" << std::endl;
    std::cout << "Segundos: " << seconds << std::endl;
    std::cout << "Microsegundos: " << microseconds << std::endl;
    std::cout << "Tiempo total en segundos: " << totalTimeInSeconds << std::endl;
}

int main() {
    // Ejecutar una operación simple para tener algún tiempo de CPU
    for (int i = 0; i < 100000000; ++i);

    // Imprimir el tiempo de CPU
    printCPUUsage();

    return 0;
}
