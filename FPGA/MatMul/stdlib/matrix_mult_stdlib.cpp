#include <iostream>
#include <vector>
#include <stdexcept> // Incluimos para std::invalid_argument

// Función para multiplicar matrices
std::vector<std::vector<int>> matrix_mult(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, int M, int N, int P) {
    // Verificación de dimensiones para asegurar que las matrices se puedan multiplicar
    if (A[0].size() != static_cast<size_t>(N) || B.size() != static_cast<size_t>(N) || A.size() != static_cast<size_t>(M) || B[0].size() != static_cast<size_t>(P)) {
        throw std::invalid_argument("Dimensiones de matrices no válidas para la multiplicación");
    }

    // Inicialización de la matriz resultante C de MxP
    std::vector<std::vector<int>> C(M, std::vector<int>(P, 0));

    // Realizar la multiplicación de matrices
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            for (int k = 0; k < N; k++) {
                // Multiplicar elemento de A con el elemento correspondiente de B y acumular en C
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

int main() {
    // Dimensiones de las matrices
    int M = 2, N = 3, P = 2;

    // Inicialización de matrices A y B
    std::vector<std::vector<int>> A = {{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<int>> B = {{7, 8}, {9, 10}, {11, 12}};

    // Llamada a la función de multiplicación de matrices
    std::vector<std::vector<int>> C = matrix_mult(A, B, M, N, P);

    // Mostrar el resultado de la multiplicación
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
