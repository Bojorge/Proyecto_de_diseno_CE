#include <hls_half.h> // Biblioteca para manejar FP16 

// Función principal del kernel de multiplicación de matrices
// A, B y C son punteros a matrices en formato FP16 (half)
// M es el número de filas de la matriz A
// N es el número de columnas de la matriz A y filas de la matriz B (es igual para que cumplan la condicion de la multiplicacion de matrices)
// P es el número de columnas de la matriz B
void matrix_mult(const half *A, const half *B, half *C, int M, int N, int P) {
    
    // Pragma para definir la interfaz de los puertos de la función
    // `m_axi` indica que estos puertos usarán la interfaz AXI de memoria global
    // `port=A` asocia el puerto A a la interfaz, `offset=slave` indica que el acceso será como esclavo
    // `bundle=gmem` agrupa estos puertos bajo el mismo "bundle" de memoria global
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem

    // Pragma para definir la interfaz de control
    // `s_axilite` define una interfaz de control AXI Lite para los parámetros de la función
    // `bundle=control` agrupa todos los puertos de control bajo el mismo bundle
    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=M bundle=control
    #pragma HLS INTERFACE s_axilite port=N bundle=control
    #pragma HLS INTERFACE s_axilite port=P bundle=control

    // Pragma para la interfaz de retorno de la función, usando AXI Lite
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Recorre las filas de la matriz A
    for (int i = 0; i < M; i++) {
        // Recorre las columnas de la matriz B
        for (int j = 0; j < P; j++) {
            // Inicializa el acumulador
            half sum = 0;
            // Realiza el producto escalar entre la fila `i` de A y la columna `j` de B
            for (int k = 0; k < N; k++) {
                // Accede al elemento usando índices lineales
                sum += A[i * N + k] * B[k * P + j];
            }
            // Guarda el resultado en la matriz C en la posición correspondiente
            C[i * P + j] = sum;
        }
    }
}
