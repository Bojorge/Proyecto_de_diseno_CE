#include <iostream>

// Tamaño máximo para la cadena de tiempo
#define MAX_TIME_LENGTH 21

struct Sentence {
    char character;
    char time[MAX_TIME_LENGTH];
};

class SharedSentenceBuffer {
public:
    SharedSentenceBuffer(Sentence* buffer) : buffer_(buffer) {}

    char getCharacter() const {
        return buffer_->character;
    }

    void setCharacter(char c) {
        std::cout << "esve" << std::endl;
        buffer_->character = c;
    }

    // Otros métodos para manipular los datos de Sentence

private:
    Sentence* buffer_;
};
