#include <iostream>
#include <string>
#include "graph.h"          
#include "extract_graph.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    Graph graph = loadGGUF(filename);

    simplifyGraph(graph);

    listAccelerators(graph);

    return 0;
}
