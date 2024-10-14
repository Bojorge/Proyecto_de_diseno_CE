#include "graph_builder.h" 
#include "mock_graph.h"         

int main(int argc, char* argv[]) {

/*
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <llm_model.gguf>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
*/
    struct gguf_context * example_ctx = create_mock_ctx();
    gguf_print_context(*example_ctx);

    return 0;
}
