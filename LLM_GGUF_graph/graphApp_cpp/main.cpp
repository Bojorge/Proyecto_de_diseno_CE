#include "graph_builder.h" 
#include "mock_graph.h"   
#include "graph_to_file.h"      

int main(int argc, char* argv[]) {

/*
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <llm_model.gguf>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
*/

    struct Graph graph  = create_mock_graph();

    create_dot_file(graph);

    show_dot_file();

    //print_graph(graph);
    
    return 0;
}
