# extract_graph.py

from graph import Graph, Node, Tensor
import struct

def simplify_graph(graph):
    """
    Simplifies the graph by applying optimization rules
    """
    nodes_to_remove = []
    for node in graph.get_nodes():
        if node.operation == "duplicate":
            print(f"Eliminating redundant node with operation: {node.operation}")
            nodes_to_remove.append(node.id)

    for node_id in nodes_to_remove:
        graph.remove_node(node_id)

def list_accelerators(graph):
    """
    Lists compatible accelerators based on the operations in the graph
    """
    print("List of compatible accelerators:")
    for node in graph.get_nodes():
        if node.operation == "MatMul":
            print(f" - Operation: {node.operation} can be accelerated using GPUs or specialized hardware (e.g., TPUs)")

def load_gguf(filename):
    graph = Graph()

    start = 0     # Inicio de lectura
    end = 1000   # Fin de lectura (n bytes)

    try:
        # Abrir el archivo en modo binario
        with open(filename, 'rb') as file:
            # Obtener el tamaño total del archivo
            file.seek(0, 2)  # Mover al final del archivo
            file_size = file.tell()

            print(f"\n >>> Tamaño del archivo: {file_size} [bytes]")
            
            # Ajustar end si es mayor que el tamaño del archivo
            if end > file_size:
                end = file_size
            
            # Mover al inicio de lectura
            file.seek(start)

            # Calcular el número de bytes a leer
            bytes_to_read = end - start

            # Verificar que el rango de lectura es válido
            if bytes_to_read <= 0:
                print("Error: El rango de lectura no es válido.")
                return graph  # Retornar un grafo vacío

            # Leer los datos de operaciones
            read_data = file.read(bytes_to_read)

            # Verificar si la lectura fue exitosa
            if not read_data:
                print("Error: Fallo en la lectura del archivo.")
                return graph  # Retornar un grafo vacío

            # Decodificar los datos como texto
            read_data_str = read_data.decode('utf-8', errors='ignore')

            print("\n >>> Datos leidos:")
            print(read_data_str)

    except FileNotFoundError:
        print(f"Error: No se pudo abrir el archivo {filename}")
        return graph  # Retornar un grafo vacío en caso de error

    return graph

