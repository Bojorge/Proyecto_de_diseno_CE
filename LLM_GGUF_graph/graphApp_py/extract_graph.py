from graph import Graph, Node, Tensor
import struct
import os

def print_metadata_as_table(metadata):
    # Lista de prefijos de claves que deseamos extraer
    key_prefixes = ["llama.", "general.", "tokenizer."]
    position = 0  # Para manejar índices

    while position < len(metadata):
        found_key = False  # Para determinar si se encontró un key

        for prefix in key_prefixes:
            start_position = metadata.find(prefix, position)
            if start_position != -1:  # Se encontró un key
                found_key = True

                # Buscar el siguiente key
                end_position = len(metadata)  # Resetear end_position
                for next_prefix in key_prefixes:
                    next_position = metadata.find(next_prefix, start_position + len(prefix))
                    if next_position != -1:
                        end_position = next_position
                        break  # Salir si se encontró el siguiente key

                # Tomar la línea de texto desde start_position hasta end_position
                text_line = metadata[start_position:end_position]
                print(text_line)

                position = end_position - 1  # Continuar desde el final del key encontrado
                break  # Salir del bucle de prefijos si se encontró uno

        # Si no se encontró un key, avanzar la posición
        if not found_key:
            position += 1


def load_gguf(filename):
    graph = {}  # Crear un grafo vacío (o un objeto Graph si tienes una clase Graph definida)
    
    # Variables para los encabezados
    gguf_magic_number = 0
    gguf_version = 0
    tensor_count = 0
    kv_count = 0

    if not os.path.isfile(filename):
        print(f"Error: No se pudo abrir el archivo {filename}")
        return graph  # Retornar un grafo vacío en caso de error

    # Leer el tamaño del archivo completo
    file_size = os.path.getsize(filename)
    print(f"\n >>> Tamaño del archivo:  {file_size}  [bytes]")

    with open(filename, "rb") as file:
        # Leer los primeros datos: GGUF magic number, versión, tensor_count, kv_count
        gguf_magic_number = struct.unpack('I', file.read(4))[0]  # uint32_t
        gguf_version = struct.unpack('I', file.read(4))[0]      # uint32_t
        tensor_count = struct.unpack('Q', file.read(8))[0]      # uint64_t
        kv_count = struct.unpack('Q', file.read(8))[0]          # uint64_t

        # El punto actual en el archivo es donde empieza la metadata
        start = file.tell()

        # Determinar cuántos bytes leer para la metadata
        metadata_bytes_to_read = 5000  # cantidad de metadata a leer
        end = start + metadata_bytes_to_read

        # Verificar que el rango de lectura es válido
        if end > file_size:
            end = file_size

        bytes_to_read = end - start
        if bytes_to_read <= 0:
            print("Error: El rango de lectura no es válido.")
            return graph  # Retornar un grafo vacío

        # Leer la metadata
        metadata_data = file.read(bytes_to_read).decode('utf-8', errors='ignore')

        # Imprimir metadata como tabla
        print(f"\n > gguf_magic_number: {gguf_magic_number}")
        print(f" > gguf_version: {gguf_version}")
        print(f" > tensor_count: {tensor_count}")
        print(f" > kv_count: {kv_count}")

        print_metadata_as_table(metadata_data)

    return graph
