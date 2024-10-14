#include "mock_graph.h"

const char* gguf_type_to_string(enum gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return "GGUF_TYPE_UINT8";
        case GGUF_TYPE_INT8:    return "GGUF_TYPE_INT8";
        case GGUF_TYPE_UINT16:  return "GGUF_TYPE_UINT16";
        case GGUF_TYPE_INT16:   return "GGUF_TYPE_INT16";
        case GGUF_TYPE_UINT32:  return "GGUF_TYPE_UINT32";
        case GGUF_TYPE_INT32:   return "GGUF_TYPE_INT32";
        case GGUF_TYPE_FLOAT32: return "GGUF_TYPE_FLOAT32";
        case GGUF_TYPE_BOOL:    return "GGUF_TYPE_BOOL";
        case GGUF_TYPE_STRING:  return "GGUF_TYPE_STRING";
        case GGUF_TYPE_ARRAY:   return "GGUF_TYPE_ARRAY";
        case GGUF_TYPE_UINT64:  return "GGUF_TYPE_UINT64";
        case GGUF_TYPE_INT64:   return "GGUF_TYPE_INT64";
        case GGUF_TYPE_FLOAT64: return "GGUF_TYPE_FLOAT64";
        default:                return "Unknown Type";
    }
}

const char* ggml_type_to_string(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:     return "GGML_TYPE_F32";
        case GGML_TYPE_F16:     return "GGML_TYPE_F16";
        case GGML_TYPE_Q4_0:    return "GGML_TYPE_Q4_0";
        case GGML_TYPE_Q4_1:    return "GGML_TYPE_Q4_1";
        case GGML_TYPE_Q5_0:    return "GGML_TYPE_Q5_0";
        case GGML_TYPE_Q5_1:    return "GGML_TYPE_Q5_1";
        case GGML_TYPE_Q8_0:    return "GGML_TYPE_Q8_0";
        case GGML_TYPE_Q8_1:    return "GGML_TYPE_Q8_1";
        case GGML_TYPE_Q2_K:    return "GGML_TYPE_Q2_K";
        case GGML_TYPE_Q3_K:    return "GGML_TYPE_Q3_K";
        case GGML_TYPE_Q4_K:    return "GGML_TYPE_Q4_K";
        case GGML_TYPE_Q5_K:    return "GGML_TYPE_Q5_K";
        case GGML_TYPE_Q6_K:    return "GGML_TYPE_Q6_K";
        case GGML_TYPE_Q8_K:    return "GGML_TYPE_Q8_K";
        case GGML_TYPE_IQ2_XXS: return "GGML_TYPE_IQ2_XXS";
        case GGML_TYPE_IQ2_XS:  return "GGML_TYPE_IQ2_XS";
        case GGML_TYPE_IQ3_XXS: return "GGML_TYPE_IQ3_XXS";
        case GGML_TYPE_IQ1_S:   return "GGML_TYPE_IQ1_S";
        case GGML_TYPE_IQ4_NL:  return "GGML_TYPE_IQ4_NL";
        case GGML_TYPE_IQ3_S:   return "GGML_TYPE_IQ3_S";
        case GGML_TYPE_IQ2_S:   return "GGML_TYPE_IQ2_S";
        case GGML_TYPE_IQ4_XS:  return "GGML_TYPE_IQ4_XS";
        case GGML_TYPE_I8:      return "GGML_TYPE_I8";
        case GGML_TYPE_I16:     return "GGML_TYPE_I16";
        case GGML_TYPE_I32:     return "GGML_TYPE_I32";
        case GGML_TYPE_I64:     return "GGML_TYPE_I64";
        case GGML_TYPE_F64:     return "GGML_TYPE_F64";
        case GGML_TYPE_IQ1_M:   return "GGML_TYPE_IQ1_M";
        case GGML_TYPE_BF16:    return "GGML_TYPE_BF16";
        default:                return "Unknown Type";
    }
}

const char* gguf_value_to_string(enum gguf_type type, union gguf_value value) {
    static char buffer[64];  // Buffer estático para almacenar valores numéricos como cadenas

    switch (type) {
        case GGUF_TYPE_UINT8:
            snprintf(buffer, sizeof(buffer), "%u", value.uint8);
            return buffer;
        case GGUF_TYPE_INT8:
            snprintf(buffer, sizeof(buffer), "%d", value.int8);
            return buffer;
        case GGUF_TYPE_UINT16:
            snprintf(buffer, sizeof(buffer), "%u", value.uint16);
            return buffer;
        case GGUF_TYPE_INT16:
            snprintf(buffer, sizeof(buffer), "%d", value.int16);
            return buffer;
        case GGUF_TYPE_UINT32:
            snprintf(buffer, sizeof(buffer), "%u", value.uint32);
            return buffer;
        case GGUF_TYPE_INT32:
            snprintf(buffer, sizeof(buffer), "%d", value.int32);
            return buffer;
        case GGUF_TYPE_UINT64:
            snprintf(buffer, sizeof(buffer), "%lu", value.uint64);
            return buffer;
        case GGUF_TYPE_INT64:
            snprintf(buffer, sizeof(buffer), "%ld", value.int64);
            return buffer;
        case GGUF_TYPE_FLOAT32:
            snprintf(buffer, sizeof(buffer), "%f", value.float32);
            return buffer;
        case GGUF_TYPE_FLOAT64:
            snprintf(buffer, sizeof(buffer), "%lf", value.float64);
            return buffer;
        case GGUF_TYPE_BOOL:
            return value.bool_ ? "true" : "false";
        case GGUF_TYPE_STRING:
            return value.str.data;  // Retorna el puntero a los datos de la cadena
        case GGUF_TYPE_ARRAY:
            snprintf(buffer, sizeof(buffer), "Array of %lu elements", value.arr.n);
            return buffer;
        default:
            return "Unknown Value";
    }
}

void gguf_print_context(const struct gguf_context ctx) {
    printf("Magic: %.4s\n", ctx.header.magic);
    printf("Version: %u\n", ctx.header.version);
    printf("Number of tensors: %lu\n", ctx.header.n_tensors);
    printf("Number of key-value pairs: %lu\n", ctx.header.n_kv);

    for (uint64_t i = 0; i < ctx.header.n_kv; ++i) {
        printf("Key: %.*s\n", (int)ctx.kv[i].key.n, ctx.kv[i].key.data);
        
        printf("Type: %s\n", gguf_type_to_string(ctx.kv[i].type));

        printf("Value: %s\n", gguf_value_to_string(ctx.kv[i].type, ctx.kv[i].value));

        printf("........................................\n");
        
    }

    for (uint64_t i = 0; i < ctx.header.n_tensors; ++i) {
        printf("Tensor Name: %.*s\n", (int)ctx.infos[i].name.n, ctx.infos[i].name.data);
        printf("Precision: %s\n", ggml_type_to_string(ctx.infos[i].type));
        printf("Number of Dimensions: %u\n", ctx.infos[i].n_dims);
        // Imprimir las dimensiones
        if (ctx.infos[i].n_dims == 1) {
            printf("Shape [#elements]: [%lu]\n", ctx.infos[i].ne[0]);
        } else if (ctx.infos[i].n_dims == 2) {
            printf("Shape [#rows] [#cols]: [%lu] [%lu]\n", ctx.infos[i].ne[0], ctx.infos[i].ne[1]);
        }
        printf("Data Size: %zu\n", ctx.infos[i].size);
        printf("Offsets: %lu\n", ctx.infos[i].offset);
        printf("\n---------------------------------------------------\n");
    }

}

size_t gguf_get_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return sizeof(float);     // 4 bytes
        case GGML_TYPE_I32: return sizeof(int32_t);   // 4 bytes
        case GGML_TYPE_Q2_K: return 2;                // 2 bytes para GGML_TYPE_Q2_K
        // ... otros tipos
        default: return 0; // Tipo no soportado
    }
}

size_t gguf_calculate_tensor_size(struct gguf_tensor_info * info) {
    size_t num_elements = 1;
    for (uint32_t j = 0; j < info->n_dims; ++j) {
        num_elements *= info->ne[j]; 
    }

    return num_elements * gguf_get_type_size(info->type);  // Tamaño total = num_elementos * tamaño_tipo
}

struct gguf_context * create_mock_ctx(){
    struct gguf_context * mock_ctx = (struct gguf_context *) calloc(1, sizeof(struct gguf_context));
    size_t offset = 0;
    size_t size = 0;
    size_t block_number = 0;
    size_t tensors_per_block = 8;
    size_t total_tensors = 0;

    const char *base_key = "mock_key";
    const char *base_tensor = "mock_tensor";
    
    strncpy(mock_ctx->header.magic, "gguf", 4);
    mock_ctx->header.magic[sizeof(mock_ctx->header.magic) - 1] = '\0';      
    

    mock_ctx->kv    = NULL;
    mock_ctx->infos = NULL;
    mock_ctx->data  = NULL;

    mock_ctx->header.version = 2;
    mock_ctx->header.n_tensors = 291;
    mock_ctx->header.n_kv = 20;

    //cargar información de los pares llave-valor (kv)
    const uint64_t n_kv = mock_ctx->header.n_kv;
    mock_ctx->header.n_kv = 0;
    mock_ctx->kv = (struct gguf_kv *) calloc(n_kv, sizeof(struct gguf_kv));
    for (uint64_t i = 0; i < n_kv; ++i) {
        struct gguf_kv * kv = &mock_ctx->kv[i];

        char key[15];
        // Concatenar el índice a la clave base
        snprintf(key, sizeof(key), "%s_%lu", base_key, i);
        kv->key.data = (char *)malloc(strlen(key) + 1);
        strcpy(kv->key.data, key);
        kv->key.n = strlen(key);

        kv->type = GGUF_TYPE_UINT64; 
        kv->value.uint64 = i;

        mock_ctx->header.n_kv++;
    }

    // Reservar memoria para los tensores
    const uint64_t n_tensors = mock_ctx->header.n_tensors;
    mock_ctx->infos = (struct gguf_tensor_info *) calloc(n_tensors, sizeof(struct gguf_tensor_info));

    // Cargar la información de los tensores
    while (total_tensors < n_tensors) {

        for(uint64_t j = 0; j < tensors_per_block; ++j){
            struct gguf_tensor_info * info = &mock_ctx->infos[total_tensors];
            char tensor_name[60];

            snprintf(tensor_name, sizeof(tensor_name), "blk.%zu_%s.%lu", block_number, base_tensor, (unsigned long)j);
            info->name.data = (char *)malloc(strlen(tensor_name) + 1);
            strcpy(info->name.data, tensor_name);
            info->name.n = strlen(tensor_name);

            // dimensiones del tensor (algunos tensores con 1 dimension y otros con 2)
            info->n_dims = (total_tensors % 2 == 0) ? 1 : 2;  // Alternar entre 1 y 2 dimensiones

            for (uint32_t k = 0; k < info->n_dims; ++k) {
                info->ne[k] = (total_tensors+2) * (k+2);  // Asignar tamaños de dimensión incrementales basados en el índice para cada dimensión
            }

            // Asignar tipo de tensor (alternar entre diferentes tipos)
            info->type = (total_tensors % 3 == 0) ? GGML_TYPE_F32 : (total_tensors % 3 == 1) ? GGML_TYPE_I32 : GGML_TYPE_Q2_K;

            size_t tensor_size = gguf_calculate_tensor_size(info);
            info->size = tensor_size;
            info->offset = offset;
            offset += tensor_size;

            // Asignar un bloque de datos simulado
            info->data = malloc(tensor_size);

            total_tensors += 1;

            info->data = malloc(tensor_size);
        }

        block_number += 1;
        
        if((n_tensors - total_tensors) < tensors_per_block && (n_tensors - total_tensors) > 0){
            tensors_per_block = (n_tensors - total_tensors);
        }
    }

    size = offset;
    mock_ctx->offset = offset;     
    mock_ctx->size = size;
    mock_ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    mock_ctx->data = malloc(size);

    return mock_ctx;

}

/*
int main() {

    struct gguf_context * example_ctx = create_mock_ctx();
    gguf_print_context(*example_ctx);

    return 0;
}
*/

 
    /*
    //header
    METADATA            VALUE
        
    version 	          2
    tensor_count 	     291
    kv_count 	          20

    //pares key-value (n_kv  = 20)
    KEY                                             VALUE
    general.architecture 	                        llama

    general.name 	                                adaptllm_law-llm

    general.file_type 	                            MOSTLY_Q2_K

    general.quantization_version 	                2

    llama.context_length 	                        2048

    llama.embedding_length 	                        4096

    llama.block_count 	                            32

    llama.feed_forward_length 	                    11008

    llama.rope.dimension_count 	                    128

    llama.attention.head_count 	                    32

    llama.attention.head_count_kv 	                32

    llama.attention.layer_norm_rms_epsilon 	        9.999999974752427e-7

    tokenizer.ggml.model 	                        llama

    tokenizer.ggml.tokens 	                        [<unk>, <s>, </s>, <0x00>, <0x01>, ...]

    tokenizer.ggml.scores 	                        [0, 0, 0, 0, 0, ...]

    tokenizer.ggml.token_type 	                    [2, 3, 3, 6, 6, ...]

    tokenizer.ggml.bos_token_id 	                1

    tokenizer.ggml.eos_token_id 	                2

    tokenizer.ggml.unknown_token_id 	            0

    tokenizer.ggml.padding_token_id 	            32000
    */
/////////////////////////////////////////////////////////////////////
   //información de tensores
   /*
   Tensors                              Shape 	                    Precision
    token_embd.weight 	                [4 096, 32 001] 	            Q2_K
    
    blk.0.attn_k.weight 	            [4 096, 4 096] 	                Q2_K
    
    blk.0.attn_norm.weight 	            [4 096] 	                    F32
    
    blk.0.attn_q.weight 	            [4 096, 4 096] 	                Q2_K
    
    blk.0.attn_v.weight 	            [4 096, 4 096] 	                Q3_K
    
    blk.0.ffn_down.weight 	            [11 008, 4 096] 	            Q3_K
    
    blk.0.ffn_gate.weight 	            [4 096, 11 008] 	            Q3_K
    
    blk.0.ffn_norm.weight 	            [4 096] 	                    F32
    
    blk.0.ffn_up.weight 	            [4 096, 11 008] 	            Q3_K
    
    blk.0.attn_output.weight 	        [4 096, 4 096] 	            Q3_K
    
    blk.1.attn_k.weight 	            [4 096, 4 096] 	                Q2_K
    
    blk.1.attn_norm.weight 	            [4 096] 	                    F32
    
    blk.1.attn_q.weight 	            [4 096, 4 096] 	                Q2_K
    
    blk.1.attn_v.weight 	            [4 096, 4 096] 	                Q3_K
    
    blk.1.ffn_down.weight 	            [11 008, 4 096] 	            Q3_K
    
    blk.1.ffn_gate.weight 	            [4 096, 11 008] 	            Q3_K
    
    blk.1.ffn_norm.weight 	            [4 096] 	                    F32
    
    blk.1.ffn_up.weight 	            [4 096, 11 008] 	            Q3_K
    
    blk.1.attn_output.weight 	        [4 096, 4 096] 	            Q3_K
    
    blk.2.attn_k.weight 	            [4 096, 4 096] 	                Q2_K
    
    blk.2.attn_norm.weight          	[4 096] 	                    F32
    
    blk.2.attn_q.weight 	            [4 096, 4 096] 	                Q2_K
    
    blk.2.attn_v.weight 	            [4 096, 4 096] 	                Q3_K
    
    blk.2.ffn_down.weight 	            [11 008, 4 096] 	            Q3_K
    
    blk.2.ffn_gate.weight 	            [4 096, 11 008] 	            Q3_K
    
    blk.2.ffn_norm.weight 	            [4 096] 	                    F32
    
    blk.2.ffn_up.weight 	            [4 096, 11 008] 	            Q3_K
    
    blk.2.attn_output.weight 	        [4 096, 4 096] 	            Q3_K

    blk.3.attn_k.weight 	            [4 096, 4 096] 	                Q2_K
    
    blk.3.attn_norm.weight          	[4 096] 	                    F32
    
    blk.3.attn_q.weight 	            [4 096, 4 096] 	                Q2_K
    
    blk.3.attn_v.weight 	            [4 096, 4 096] 	                Q3_K
    
    blk.3.ffn_down.weight           	[11 008, 4 096] 	            Q3_K
    
    blk.3.ffn_gate.weight           	[4 096, 11 008] 	            Q3_K
    
    blk.3.ffn_norm.weight           	[4 096] 	                    F32
    
    blk.3.ffn_up.weight 	            [4 096, 11 008] 	            Q3_K
    
    blk.3.attn_output.weight        	[4 096, 4 096] 	            Q3_K
    
    blk.4.attn_k.weight 	            [4 096, 4 096] 	                Q2_K
    
    blk.4.attn_norm.weight          	[4 096] 	                    F32
    
    blk.4.attn_q.weight 	            [4 096, 4 096] 	                Q2_K
    
    blk.4.attn_v.weight 	            [4 096, 4 096] 	                Q3_K    
    
    blk.4.ffn_down.weight 	            [11 008, 4 096] 	            Q3_K    
    
    blk.4.ffn_gate.weight 	            [4 096, 11 008] 	            Q3_K    
    
    blk.4.ffn_norm.weight 	            [4 096] 	                    F32     
    
    blk.4.ffn_up.weight 	            [4 096, 11 008] 	            Q3_K
    
    blk.4.attn_output.weight 	        [4 096, 4 096] 	

    Q3_K
    blk.5.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.5.attn_norm.weight 	[4 096] 	

    F32
    blk.5.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.5.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.5.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.5.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.5.ffn_norm.weight 	[4 096] 	

    F32
    blk.5.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.5.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.6.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.6.attn_norm.weight 	[4 096] 	

    F32
    blk.6.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.6.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.6.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.6.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.6.ffn_norm.weight 	[4 096] 	

    F32
    blk.6.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.6.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.7.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.7.attn_norm.weight 	[4 096] 	

    F32
    blk.7.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.7.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.7.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.7.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.7.ffn_norm.weight 	[4 096] 	

    F32
    blk.7.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.7.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.8.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.8.attn_norm.weight 	[4 096] 	

    F32
    blk.8.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.8.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.8.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.8.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.8.ffn_norm.weight 	[4 096] 	

    F32
    blk.8.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.8.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.9.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.9.attn_norm.weight 	[4 096] 	

    F32
    blk.9.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.9.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.9.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.9.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.9.ffn_norm.weight 	[4 096] 	

    F32
    blk.9.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.9.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.10.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.10.attn_norm.weight 	[4 096] 	

    F32
    blk.10.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.10.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.10.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.10.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.10.ffn_norm.weight 	[4 096] 	

    F32
    blk.10.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.10.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.11.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.11.attn_norm.weight 	[4 096] 	

    F32
    blk.11.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.11.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.11.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.11.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.11.ffn_norm.weight 	[4 096] 	

    F32
    blk.11.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.11.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.12.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.12.attn_norm.weight 	[4 096] 	

    F32
    blk.12.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.12.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.12.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.12.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.12.ffn_norm.weight 	[4 096] 	

    F32
    blk.12.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.12.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.13.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.13.attn_norm.weight 	[4 096] 	

    F32
    blk.13.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.13.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.13.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.13.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.13.ffn_norm.weight 	[4 096] 	

    F32
    blk.13.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.13.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.14.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.14.attn_norm.weight 	[4 096] 	

    F32
    blk.14.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.14.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.14.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.14.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.14.ffn_norm.weight 	[4 096] 	

    F32
    blk.14.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.14.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.15.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.15.attn_norm.weight 	[4 096] 	

    F32
    blk.15.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.15.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.15.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.15.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.15.ffn_norm.weight 	[4 096] 	

    F32
    blk.15.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.15.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.16.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.16.attn_norm.weight 	[4 096] 	

    F32
    blk.16.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.16.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.16.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.16.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.16.ffn_norm.weight 	[4 096] 	

    F32
    blk.16.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.16.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.17.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.17.attn_norm.weight 	[4 096] 	

    F32
    blk.17.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.17.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.17.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.17.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.17.ffn_norm.weight 	[4 096] 	

    F32
    blk.17.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.17.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.18.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.18.attn_norm.weight 	[4 096] 	

    F32
    blk.18.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.18.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.18.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.18.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.18.ffn_norm.weight 	[4 096] 	

    F32
    blk.18.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.18.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.19.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.19.attn_norm.weight 	[4 096] 	

    F32
    blk.19.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.19.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.19.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.19.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.19.ffn_norm.weight 	[4 096] 	

    F32
    blk.19.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.19.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.20.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.20.attn_norm.weight 	[4 096] 	

    F32
    blk.20.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.20.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.20.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.20.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.20.ffn_norm.weight 	[4 096] 	

    F32
    blk.20.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.20.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.21.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.21.attn_norm.weight 	[4 096] 	

    F32
    blk.21.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.21.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.21.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.21.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.21.ffn_norm.weight 	[4 096] 	

    F32
    blk.21.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.21.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.22.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.22.attn_norm.weight 	[4 096] 	

    F32
    blk.22.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.22.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.22.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.22.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.22.ffn_norm.weight 	[4 096] 	

    F32
    blk.22.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.22.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.23.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.23.attn_norm.weight 	[4 096] 	

    F32
    blk.23.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.23.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.23.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.23.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.23.ffn_norm.weight 	[4 096] 	

    F32
    blk.23.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.23.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.24.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.24.attn_norm.weight 	[4 096] 	

    F32
    blk.24.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.24.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.24.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.24.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.24.ffn_norm.weight 	[4 096] 	

    F32
    blk.24.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.24.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.25.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.25.attn_norm.weight 	[4 096] 	

    F32
    blk.25.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.25.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.25.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.25.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.25.ffn_norm.weight 	[4 096] 	

    F32
    blk.25.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.25.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.26.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.26.attn_norm.weight 	[4 096] 	

    F32
    blk.26.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.26.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.26.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.26.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.26.ffn_norm.weight 	[4 096] 	

    F32
    blk.26.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.26.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.27.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.27.attn_norm.weight 	[4 096] 	

    F32
    blk.27.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.27.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.27.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.27.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.27.ffn_norm.weight 	[4 096] 	

    F32
    blk.27.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.27.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.28.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.28.attn_norm.weight 	[4 096] 	

    F32
    blk.28.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.28.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.28.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.28.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.28.ffn_norm.weight 	[4 096] 	

    F32
    blk.28.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.28.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.29.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.29.attn_norm.weight 	[4 096] 	

    F32
    blk.29.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.29.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.29.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.29.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.29.ffn_norm.weight 	[4 096] 	

    F32
    blk.29.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.29.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.30.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.30.attn_norm.weight 	[4 096] 	

    F32
    blk.30.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.30.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.30.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K
    blk.30.ffn_gate.weight 	[4 096, 11 008] 	

    Q3_K
    blk.30.ffn_norm.weight 	[4 096] 	

    F32
    blk.30.ffn_up.weight 	[4 096, 11 008] 	

    Q3_K
    blk.30.attn_output.weight 	[4 096, 4 096] 	

    Q3_K
    blk.31.attn_k.weight 	[4 096, 4 096] 	

    Q2_K
    blk.31.attn_norm.weight 	[4 096] 	

    F32
    blk.31.attn_q.weight 	[4 096, 4 096] 	

    Q2_K
    blk.31.attn_v.weight 	[4 096, 4 096] 	

    Q3_K
    blk.31.ffn_down.weight 	[11 008, 4 096] 	

    Q3_K

    blk.31.ffn_gate.weight 	            [4 096, 11 008] 	            Q3_K
    
    blk.31.ffn_norm.weight 	            [4 096] 	                    F32
    
    blk.31.ffn_up.weight 	            [4 096, 11 008] 	            Q3_K
    
    blk.31.attn_output.weight 	        [4 096, 4 096] 	                Q3_K
    
    output.weight 	                    [4 096, 32 001] 	            Q6_K
    
    output_norm.weight 	                [4 096] 	                    F32

   */

