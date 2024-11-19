#include <stdio.h>     
#include <stdlib.h>     
#include <string.h>    
#include <stdint.h>    
#include <stdbool.h>   
#include <inttypes.h>
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>   
#include <sched.h> 
//#include <cfloat>
#include <stdatomic.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define GGML_MAX_DIMS 4

#define GROUP_MAX_EPS_IQ3_XXS 1e-8f

#define GROUP_MAX_EPS_IQ2_S 1e-8f

#define GGUF_DEFAULT_ALIGNMENT 32

#define GGUF_MAGIC "GGUF" 

#define GGML_MAX_CONTEXTS       64

#define GGML_CALLOC(num, size) ggml_calloc(num, size)

#define GGML_NUMA_MAX_NODES 8

#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)

#define GGML_PRINT(...) printf(__VA_ARGS__)

#define GROUP_MAX_EPS 1e-15f

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

//#define GGML_ALIGNED_FREE(ptr)    _aligned_free(ptr)
#define GGML_ALIGNED_FREE(ptr) free(ptr)

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fflush(stdout); \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#define ggml_assert_aligned(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#ifdef __GNUC__
#    define GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define GGML_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define GGML_DEPRECATED(func, hint) func
#endif

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_API __declspec(dllexport)
#        else
#            define GGML_API __declspec(dllimport)
#        endif
#    else
#        define GGML_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GGML_API
#endif

#ifdef GGML_MULTIPLATFORM
#    if defined(_WIN32)
#        define GGML_CALL
#    else
#        define GGML_CALL __attribute__((__ms_abi__))
#    endif
#else
#    define GGML_CALL
#endif

#define GGML_MAX_OP_PARAMS      64

#define GGML_MAX_NAME           64

#define GGML_MAX_SRC            10

#define GGML_NUMA_MAX_CPUS 512

#if UINTPTR_MAX == 0xFFFFFFFF
    #define GGML_MEM_ALIGN 4
#else
    #define GGML_MEM_ALIGN 16
#endif

static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
static const float GELU_COEF_A     = 0.044715f;

typedef struct {
    uint32_t * grid;
    int      * map;
    uint16_t * neighbours;
} iq3_entry_t;

static iq3_entry_t iq3_data[2] = {
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
};

inline static void * ggml_aligned_malloc(size_t size) {
    if (size == 0) {
        GGML_PRINT("WARNING: Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        return NULL;
    }
    void * aligned_memory = NULL;
#ifdef GGML_USE_CPU_HBM
    int result = hbw_posix_memalign(&aligned_memory, 16, size);
#elif GGML_USE_METAL
    int result = posix_memalign(&aligned_memory, sysconf(_SC_PAGESIZE), size);
#else
    int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);
#endif
    if (result != 0) {
        // Handle allocation failure
        const char *error_desc = "unknown allocation error";
        switch (result) {
            case EINVAL:
                error_desc = "invalid alignment value";
                break;
            case ENOMEM:
                error_desc = "insufficient memory";
                break;
        }
        GGML_PRINT("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        GGML_ASSERT(false);
        return NULL;
    }
    return aligned_memory;
}

#define GGML_ALIGNED_MALLOC(size) ggml_aligned_malloc(size)

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

typedef double ggml_float;

#define QK_K 256
#define K_SCALE_SIZE 12

#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_half) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_half) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_half) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half s; // d * sum(qs[i])
        } GGML_COMMON_AGGR;
        ggml_half2 ds;
    };
    int8_t qs[QK8_1]; // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1, "wrong q8_1 block size/padding");

//typedef __fp16 ggml_fp16_internal_t;
typedef float ggml_fp16_internal_t;  // Cambiado a float

#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

typedef uint16_t ggml_fp16_t;

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    ggml_fp16_internal_t tmp;
    memcpy(&tmp, &h, sizeof(ggml_fp16_t));
    return (float)tmp;
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
    ggml_fp16_t res;
    ggml_fp16_internal_t tmp = f;
    memcpy(&res, &tmp, sizeof(ggml_fp16_t));
    return res;
}

//
// Super-block quantization structures
//

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_half) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight
typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");

// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

// This is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");

// (Almost) "true" 2-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 2.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t), "wrong iq2_xxs block size/padding");

// 2.3125 bpw quants
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;
static_assert(sizeof(block_iq2_xs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t) + QK_K/32, "wrong iq2_xs block size/padding");

// 2.5625 bpw quants
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
} block_iq2_s;
static_assert(sizeof(block_iq2_s) == sizeof(ggml_half) + QK_K/4 + QK_K/16, "wrong iq2_s block size/padding");

// (Almost) "true" 3-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 3.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_half d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3*(QK_K/8), "wrong iq3_xxs block size/padding");

// 3.4375 bpw
#define IQ3S_N_SCALE QK_K/64
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
static_assert(sizeof(block_iq3_s) == sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");

typedef struct {
    ggml_half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;
static_assert(sizeof(block_iq1_s) == sizeof(ggml_half) + QK_K/8 + QK_K/16, "wrong iq1_s block size/padding");

// 1.75 bpw
typedef struct {
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits
    uint8_t  qh[QK_K/16];     // grid index, high 3 bits + grid shift bit (for two groups of 8)
    uint8_t  scales[QK_K/32]; // 3-bit block scales (4-bit if QK_K == 64)
} block_iq1_m;
static_assert(sizeof(block_iq1_m) == QK_K/8 + QK_K/16 + QK_K/32, "wrong iq1_m block size/padding");

// Used by IQ1_M quants
typedef union {
    ggml_half f16;
    uint16_t  u16;
} iq1m_scale_t;

// Non-linear quants
#define QK4_NL 32
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2, "wrong iq4_nl block size/padding");

typedef struct {
    ggml_half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;
static_assert(sizeof(block_iq4_xs) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/64 + QK_K/2, "wrong iq4_xs block size/padding");


//typedef volatile LONG atomic_int;
//typedef volatile int64_t atomic_int;
//typedef int64_t atomic_int;

/*
#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif
*/

enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
};

enum ggml_object_type {
    GGML_OBJECT_TYPE_TENSOR,      // 0
    GGML_OBJECT_TYPE_GRAPH,       // 1
    GGML_OBJECT_TYPE_WORK_BUFFER   // 2
};

struct ggml_scratch {
        size_t offs;
        size_t size;
        void * data;
};

struct ggml_object {
        size_t offs;
        size_t size;

        struct ggml_object * next;

        enum ggml_object_type type;

        char padding[4];
};

struct ggml_context {
    size_t mem_size;
    void* mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
    bool   no_alloc_save; // this is used to save the no_alloc state when using scratch buffers

    int    n_objects;

    struct ggml_object* objects_begin;
    struct ggml_object* objects_end;

    struct ggml_scratch scratch;
    struct ggml_scratch scratch_save;
};

static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

static struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    struct ggml_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    char * const mem_buffer = ctx->mem_buffer;
    struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

    if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
        GGML_PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed, ctx->mem_size);
        assert(false);
        return NULL;
    }

    *obj_new = (struct ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
        .type = type,
    };

    ggml_assert_aligned(mem_buffer + obj_new->offs);

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    return obj_new;
}

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_context_container {
    bool used;

    struct ggml_context context;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting

    uint32_t cpuset;
    /*
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
    */
};

static const float GELU_QUICK_COEF = -1.702f;

enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        GGML_TYPE_COUNT,
};

enum ggml_backend_type {
    GGML_BACKEND_TYPE_CPU = 0,
    GGML_BACKEND_TYPE_GPU = 10,
    GGML_BACKEND_TYPE_GPU_SPLIT = 20,
};

enum ggml_op {
        GGML_OP_NONE = 0,

        GGML_OP_DUP,
        GGML_OP_ADD,
        GGML_OP_ADD1,
        GGML_OP_ACC,
        GGML_OP_SUB,
        GGML_OP_MUL,
        GGML_OP_DIV,
        GGML_OP_SQR,
        GGML_OP_SQRT,
        GGML_OP_LOG,
        GGML_OP_SUM,
        GGML_OP_SUM_ROWS,
        GGML_OP_MEAN,
        GGML_OP_ARGMAX,
        GGML_OP_REPEAT,
        GGML_OP_REPEAT_BACK,
        GGML_OP_CONCAT,
        GGML_OP_SILU_BACK,
        GGML_OP_NORM, // normalize
        GGML_OP_RMS_NORM,
        GGML_OP_RMS_NORM_BACK,
        GGML_OP_GROUP_NORM,

        GGML_OP_MUL_MAT,
        GGML_OP_MUL_MAT_ID,
        GGML_OP_OUT_PROD,

        GGML_OP_SCALE,
        GGML_OP_SET,
        GGML_OP_CPY,
        GGML_OP_CONT,
        GGML_OP_RESHAPE,
        GGML_OP_VIEW,
        GGML_OP_PERMUTE,
        GGML_OP_TRANSPOSE,
        GGML_OP_GET_ROWS,
        GGML_OP_GET_ROWS_BACK,
        GGML_OP_DIAG,
        GGML_OP_DIAG_MASK_INF,
        GGML_OP_DIAG_MASK_ZERO,
        GGML_OP_SOFT_MAX,
        GGML_OP_SOFT_MAX_BACK,
        GGML_OP_ROPE,
        GGML_OP_ROPE_BACK,
        GGML_OP_CLAMP,
        GGML_OP_CONV_TRANSPOSE_1D,
        GGML_OP_IM2COL,
        GGML_OP_CONV_TRANSPOSE_2D,
        GGML_OP_POOL_1D,
        GGML_OP_POOL_2D,
        GGML_OP_UPSCALE, // nearest interpolate
        GGML_OP_PAD,
        GGML_OP_ARANGE,
        GGML_OP_TIMESTEP_EMBEDDING,
        GGML_OP_ARGSORT,
        GGML_OP_LEAKY_RELU,

        GGML_OP_FLASH_ATTN_EXT,
        GGML_OP_FLASH_ATTN_BACK,
        GGML_OP_SSM_CONV,
        GGML_OP_SSM_SCAN,
        GGML_OP_WIN_PART,
        GGML_OP_WIN_UNPART,
        GGML_OP_GET_REL_POS,
        GGML_OP_ADD_REL_POS,

        GGML_OP_UNARY,

        GGML_OP_MAP_UNARY,
        GGML_OP_MAP_BINARY,

        GGML_OP_MAP_CUSTOM1_F32,
        GGML_OP_MAP_CUSTOM2_F32,
        GGML_OP_MAP_CUSTOM3_F32,

        GGML_OP_MAP_CUSTOM1,
        GGML_OP_MAP_CUSTOM2,
        GGML_OP_MAP_CUSTOM3,

        GGML_OP_CROSS_ENTROPY_LOSS,
        GGML_OP_CROSS_ENTROPY_LOSS_BACK,

        GGML_OP_COUNT,
};

 typedef struct { uint16_t bits; } ggml_bf16_t;

static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

float ggml_bf16_to_fp32(ggml_bf16_t x) {
#define ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    return GGML_BF16_TO_FP32(x);  // it just left shifts
}

inline static float ggml_gelu_quick_f32(float x) {
    return x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x)));
}


void ggml_fp32_to_bf16_row(const float * x, ggml_bf16_t * y, int64_t n) {
  int i = 0;
  /*
#if defined(__AVX512BF16__)
  for (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    for (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
    */
}

void ggml_bf16_to_fp32_row(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
    /*
#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(y + i,
                         _mm512_castsi512_ps(
                             _mm512_slli_epi32(
                                 _mm512_cvtepu16_epi32(
                                     _mm256_loadu_si256(
                                         (const __m256i *)(x + i))),
                                 16)));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i,
                         _mm256_castsi256_ps(
                             _mm256_slli_epi32(
                                 _mm256_cvtepu16_epi32(
                                     _mm_loadu_si128(
                                         (const __m128i *)(x + i))),
                                 16)));
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
    */
}

typedef uint16_t ggml_fp16_t;

#define UNUSED GGML_UNUSED
#define GGML_UNUSED(x) (void)(x)


GGML_API void        ggml_fp16_to_fp32_row(const ggml_fp16_t *, float *, int64_t);
void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

GGML_API void        ggml_fp32_to_fp16_row(const float *, ggml_fp16_t *, int64_t);
void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for(; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

#ifdef  __cplusplus
// restrict not standard in C++
#define GGML_RESTRICT
#else
#define GGML_RESTRICT restrict
#endif
    typedef void (*ggml_to_float_t)  (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    typedef void (*ggml_from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int64_t k);
    typedef void (*ggml_vec_dot_t)   (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                      const void * GGML_RESTRICT y, size_t by, int nrc);

    typedef struct {
        const char      * type_name;
        int               blck_size;
        size_t            type_size;
        bool              is_quantized;
        ggml_to_float_t   to_float;
        ggml_from_float_t from_float;
        ggml_from_float_t from_float_reference;
        ggml_vec_dot_t    vec_dot;
        enum ggml_type    vec_dot_type;
        int64_t           nrows; // number of rows to process simultaneously;
    } ggml_type_traits_t;

    GGML_API ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);

static inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static void ggml_vec_dot_f32(int n, float * restrict s, size_t bs, const float * restrict x, size_t bx, const float * restrict y, size_t by, int nrc);


static void ggml_vec_dot_f16(int n, float * restrict s, size_t bs, ggml_fp16_t * restrict x, size_t bx, ggml_fp16_t * restrict y, size_t by, int nrc);
static void ggml_vec_dot_f16(int n, float * restrict s, size_t bs, ggml_fp16_t * restrict x, size_t bx, ggml_fp16_t * restrict y, size_t by, int nrc) {
    assert(nrc == 1);
    ggml_float sumf = 0.0;
    /*
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    ggml_float sumf = 0.0;

#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC sum[GGML_F16_ARR] = { GGML_F16_VEC_ZERO };

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

            sum[j] = GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F16_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i])*GGML_FP16_TO_FP32(y[i]));
    }
#else
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i])*GGML_FP16_TO_FP32(y[i]));
    }
#endif
    */

    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i])*GGML_FP16_TO_FP32(y[i]));
    }
    *s = sumf;
}

static void ggml_vec_dot_bf16(int n, float * restrict s, size_t bs, ggml_bf16_t * restrict x, size_t bx, ggml_bf16_t * restrict y, size_t by, int nrc);
static void ggml_vec_dot_bf16(int n, float * restrict s, size_t bs, ggml_bf16_t * restrict x, size_t bx, ggml_bf16_t * restrict y, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    int i = 0;
    ggml_float sumf = 0;

#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
#define LOAD(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c2);
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#undef LOAD
#elif defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 8), LOAD(y + i + 8)), c2);
        c3 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c3);
        c4 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 24), LOAD(y + i + 24)), c4);
    }
    __m128 g;
    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3),
                       _mm256_add_ps(c2, c4));
    g = _mm_add_ps(_mm256_extractf128_ps(c1, 1),
                   _mm256_castps256_ps128(c1));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    sumf += (ggml_float)_mm_cvtss_f32(g);

#undef LOAD
#endif

    for (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_TO_FP32(x[i]) *
                             GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;
}

static void ggml_vec_dot_f32(int n, float * restrict s, size_t bs, const float * restrict x, size_t bx, const float * restrict y, size_t by, int nrc) {
   assert(nrc == 1);
   /*
   UNUSED(nrc);
   UNUSED(bx);
   UNUSED(by);
   UNUSED(bs);

#if defined(GGML_SIMD)
    float sumf = 0.0f;
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

            sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i]*y[i];
    }
#else
    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#endif
*/
     // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }

    *s = sumf;
}

// Quantization
void quantize_row_q4_0_reference(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
// reference implementation for deterministic creation of model files
void quantize_row_q4_0_reference(const float * restrict x, block_q4_0 * restrict y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void quantize_row_q4_1_reference(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_1_reference(const float * restrict x, block_q4_1 * restrict y, int64_t k) {
    const int qk = QK4_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        //y[i].d = GGML_FP32_TO_FP16(d);
        //y[i].m = GGML_FP32_TO_FP16(min);
        y[i].GGML_COMMON_AGGR.d = GGML_FP32_TO_FP16(d);
        y[i].GGML_COMMON_AGGR.m = GGML_FP32_TO_FP16(min);


        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void quantize_row_q5_0_reference(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_0_reference(const float * restrict x, block_q5_0 * restrict y, int64_t k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}

void quantize_row_q5_1_reference(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_1_reference(const float * restrict x, block_q5_1 * restrict y, int64_t k) {
    const int qk = QK5_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 5) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        //y[i].d = GGML_FP32_TO_FP16(d);
        //y[i].m = GGML_FP32_TO_FP16(min);
        y[i].GGML_COMMON_AGGR.d = GGML_FP32_TO_FP16(d);
        y[i].GGML_COMMON_AGGR.m = GGML_FP32_TO_FP16(min);

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
    }
}

void quantize_row_q8_0_reference(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0_reference(const float * restrict x, block_q8_0 * restrict y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

void quantize_row_q8_1_reference(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1_reference(const float * restrict x, block_q8_1 * restrict y, int64_t k) {
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_1; j++) {
            const float v = x[i*QK8_1 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        //y[i].d = GGML_FP32_TO_FP16(d);
        y[i].GGML_COMMON_AGGR.d = GGML_FP32_TO_FP16(d);

        int sum = 0;

        for (int j = 0; j < QK8_1/2; ++j) {
            const float v0 = x[i*QK8_1           + j]*id;
            const float v1 = x[i*QK8_1 + QK8_1/2 + j]*id;

            y[i].qs[          j] = roundf(v0);
            y[i].qs[QK8_1/2 + j] = roundf(v1);

            sum += y[i].qs[          j];
            sum += y[i].qs[QK8_1/2 + j];
        }

        //y[i].s = GGML_FP32_TO_FP16(sum*d);
        y[i].GGML_COMMON_AGGR.s = GGML_FP32_TO_FP16(sum*d);
    }
}

static float make_qkx2_quants(int n, int nmax, const float * restrict x, const float * restrict weights,
        uint8_t * restrict L, float * restrict the_min, uint8_t * restrict Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
//#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
 //   for (volatile int i = 1; i < n; ++i) {
//#else
    for (int i = 1; i < n; ++i) {
//#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}


static float make_qx_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, int rmse_type,
        const float * restrict qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

void quantize_row_q2_K_reference(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
void quantize_row_q2_K_reference(const float * restrict x, block_q2_K * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float   weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];

    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            //y[i].d = GGML_FP32_TO_FP16(max_scale/q4scale);
            y[i].GGML_COMMON_AGGR.d = GGML_FP32_TO_FP16(max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            //y[i].d = GGML_FP32_TO_FP16(0.f);
            y[i].GGML_COMMON_AGGR.d = GGML_FP32_TO_FP16(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            //y[i].dmin = GGML_FP32_TO_FP16(max_min/q4scale);
            y[i].GGML_COMMON_AGGR.dmin = GGML_FP32_TO_FP16(max_min/q4scale);
        } else {
            //y[i].dmin = GGML_FP32_TO_FP16(0.f);
            y[i].GGML_COMMON_AGGR.dmin = GGML_FP32_TO_FP16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            //const float d = GGML_FP16_TO_FP32(y[i].d) * (y[i].scales[j] & 0xF);
            const float d = GGML_FP16_TO_FP32(y[i].GGML_COMMON_AGGR.d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            //const float dm = GGML_FP16_TO_FP32(y[i].dmin) * (y[i].scales[j] >> 4);
            const float dm = GGML_FP16_TO_FP32(y[i].GGML_COMMON_AGGR.dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16*j + ii] + dm)/d);
                l = MAX(0, MIN(3, l));
                L[16*j + ii] = l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}
static inline void get_scale_min_k4(int j, const uint8_t * restrict q, uint8_t * restrict d, uint8_t * restrict m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static float make_q3_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = MAX(-nmax, MIN(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return sumlx / suml2;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}

void quantize_row_q3_K_reference(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
void quantize_row_q3_K_reference(const float * restrict x, block_q3_K * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    float scales[QK_K / 16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_q3_quants(16, 4, x + 16*j, L + 16*j, true);
            float scale = fabsf(scales[j]);
            if (scale > amax) {
                amax = scale; max_scale = scales[j];
            }
        }

        memset(y[i].scales, 0, 12);
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int8_t l = nearest_int(iscale*scales[j]);
                l = MAX(-32, MIN(31, l)) + 32;
                if (j < 8) {
                    y[i].scales[j] = l & 0xF;
                } else {
                    y[i].scales[j-8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
            }
            y[i].d = GGML_FP32_TO_FP16(1/iscale);
        } else {
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }

        int8_t sc;
        for (int j = 0; j < QK_K/16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
            sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
            float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-4, MIN(3, l));
                L[16*j + ii] = l + 4;
            }
        }

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

void quantize_row_q4_K_reference(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_K_reference(const float * restrict x, block_q4_K * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        //y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].GGML_COMMON_AGGR.d = GGML_FP32_TO_FP16(max_scale/63.f);
        //y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);
        y[i].GGML_COMMON_AGGR.dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            //const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            const float d = GGML_FP16_TO_FP32(y[i].GGML_COMMON_AGGR.d) * sc;
            if (!d) continue;
            //const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            const float dm = GGML_FP16_TO_FP32(y[i].GGML_COMMON_AGGR.dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

static int iq3_find_best_neighbour(const uint16_t * restrict neighbours, const uint32_t * restrict grid,
        const float * restrict xval, const float * restrict weight, float scale, int8_t * restrict L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

void quantize_row_q5_K_reference(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_K_reference(const float * restrict x, block_q5_K * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint8_t L[QK_K];
    float mins[QK_K/32];
    float scales[QK_K/32];
    float weights[32];
    uint8_t Laux[32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 31, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 31, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        //y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        //y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);
        y[i].GGML_COMMON_AGGR.d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].GGML_COMMON_AGGR.dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            //const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            const float d = GGML_FP16_TO_FP32(y[i].GGML_COMMON_AGGR.d) * sc;
            if (!d) continue;
            //const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            const float dm = GGML_FP16_TO_FP32(y[i].GGML_COMMON_AGGR.dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(31, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * restrict qh = y[i].qh;
        uint8_t * restrict ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }

        x += QK_K;
    }
}

static inline int iq3_data_index(int grid_size) {
    (void)grid_size;
    GGML_ASSERT(grid_size == 256 || grid_size == 512);
    return grid_size == 256 ? 0 : 1;
}

static void quantize_row_iq3_xxs_impl(int grid_size, const float * restrict x, void * restrict vy, int64_t n,
        const float * restrict quant_weights) {

    const int gindex = iq3_data_index(grid_size);

    const uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    const int      * kmap_q3xs       = iq3_data[gindex].map;
    const uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n/QK_K;

    ggml_fp16_t * dh;
    uint8_t * qs;
    int block_size;
    if (grid_size == 256) {
        block_iq3_xxs * y = vy;
        dh = &y->d;
        qs = y->qs;
        block_size = sizeof(block_iq3_xxs);
    } else {
        block_iq3_s * y = vy;
        dh = &y->d;
        qs = y->qs;
        block_size = sizeof(block_iq3_s);
    }
    int quant_size = block_size - sizeof(ggml_fp16_t);

    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float  waux[32];
    bool   is_on_grid[8];
    bool   is_on_grid_aux[8];
    uint8_t block_signs[8];
    uint8_t q3[3*(QK_K/8)+QK_K/32];
    uint32_t * scales_and_signs = (uint32_t *)(q3 + QK_K/4);
    uint8_t  * qh = q3 + 3*(QK_K/8);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        dh[0] = GGML_FP32_TO_FP16(0.f);
        memset(q3, 0, 3*QK_K/8+QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 32*ib;
                for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < 32; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            if (max < GROUP_MAX_EPS_IQ3_XXS) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int is = -15; is <= 15; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 8; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 32; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  8; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 8; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 8; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    GGML_ASSERT(false);
                }
                if (grid_size == 256) {
                    q3[8*ib+k] = grid_index;
                } else {
                    q3[8*ib+k] = grid_index & 255;
                    qh[ib] |= ((grid_index >> 8) << k);
                }

            }
            scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(qs, 0, quant_size);
            dh += block_size/sizeof(ggml_fp16_t);
            qs += block_size;
            continue;
        }

        float d = max_scale/31;
        dh[0] = GGML_FP32_TO_FP16(d * 1.0125f);  // small improvement via this fudge factor
        float id = 1/d;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            scales_and_signs[ib] |= ((uint32_t)l << 28);
        }
        memcpy(qs, q3, quant_size);

        dh += block_size/sizeof(ggml_fp16_t);
        qs += block_size;

    }
}

static inline int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static void quantize_row_iq4_nl_impl(const int super_block_size, const int block_size, const float * restrict x,
        ggml_fp16_t * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(q4, 0, super_block_size/2);
    dh[0] = GGML_FP32_TO_FP16(0.f);

    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        uint8_t * Lb = L + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (amax < GROUP_MAX_EPS) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        float id = 1/d;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < block_size; ++j) {
            float al = id*xb[j];
            int l = best_index_int8(16, values, al);
            Lb[j] = l;
            float q = values[l];
            float w = weight[j];
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        d = sumqx/sumq2;
        float best = d*sumqx;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
            sumqx = sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                float al = id*xb[j];
                int l = best_index_int8(16, values, al);
                float q = values[l];
                float w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                d = sumqx/sumq2; best = d * sumqx;
            }
        }
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    if (super_block_size/block_size > 1) {
        int nb = super_block_size/block_size;
        memset(scales_h, 0, ((nb+7)/8)*sizeof(uint16_t));
        float d = -max_scale/32;
        dh[0] = GGML_FP32_TO_FP16(d);
        float id = d ? 1/d : 0.f;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            int l = nearest_int(id*scales[ib]);
            l = MAX(-32, MIN(31, l));
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            uint8_t * Lb = L + ib*block_size;
            const float * xb = x + ib*block_size;
            for (int j = 0; j < block_size; ++j) {
                Lb[j] = best_index_int8(16, values, idl*xb[j]);
            }
            l += 32;
            uint8_t l_l = l & 0xf;
            uint8_t l_h = l >>  4;
            if (ib%2 == 0) scales_l[ib/2] = l_l;
            else scales_l[ib/2] |= (l_l << 4);
            scales_h[ib/8] |= (l_h << 2*(ib%8));
        }
    } else {
        dh[0] = GGML_FP32_TO_FP16(scales[0]);
        if (ntry > 0) {
            float id = scales[0] ? 1/scales[0] : 0;
            for (int j = 0; j < super_block_size; ++j) {
                L[j] = best_index_int8(16, values, id*x[j]);
            }
        }
    }

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

void quantize_row_q6_K_reference(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
void quantize_row_q6_K_reference(const float * restrict x, block_q6_K * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * restrict ql = y[i].ql;
        uint8_t * restrict qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

size_t quantize_iq4_xs(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    uint8_t L[QK_K];
    float weight[32];
    float scales[QK_K/32];
    for (int64_t row = 0; row < nrow; ++row) {
        block_iq4_xs * iq4 = (block_iq4_xs *)qrow;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw = quant_weights ? quant_weights + QK_K*ibl : NULL;
            quantize_row_iq4_nl_impl(QK_K, 32, src + QK_K*ibl, &iq4[ibl].d, iq4[ibl].qs, &iq4[ibl].scales_h, iq4[ibl].scales_l,
                    scales, weight, L, kvalues_iq4nl, qw, 7);
        }
        src += n_per_row;
        qrow += nblock*sizeof(block_iq4_xs);
    }
    return nrow * nblock * sizeof(block_iq4_xs);
}

void quantize_row_q8_K_reference(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K_reference(const float * restrict x, block_q8_K * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

static void quantize_row_iq3_s_impl(int block_size, const float * restrict x, void * restrict vy, int n,
        const float * restrict quant_weights,
        float   * scales,
        float   * weight,
        float   * xval,
        int8_t  * L,
        int8_t  * Laux,
        float   * waux,
        bool    * is_on_grid,
        bool    * is_on_grid_aux,
        uint8_t * block_signs) {

    const int gindex = iq3_data_index(512);

    const uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    const int      * kmap_q3xs       = iq3_data[gindex].map;
    const uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n/QK_K;

    block_iq3_s * y = vy;

    const int bs4 = block_size/4;
    const int bs8 = block_size/8;

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        uint8_t * qs = y[ibl].qs;
        uint8_t * qh = y[ibl].qh;
        uint8_t * signs = y[ibl].signs;

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < block_size; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < bs8; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < block_size; ++i) max = MAX(max, xval[i]);
            if (!max) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int k = 0; k < bs4; ++k) is_on_grid[k] = false;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < bs4; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < block_size; ++i) L[i] = Laux[i];
                    for (int k = 0; k < bs4; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < bs4; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < bs4; ++k) {
                    //if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < bs8; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < bs4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    GGML_ASSERT(false);
                }
                qs[k] = grid_index & 255;
                qh[(ib*bs4+k)/8] |= ((grid_index >> 8) << ((ib*bs4+k)%8));
            }
            qs += bs4;
            for (int k = 0; k < bs8; ++k) signs[k] = block_signs[k];
            signs += bs8;
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d * 1.033f);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/block_size; ib += 2) {
            int l1 = nearest_int(0.5f*(id*scales[ib+0]-1));
            l1 = MAX(0, MIN(15, l1));
            int l2 = nearest_int(0.5f*(id*scales[ib+1]-1));
            l2 = MAX(0, MIN(15, l2));
            y[ibl].scales[ib/2] = l1 | (l2 << 4);
        }

    }
}

#define IQ3S_BLOCK_SIZE 32
size_t quantize_iq3_s(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    float scales[QK_K/IQ3S_BLOCK_SIZE];
    float weight[IQ3S_BLOCK_SIZE];
    float xval[IQ3S_BLOCK_SIZE];
    int8_t L[IQ3S_BLOCK_SIZE];
    int8_t Laux[IQ3S_BLOCK_SIZE];
    float  waux[IQ3S_BLOCK_SIZE];
    bool   is_on_grid[IQ3S_BLOCK_SIZE/4];
    bool   is_on_grid_aux[IQ3S_BLOCK_SIZE/4];
    uint8_t block_signs[IQ3S_BLOCK_SIZE/8];
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq3_s_impl(IQ3S_BLOCK_SIZE, src, qrow, n_per_row, quant_weights,
                scales, weight, xval, L, Laux, waux, is_on_grid, is_on_grid_aux, block_signs);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq3_s);
    }
    return nrow * nblock * sizeof(block_iq3_s);
}

void quantize_row_iq3_xxs_reference(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
void quantize_row_iq3_xxs_reference(const float * restrict x, block_iq3_xxs * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_row_iq3_xxs_impl(256, x, y, k, NULL);
}

void quantize_row_iq4_nl_reference (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);

void quantize_row_iq4_nl_reference(const float * restrict x, block_iq4_nl * restrict y, int64_t k) {
    assert(k % QK4_NL == 0);
    quantize_row_iq4_nl(x, y, k);
}

typedef struct {
    uint64_t * grid;
    int      * map;
    uint16_t * neighbours;
} iq2_entry_t;

static iq2_entry_t iq2_data[4] = {
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
};

static inline int iq2_data_index(enum ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_S);
    return type == GGML_TYPE_IQ2_XXS ? 0 :
           type == GGML_TYPE_IQ2_XS  ? 1 :
           type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ? 2 : 3;
}

static int iq2_find_best_neighbour(const uint16_t * restrict neighbours, const uint64_t * restrict grid,
        const float * restrict xval, const float * restrict weight, float scale, int8_t * restrict L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

static void quantize_row_iq2_s_impl(const float * restrict x, void * restrict vy, int64_t n, const float * restrict quant_weights) {

    const int gindex = iq2_data_index(GGML_TYPE_IQ2_S);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    GGML_ASSERT(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n/QK_K;

    block_iq2_s * y = vy;

    float scales[QK_K/16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 16*ib;
                for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < 16; ++i) weight[i] = 0.25f*sigma2 + xb[i]*xb[i];
            }
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = MAX(max, xval[i]);
            if (max < GROUP_MAX_EPS_IQ2_S) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            is_on_grid[0] = is_on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                        L[8*k + i] = l;
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    GGML_ASSERT(false);
                }
                const int i8 = 2*ib + k;
                y[ibl].qs[i8] = grid_index & 255;
                y[ibl].qh[i8/4] |= ((grid_index >> 8) << 2*(i8%4));
                y[ibl].qs[QK_K/8 + i8] = block_signs[k];
            }
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d * 0.9875f);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            if (ib%2 == 0) y[ibl].scales[ib/2] = l;
            else y[ibl].scales[ib/2] |= (l << 4);
        }
    }
}

size_t quantize_iq2_s(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq2_s_impl(src, qrow, n_per_row, quant_weights);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq2_s);
    }
    return nrow * nblock * sizeof(block_iq2_s);
}

void quantize_row_iq4_xs_reference (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
void quantize_row_iq4_xs_reference(const float * restrict x, block_iq4_xs * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, NULL);
}

void quantize_row_iq3_s_reference  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
void quantize_row_iq3_s_reference(const float * restrict x, block_iq3_s * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq3_s(x, y, 1, k, NULL);
}

void quantize_row_iq2_s_reference  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);
void quantize_row_iq2_s_reference(const float * restrict x, block_iq2_s * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_s(x, y, 1, k, NULL);
}

void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q4_0_reference(x, y, k);
}

void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_1(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q4_1_reference(x, y, k);
}

void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q5_0_reference(x, y, k);
}

void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_1(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q5_1_reference(x, y, k);
}

void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0(const float * restrict x, void * restrict vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
        }
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = wasm_v128_load(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(wasm_f32x4_extract_lane(amaxv[0], 0),
                                   wasm_f32x4_extract_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_extract_lane(amaxv[0], 2),
                                   wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_extract_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_extract_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_extract_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_extract_lane(vi, 3);
        }
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#elif defined(__riscv_v_intrinsic)

    size_t vl = __riscv_vsetvl_e32m4(QK8_0);

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(x+i*QK8_0, vl);

        vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m4_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        vfloat32m4_t x0 = __riscv_vfmul_vf_f32m4(v_x, id, vl);

        // convert to integer
        vint16m2_t   vi = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
        vint8m1_t    vs = __riscv_vncvt_x_x_w_i8m1(vi, vl);

        // store result
        __riscv_vse8_v_i8m1(y[i].qs , vs, vl);
    }

#elif defined(__POWER9_VECTOR__)
    for (int i = 0; i < nb; i++) {
        vector float srcv [8];
        vector float asrcv[8];
        vector float amaxv[8];
        vector signed int vi[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        const vector float vid = vec_splats(id);

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const vector float v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

#elif defined(__loongarch_asx)
    for (int i = 0; i < nb; i++) {
        ft_union fi;
        __m256 v0 = (__m256)__lasx_xvld( x , 0);
        __m256 v1 = (__m256)__lasx_xvld( x , 32);
        __m256 v2 = (__m256)__lasx_xvld( x , 64);
        __m256 v3 = (__m256)__lasx_xvld( x , 96);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 sign_bit = __lasx_xvreplfr2vr_s( -0.0f );
        __m256 max_abs = (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v0 );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v1 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v2 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v3 ) );

        __m128 max4 = __lsx_vfmax_s( lasx_extractf128( max_abs, 1 ), lasx_extractf128( max_abs , 0) );
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4 ) );
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vinsgr2vr_w(tmp, __lsx_vpickve2gr_w( max4, 1 ), 0 ));
        fi.i = __lsx_vpickve2gr_w( (__m128i)max4, 0 );
        const float max_scalar = fi.f;

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = (__m256)__lasx_xvreplfr2vr_s( id );

        // Apply the multiplier
        v0 = __lasx_xvfmul_s( v0, mul );
        v1 = __lasx_xvfmul_s( v1, mul );
        v2 = __lasx_xvfmul_s( v2, mul );
        v3 = __lasx_xvfmul_s( v3, mul );

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s( v0 );
        __m256i i1 = __lasx_xvftintrne_w_s( v1 );
        __m256i i2 = __lasx_xvftintrne_w_s( v2 );
        __m256i i3 = __lasx_xvftintrne_w_s( v3 );

        __m128i ni0 = lasx_extracti128( i0, 0 );
        __m128i ni1 = lasx_extracti128( i0, 1);
        __m128i ni2 = lasx_extracti128( i1, 0);
        __m128i ni3 = lasx_extracti128( i1, 1);
        __m128i ni4 = lasx_extracti128( i2, 0);
        __m128i ni5 = lasx_extracti128( i2, 1);
        __m128i ni6 = lasx_extracti128( i3, 0);
        __m128i ni7 = lasx_extracti128( i3, 1);

        // Convert int32 to int16
        ni0 = lsx_packs_w( ni0, ni1 );
        ni2 = lsx_packs_w( ni2, ni3 );
        ni4 = lsx_packs_w( ni4, ni5 );
        ni6 = lsx_packs_w( ni6, ni7 );
        // Convert int16 to int8
        ni0 = lsx_packs_h( ni0, ni2 );
        ni4 = lsx_packs_h( ni4, ni6 );

        __lsx_vst(ni0, (__m128i *)(y[i].qs +  0), 0);
        __lsx_vst(ni4, (__m128i *)(y[i].qs + 16), 0);

    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_reference(x, y, k);
#endif
}

void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    block_q8_1 * restrict y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);

            accv = vaddq_s32(accv, vi);
        }

        y[i].s = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = wasm_v128_load(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(wasm_f32x4_extract_lane(amaxv[0], 0),
                                   wasm_f32x4_extract_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_extract_lane(amaxv[0], 2),
                                   wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        v128_t accv = wasm_i32x4_splat(0);

        for (int j = 0; j < 8; j++) {
            const v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_extract_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_extract_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_extract_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_extract_lane(vi, 3);

            accv = wasm_i32x4_add(accv, vi);
        }

        y[i].s = GGML_FP32_TO_FP16(
                d * (wasm_i32x4_extract_lane(accv, 0) +
                     wasm_i32x4_extract_lane(accv, 1) +
                     wasm_i32x4_extract_lane(accv, 2) +
                     wasm_i32x4_extract_lane(accv, 3)));
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float max_scalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Compute the sum of the quants and set y[i].s
        y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))));

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Compute the sum of the quants and set y[i].s
        const __m128i s0 = _mm_add_epi32(_mm_add_epi32(ni0, ni1), _mm_add_epi32(ni2, ni3));
        const __m128i s1 = _mm_add_epi32(_mm_add_epi32(ni4, ni5), _mm_add_epi32(ni6, ni7));
        y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_4(_mm_add_epi32(s0, s1)));

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#elif defined(__riscv_v_intrinsic)

    size_t vl = __riscv_vsetvl_e32m4(QK8_1);

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(x+i*QK8_1, vl);

        vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m4_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d  = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        vfloat32m4_t x0 = __riscv_vfmul_vf_f32m4(v_x, id, vl);

        // convert to integer
        vint16m2_t   vi = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
        vint8m1_t    vs = __riscv_vncvt_x_x_w_i8m1(vi, vl);

        // store result
        __riscv_vse8_v_i8m1(y[i].qs , vs, vl);

        // compute sum for y[i].s
        vint16m1_t tmp2 = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t vwrs = __riscv_vwredsum_vs_i8m1_i16m1(vs, tmp2, vl);

        // set y[i].s
        int sum = __riscv_vmv_x_s_i16m1_i16(vwrs);
        y[i].s = GGML_FP32_TO_FP16(sum*d);
    }

#elif defined(__POWER9_VECTOR__)
    for (int i = 0; i < nb; i++) {
        vector float srcv [8];
        vector float asrcv[8];
        vector float amaxv[8];
        vector signed int vi[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        const vector float vid = vec_splats(id);

        y[i].d = GGML_FP32_TO_FP16(d);

        vector int accv = vec_splats(0);

        for (int j = 0; j < 8; j++) {
            const vector float v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_add(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_add(accv, vec_sld(accv, accv, 4));
        accv = vec_add(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_FP32_TO_FP16(d * vec_extract(accv, 0));

#elif defined(__loongarch_asx)
    for (int i = 0; i < nb; i++) {
        ft_union ft;
        __m256 v0 = (__m256)__lasx_xvld( x , 0 );
        __m256 v1 = (__m256)__lasx_xvld( x , 32 );
        __m256 v2 = (__m256)__lasx_xvld( x , 64 );
        __m256 v3 = (__m256)__lasx_xvld( x , 96 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 sign_bit = __lasx_xvreplfr2vr_s( -0.0f );
        __m256 max_abs = (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v0 );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v1 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v2 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v3 ) );

        __m128 max4 = __lsx_vfmax_s( lasx_extractf128( max_abs, 1 ), lasx_extractf128( max_abs, 0) );
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4 ) );
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vextrins_w((__m128i)tmp, (__m128i)max4, 0x10 ));
        ft.i = __lsx_vpickve2gr_w( (__m128i)max4, 0 );
        const float max_scalar = ft.f;

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = __lasx_xvreplfr2vr_s( id );

        // Apply the multiplier
        v0 = __lasx_xvfmul_s( v0, mul );
        v1 = __lasx_xvfmul_s( v1, mul );
        v2 = __lasx_xvfmul_s( v2, mul );
        v3 = __lasx_xvfmul_s( v3, mul );

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s( v0 );
        __m256i i1 = __lasx_xvftintrne_w_s( v1 );
        __m256i i2 = __lasx_xvftintrne_w_s( v2 );
        __m256i i3 = __lasx_xvftintrne_w_s( v3 );

        __m128i ni0 = lasx_extracti128(i0, 0);
        __m128i ni1 = lasx_extracti128( i0, 1);
        __m128i ni2 = lasx_extracti128( i1, 0);
        __m128i ni3 = lasx_extracti128( i1, 1);
        __m128i ni4 = lasx_extracti128( i2, 0 );
        __m128i ni5 = lasx_extracti128( i2, 1);
        __m128i ni6 = lasx_extracti128( i3, 0);
        __m128i ni7 = lasx_extracti128( i3, 1);

        // Compute the sum of the quants and set y[i].s
        const __m128i s0 = __lsx_vadd_w(__lsx_vadd_w(ni0, ni1), __lsx_vadd_w(ni2, ni3));
        const __m128i s1 = __lsx_vadd_w(__lsx_vadd_w(ni4, ni5), __lsx_vadd_w(ni6, ni7));
        y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_4(__lsx_vadd_w(s0, s1)));

        // Convert int32 to int16
        ni0 = lsx_packs_w( ni0, ni1 );
        ni2 = lsx_packs_w( ni2, ni3 );
        ni4 = lsx_packs_w( ni4, ni5 );
        ni6 = lsx_packs_w( ni6, ni7 );
        // Convert int16 to int8
        ni0 = lsx_packs_h( ni0, ni2 );
        ni4 = lsx_packs_h( ni4, ni6 );

        __lsx_vst(ni0, (__m128i *)(y[i].qs +  0), 0);
        __lsx_vst(ni4, (__m128i *)(y[i].qs + 16), 0);
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_reference(x, y, k);
#endif
}

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q2_K(const float * restrict x, void * restrict vy, int64_t k) {
    quantize_row_q2_K_reference(x, vy, k);
}

void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q3_K(const float * restrict x, void * restrict vy, int64_t k) {
    quantize_row_q3_K_reference(x, vy, k);
}

void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_K(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q4_K * restrict y = vy;
    quantize_row_q4_K_reference(x, y, k);
}

void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_K(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q5_K * restrict y = vy;
    quantize_row_q5_K_reference(x, y, k);
}

void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q6_K(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q6_K * restrict y = vy;
    quantize_row_q6_K_reference(x, y, k);
}

void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q8_K_reference(x, y, k);
}

void quantize_row_iq3_xxs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq3_xxs(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq3_xxs * restrict y = vy;
    quantize_row_iq3_xxs_reference(x, y, k);
}

void quantize_row_iq4_nl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq4_nl(const float * restrict x, void * restrict vy, int64_t k) {
    GGML_ASSERT(k%QK4_NL == 0);
    int64_t nblock = k/QK4_NL;
    uint8_t L[QK4_NL];
    float weight[QK4_NL];
    uint16_t unused_h;
    uint8_t * unused_l = NULL;
    float scale;
    block_iq4_nl * iq4 = (block_iq4_nl *)vy;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        quantize_row_iq4_nl_impl(QK4_NL, 32, x + QK4_NL*ibl, &iq4[ibl].d, iq4[ibl].qs, &unused_h, unused_l,
                &scale, weight, L, kvalues_iq4nl, NULL, -1);
    }
}

void quantize_row_iq4_xs (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq4_xs(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq4_xs * restrict y = vy;
    quantize_row_iq4_xs_reference(x, y, k);
}

void quantize_row_iq3_s  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq3_s(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq3_s * restrict y = vy;
    quantize_row_iq3_s_reference(x, y, k);
}

void quantize_row_iq2_s  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq2_s(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_s * restrict y = vy;
    quantize_row_iq2_s_reference(x, y, k);
}

// Dequantization
void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q4_0(const block_q4_0 * restrict x, float * restrict y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q4_1(const block_q4_1 * restrict x, float * restrict y, int64_t k) {
    static const int qk = QK4_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        //const float d = GGML_FP16_TO_FP32(x[i].d);
        //const float m = GGML_FP16_TO_FP32(x[i].m);
        const float d = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.d);
        const float m = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.m);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q5_0(const block_q5_0 * restrict x, float * restrict y, int64_t k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q5_1(const block_q5_1 * restrict x, float * restrict y, int64_t k) {
    static const int qk = QK5_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        //const float d = GGML_FP16_TO_FP32(x[i].d);
        //const float m = GGML_FP16_TO_FP32(x[i].m);
        const float d = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.d);
        const float m = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.m);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >>   4) | xh_1;

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q8_0(const block_q8_0 * restrict x, float * restrict y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

//void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q2_K(const block_q2_K * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        //const float d = GGML_FP16_TO_FP32(x[i].d);
        //const float min = GGML_FP16_TO_FP32(x[i].dmin);
        const float d = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.d);
        const float min = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.dmin);

        const uint8_t * q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q3_K(const block_q3_K * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t * scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }
}

void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q4_K(const block_q4_K * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;

        //const float d   = GGML_FP16_TO_FP32(x[i].d);
        //const float min = GGML_FP16_TO_FP32(x[i].dmin);
        const float d   = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.d);
        const float min = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q5_K(const block_q5_K * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;

        //const float d = GGML_FP16_TO_FP32(x[i].d);
        //const float min = GGML_FP16_TO_FP32(x[i].dmin);
        const float d = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.d);
        const float min = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q6_K(const block_q6_K * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict ql = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_q8_K(const block_q8_K * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq2_xxs(const block_iq2_xxs * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            /*
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            */
        }
    }
}

void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq2_xs(const block_iq2_xs * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            /*
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
                const uint8_t  signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            */
        }
    }
}

void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq2_s(const block_iq2_s * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            /*
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l/2];
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            */
            qs += 4;
            signs += 4;
        }
    }
}

void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq3_xxs(const block_iq3_xxs * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * scales_and_signs = qs + QK_K/4;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            /*
            for (int l = 0; l < 4; ++l) {
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            */
            qs += 8;
        }
    }
}

void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq1_s(const block_iq1_s * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl = d * (2*((qh[ib] >> 12) & 7) + 1);
            
            /*
            const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (grid[j] + delta);
                }
                y += 8;
            }
            */
            qs += 4;
        }
    }
}

void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq1_m(const block_iq1_m * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float delta[4];
    uint16_t idx[4];

    iq1m_scale_t scale;

    for (int i = 0; i < nb; i++) {

        const uint16_t * sc = (const uint16_t *)x[i].scales;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = GGML_FP16_TO_FP32(scale.f16);

        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
            const float dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);

            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);

            /*
            delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 2; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl1 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl2 * (grid[j] + delta[l]);
                }
                y += 8;
            }
            */
            qs += 4;
            qh += 2;
        }
    }
}

void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq4_nl(const block_iq4_nl * restrict x, float * restrict y, int64_t k) {
    assert(k % QK4_NL == 0);
    const int64_t nb = k / QK4_NL;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK4_NL/2; ++j) {
            y[j+       0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j+QK4_NL/2] = d * kvalues_iq4nl[qs[j] >>  4];
        }
        y  += QK4_NL;
        qs += QK4_NL/2;
    }
}

void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq4_xs(const block_iq4_xs * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j+16] = dl * kvalues_iq4nl[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq3_s(const block_iq3_s * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;

        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            /*
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            */
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}


// Dot product
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);

    assert(nrc == 1);

    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        int sumi = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[i].qs[j] & 0x0F) - 8;
            const int v1 = (x[i].qs[j] >>   4) - 8;

            sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk/2]);
        }

        sumf += sumi*GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d);
    }

    *s = sumf;
}

void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);

    assert(nrc == 1);

    const block_q4_1 * restrict x = vx;
    const block_q8_1 * restrict y = vy;

    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    /*
    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        //memcpy(auxs, x[i].scales, 12);
        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
    */
}

void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

     // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        int sumi = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk/2]);
        }

        sumf += (GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d)) * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(qk == QK5_1);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_1 * restrict x = vx;
    const block_q8_1 * restrict y = vy;

    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        int sumi = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[i].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[i].qs[j] >>  4) | xh_1;

            sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk/2]);
        }

        //sumf += (GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d))*sumi + GGML_FP16_TO_FP32(x[i].m)*GGML_FP16_TO_FP32(y[i].s);
        sumf += (GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.d)*GGML_FP16_TO_FP32(y[i].GGML_COMMON_AGGR.d))*sumi + GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.m)*GGML_FP16_TO_FP32(y[i].GGML_COMMON_AGGR.s);
    }

    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);

    assert(nrc == 1);

    const block_q8_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[i].qs[j]*y[i].qs[j];
        }

        sumf += sumi*(GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d));
    }

    *s = sumf;
}

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q2_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(nrc == 1);

    const block_q2_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        //const float dall = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        //const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);
        const float dall = y[i].d * GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);

    const block_q4_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

     const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        //const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const float d = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        //const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
        const float dmin = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}


void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy,  size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];


    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const uint8_t * restrict hm = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        //const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const float d = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        //const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
        const float dmin = GGML_FP16_TO_FP32(x[i].GGML_COMMON_AGGR.dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xxs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * restrict x = vx;
    const block_q8_K    * restrict y = vy;

    const int nb = n / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            /*
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            */
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * restrict x = vx;
    const block_q8_K   * restrict y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const uint8_t  * restrict sc = x[i].scales;
        const int8_t   * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            /*
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            */
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_s * restrict x = vx;
    const block_q8_K  * restrict y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            int ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            int ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            int sumi1 = 0, sumi2 = 0;
            /*
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            */
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }

        sumf += d * bsum;
    }

    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * restrict x = vx;
    const block_q8_K    * restrict y = vy;

    const int nb = n / QK_K;

    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict gas = x[i].qs + QK_K/4;
        const int8_t  * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            /*
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            */
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
}

void ggml_vec_dot_iq1_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_s_q8_K  (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_s * restrict x = vx;
    const block_q8_K  * restrict y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        int sumi = 0, sumi1 = 0;
        /*
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = 2*((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += GGML_FP16_TO_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
        */
    }

    *s = sumf;
}

void ggml_vec_dot_iq1_m_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_m_q8_K  (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_m * restrict x = vx;
    const block_q8_K  * restrict y = vy;

    const int nb = n / QK_K;

    iq1m_scale_t scale;

    int sum1[2], sum2[2], delta[4];

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int sumi1 = 0, sumi2 = 0;
        /*
        for (int ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            const int ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            const int ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += GGML_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
        */
    }

    *s = sumf;
}

void ggml_vec_dot_iq4_nl_q8_0 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    const block_iq4_nl * restrict x = vx;
    const block_q8_0   * restrict y = vy;

    const int nb = n / QK4_NL;

    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        const float d = GGML_FP16_TO_FP32(y[ib].d)*GGML_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_iq4_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * restrict x = vx;
    const block_q8_K   * restrict y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8*(ls1 - 32);
            const float d2 = d4d8*(ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
}


void ggml_vec_dot_iq3_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * restrict x = vx;
    const block_q8_K  * restrict y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict qs = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const uint8_t * restrict signs = x[i].signs;
        const int8_t  * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        /*
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            const uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum;

        */
    }
    *s = sumf;
}

static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = sizeof(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = sizeof(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = sizeof(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = sizeof(double),
        .is_quantized             = false,
        .nrows                    = 1,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = sizeof(float),
        .is_quantized             = false,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = sizeof(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float               = (ggml_from_float_t) ggml_fp32_to_fp16_row,
        .from_float_reference     = (ggml_from_float_t) ggml_fp32_to_fp16_row,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float               = quantize_row_q4_0,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q4_0_reference,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = sizeof(block_q4_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_1,
        .from_float               = quantize_row_q4_1,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q4_1_reference,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = GGML_TYPE_COUNT,
        .nrows                    = 1,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = GGML_TYPE_COUNT,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_size                = QK5_0,
        .type_size                = sizeof(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float               = quantize_row_q5_0,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q5_0_reference,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = sizeof(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float               = quantize_row_q5_1,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q5_1_reference,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = sizeof(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float               = quantize_row_q8_0,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q8_0_reference,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = sizeof(block_q8_1),
        .is_quantized             = true,
        .from_float               = quantize_row_q8_1,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q8_1_reference,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float               = quantize_row_q2_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q2_K_reference,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float               = quantize_row_q3_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q3_K_reference,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float               = quantize_row_q4_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q4_K_reference,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float               = quantize_row_q5_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q5_K_reference,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float               = quantize_row_q6_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q6_K_reference,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float               = quantize_row_iq3_xxs,
        .from_float_reference     = (ggml_from_float_t)quantize_row_iq3_xxs_reference,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float               = quantize_row_iq3_s,
        .from_float_reference     = (ggml_from_float_t)quantize_row_iq3_s_reference,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float               = quantize_row_iq2_s,
        .from_float_reference     = (ggml_from_float_t)quantize_row_iq2_s_reference,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = sizeof(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float               = quantize_row_iq4_nl,
        .from_float_reference     = (ggml_from_float_t)quantize_row_iq4_nl_reference,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float               = quantize_row_iq4_xs,
        .from_float_reference     = (ggml_from_float_t)quantize_row_iq4_xs_reference,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q8_K),
        .is_quantized             = true,
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = sizeof(ggml_bf16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_bf16_to_fp32_row,
        .from_float               = (ggml_from_float_t) ggml_fp32_to_bf16_row,
        .from_float_reference     = (ggml_from_float_t) ggml_fp32_to_bf16_row,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    }
};

struct ggml_tensor {
        enum ggml_type         type;

        GGML_DEPRECATED(enum ggml_backend_type backend, "use the buffer type to find the storage location of the tensor");

        struct ggml_backend_buffer * buffer;

        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

        struct ggml_tensor * grad;
        struct ggml_tensor * src[GGML_MAX_SRC];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;

        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[8];
};

static const size_t GGML_TENSOR_SIZE = sizeof(struct ggml_tensor);

GGML_CALL size_t ggml_type_size(enum ggml_type type) {
    return type_traits[type].type_size;
}

GGML_CALL size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    assert(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type)*ne/ggml_blck_size(type);
}

static struct ggml_tensor * ggml_new_tensor_impl(
        struct ggml_context * ctx,
        enum   ggml_type      type,
        int                   n_dims,
        const int64_t       * ne,
        struct ggml_tensor  * view_src,
        size_t                view_offs) {

    //assert(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    if (view_src != NULL && view_src->view_src != NULL) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    size_t data_size = ggml_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));

    void * data = view_src != NULL ? view_src->data : NULL;
    if (data != NULL) {
        data = (char *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    if (view_src == NULL && !ctx->no_alloc) {
        if (ctx->scratch.data != NULL) {
            // allocate tensor data in the scratch buffer
            if (ctx->scratch.offs + data_size > ctx->scratch.size) {
                GGML_PRINT("%s: not enough space in the scratch memory pool (needed %zu, available %zu)\n",
                        __func__, ctx->scratch.offs + data_size, ctx->scratch.size);
                assert(false);
                return NULL;
            }

            data = (char * const) ctx->scratch.data + ctx->scratch.offs;

            ctx->scratch.offs += data_size;
        } else {
            // allocate tensor data in the context's memory pool
            obj_alloc_size = data_size;
        }
    }

    struct ggml_object * const obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);

    // TODO: for recoverable errors, we would need to free the data allocated from the scratch buffer here

    struct ggml_tensor *  result = (struct ggml_tensor *)((char *)ctx->mem_buffer + obj_new->offs);

#ifdef __clang__
    // temporary until ggml_tensor::backend is removed
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

    *result = (struct ggml_tensor) {
        /*.type         =*/ type,
        /*.backend      =*/ GGML_BACKEND_TYPE_CPU,
        /*.buffer       =*/ NULL,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,  
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.grad         =*/ NULL,
        /*.src          =*/ { NULL },
        /*.perf_runs    =*/ 0,
        /*.perf_cycles  =*/ 0,
        /*.perf_time_us =*/ 0,
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (void *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ NULL,
        /*.padding      =*/ { 0 },
    };

#ifdef __clang__
    #pragma clang diagnostic pop
#endif

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //ggml_assert_aligned(result->data);

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    result->op = GGML_OP_ADD;

    return result;
}


struct ggml_state {
    struct ggml_context_container contexts[GGML_MAX_CONTEXTS];
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state;

extern float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_f16[1 << 16];

// precomputed gelu table for f16 (128 KB)
static ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
static ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

//struct ggml_tensor;
typedef void * ggml_backend_graph_plan_t;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_buffer_type;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef uint8_t ggml_guid[16];
typedef ggml_guid * ggml_guid_t;

struct ggml_backend_event {
        ggml_backend_t backend;
        void * context;
};

typedef struct ggml_backend_event * ggml_backend_event_t;

struct ggml_hash_set {
    size_t size;
    struct ggml_tensor ** keys;
};

typedef void * ggml_backend_context_t;

enum ggml_cgraph_eval_order {
        GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
        GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
        GGML_CGRAPH_EVAL_ORDER_COUNT
};

enum ggml_status {
    GGML_STATUS_ALLOC_FAILED = -2,
    GGML_STATUS_FAILED = -1,
    GGML_STATUS_SUCCESS = 0,
    GGML_STATUS_ABORTED = 1,
};

struct ggml_cgraph {
        int size;
        int n_nodes;
        int n_leafs;

        struct ggml_tensor ** nodes;
        struct ggml_tensor ** grads;
        struct ggml_tensor ** leafs;

        struct ggml_hash_set visited_hash_table;

        enum ggml_cgraph_eval_order order;

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;
};

struct ggml_backend_i {
        const char * (*GGML_CALL get_name)(ggml_backend_t backend);

        void (*GGML_CALL free)(ggml_backend_t backend);

        // buffer allocation
        ggml_backend_buffer_type_t (*GGML_CALL get_default_buffer_type)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*GGML_CALL set_tensor_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*GGML_CALL get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool (*GGML_CALL cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        // (optional) complete all pending operations
        void (*GGML_CALL synchronize)(ggml_backend_t backend);

        // compute graph with a plan (not used currently)
        ggml_backend_graph_plan_t (*GGML_CALL graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*GGML_CALL graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph with a plan
        enum ggml_status (*GGML_CALL graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // compute graph without a plan (async)
        enum ggml_status (*GGML_CALL graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*GGML_CALL supports_op)(ggml_backend_t backend, const struct ggml_tensor * op);

        // check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
        // these should be expensive operations with large batch sizes that may benefit from running on this backend
        // even if the weight has to be copied from the CPU temporarily
        bool (*GGML_CALL offload_op)(ggml_backend_t backend, const struct ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*GGML_CALL event_new)         (ggml_backend_t backend);
        void                 (*GGML_CALL event_free)        (ggml_backend_event_t event);
        void                 (*GGML_CALL event_record)      (ggml_backend_event_t event);
        void                 (*GGML_CALL event_wait)        (ggml_backend_t backend, ggml_backend_event_t event);
        void                 (*GGML_CALL event_synchronize) (ggml_backend_event_t event);
};

struct ggml_backend {
        ggml_guid_t guid;

        struct ggml_backend_i iface;
        ggml_backend_context_t context;
};

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,       // marks the end of the enum
};

enum ggml_backend_buffer_usage {
    GGML_BACKEND_BUFFER_USAGE_ANY = 0,
    GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
};

struct gguf_str {
    uint64_t n;  // GGUFv2
    char * data;
};

static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8]   = sizeof(uint8_t),
    [GGUF_TYPE_INT8]    = sizeof(int8_t),
    [GGUF_TYPE_UINT16]  = sizeof(uint16_t),
    [GGUF_TYPE_INT16]   = sizeof(int16_t),
    [GGUF_TYPE_UINT32]  = sizeof(uint32_t),
    [GGUF_TYPE_INT32]   = sizeof(int32_t),
    [GGUF_TYPE_FLOAT32] = sizeof(float),
    [GGUF_TYPE_BOOL]    = sizeof(bool),
    [GGUF_TYPE_STRING]  = sizeof(struct gguf_str),
    [GGUF_TYPE_UINT64]  = sizeof(uint64_t),
    [GGUF_TYPE_INT64]   = sizeof(int64_t),
    [GGUF_TYPE_FLOAT64] = sizeof(double),
    [GGUF_TYPE_ARRAY]   = 0, // undefined
};

typedef void * ggml_backend_buffer_type_context_t;
typedef void * ggml_backend_buffer_context_t;
struct ggml_backend_buffer;

typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

//x
GGML_API ggml_backend_buffer_t ggml_backend_reg_alloc_buffer(size_t i, size_t size);
//x

struct ggml_backend_buffer_i {
    const char * (*GGML_CALL get_name)   (ggml_backend_buffer_t buffer);
    void         (*GGML_CALL free_buffer)(ggml_backend_buffer_t buffer);
    void *       (*GGML_CALL get_base)   (ggml_backend_buffer_t buffer);
    void         (*GGML_CALL init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
    void         (*GGML_CALL set_tensor) (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void         (*GGML_CALL get_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    bool         (*GGML_CALL cpy_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst); // dst is in the buffer, src may be in any buffer
    void         (*GGML_CALL clear)      (ggml_backend_buffer_t buffer, uint8_t value);
    void         (*GGML_CALL reset)      (ggml_backend_buffer_t buffer); //reset any internal state due to tensor initialization, such as tensor extras
};

struct ggml_backend_buffer {
    struct ggml_backend_buffer_i  iface;
    ggml_backend_buffer_type_t    buft;
    ggml_backend_buffer_context_t context;
    size_t size;
    enum ggml_backend_buffer_usage usage;
};

struct ggml_backend_buffer_type_i {
    const char *          (*GGML_CALL get_name)        (ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*GGML_CALL alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
    size_t                (*GGML_CALL get_alignment)   (ggml_backend_buffer_type_t buft); // tensor alignment
    size_t                (*GGML_CALL get_max_size)    (ggml_backend_buffer_type_t buft); // allocation max size
    size_t                (*GGML_CALL get_alloc_size)  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor); // data size needed to allocate the tensor, including padding
    bool                  (*GGML_CALL supports_backend)(ggml_backend_buffer_type_t buft, ggml_backend_t backend); // check if the buffer type is usable by the backend
    // check if tensor data is in host memory
    // should be equivalent to supports_backend(buft, ggml_backend_cpu_init())
    bool                  (*GGML_CALL is_host)         (ggml_backend_buffer_type_t buft);
};

struct ggml_backend_buffer_type {
    struct ggml_backend_buffer_type_i  iface;
    ggml_backend_buffer_type_context_t context;
};

union gguf_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    struct gguf_str str;

    struct {
        enum gguf_type type;

        uint64_t n;  // GGUFv2
        void * data;
    } arr;
};

struct gguf_header {
    char magic[4];

    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct gguf_kv {
    struct gguf_str key;

    enum  gguf_type  type;
    union gguf_value value;
};

struct gguf_tensor_info {
    struct gguf_str name;

    uint32_t n_dims;
    uint64_t ne[GGML_MAX_DIMS];

    enum ggml_type type;

    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void * data;
    size_t size;
};

struct gguf_context {
    struct gguf_header header;

    struct gguf_kv * kv;
    struct gguf_tensor_info * infos;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};

struct gguf_init_params {
    bool no_alloc;

    // if not NULL, create a ggml_context and allocate the tensor data in it
    struct ggml_context ** ctx;
};

struct ggml_init_params {
    // memory pool
    size_t mem_size;   // bytes
    void * mem_buffer; // if NULL, memory will be allocated internally
    bool   no_alloc;   // don't allocate memory for the tensor data
};

struct ggml_tensor * ggml_new_tensor(
        struct ggml_context * ctx,
        enum   ggml_type      type,
        int                   n_dims,
        const int64_t       * ne) {
    return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
}

struct ggml_tensor * ggml_new_tensor_1d(
        struct ggml_context * ctx,
        enum   ggml_type      type,
        int64_t               ne0) {
    return ggml_new_tensor(ctx, type, 1, &ne0);
}

void ggml_time_init(void) {
    /*
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_start = t.QuadPart;
    */
}

static bool GGML_OP_HAS_INIT    [GGML_OP_COUNT] = { 0 };
static bool GGML_OP_HAS_FINALIZE[GGML_OP_COUNT] = { 0 };

static void ggml_setup_op_has_task_pass(void) {
    {   // INIT
        bool * p = GGML_OP_HAS_INIT;

        p[GGML_OP_ACC                    ] = true;
        p[GGML_OP_MUL_MAT                ] = true;
        p[GGML_OP_MUL_MAT_ID             ] = true;
        p[GGML_OP_OUT_PROD               ] = true;
        p[GGML_OP_SET                    ] = true;
        p[GGML_OP_GET_ROWS_BACK          ] = true;
        p[GGML_OP_DIAG_MASK_INF          ] = true;
        p[GGML_OP_DIAG_MASK_ZERO         ] = true;
        p[GGML_OP_CONV_TRANSPOSE_1D      ] = true;
        p[GGML_OP_CONV_TRANSPOSE_2D      ] = true;
        p[GGML_OP_FLASH_ATTN_BACK        ] = true;
        p[GGML_OP_CROSS_ENTROPY_LOSS     ] = true;
        p[GGML_OP_ADD_REL_POS            ] = true;
    }

    {   // FINALIZE
        bool * p = GGML_OP_HAS_FINALIZE;

        p[GGML_OP_CROSS_ENTROPY_LOSS     ] = true;
    }
}


size_t ggml_used_mem(const struct ggml_context * ctx) {
    return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

//static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
static int64_t my_atomic_fetch_add(atomic_int * ptr, int64_t inc) {
    //return InterlockedExchangeAdd(ptr, inc);
    return atomic_fetch_add(ptr, inc);
}

//static LONG atomic_fetch_sub(atomic_int * ptr, LONG dec) {
static int64_t my_atomic_fetch_sub(atomic_int * ptr, int64_t dec) {
    return my_atomic_fetch_add(ptr, -(dec));
}

static atomic_int g_state_barrier = 0;

inline static void ggml_critical_section_start(void) {
    
    int processing = my_atomic_fetch_add(&g_state_barrier, 1);
    
    while (processing > 0) {
        // wait for other threads to finish
        my_atomic_fetch_sub(&g_state_barrier, 1);
        sched_yield(); // TODO: reconsider this
        processing = my_atomic_fetch_add(&g_state_barrier, 1);
    }
}

inline static void ggml_critical_section_end(void) {
    my_atomic_fetch_sub(&g_state_barrier, 1);
}

static size_t gguf_type_size(enum gguf_type type) {
    GGML_ASSERT(0 <= type && type < GGUF_TYPE_COUNT);
    return GGUF_TYPE_SIZE[type];
}

size_t ggml_tensor_overhead(void) {
    return GGML_OBJECT_SIZE + GGML_TENSOR_SIZE;
}

static void gguf_free_kv(struct gguf_kv * kv) {
    if (kv->key.data) {
        free(kv->key.data);
    }

    if (kv->type == GGUF_TYPE_STRING) {
        if (kv->value.str.data) {
            free(kv->value.str.data);
        }
    }

    if (kv->type == GGUF_TYPE_ARRAY) {
        if (kv->value.arr.data) {
            if (kv->value.arr.type == GGUF_TYPE_STRING) {
                for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                    struct gguf_str * str = &((struct gguf_str *) kv->value.arr.data)[j];
                    if (str->data) {
                        free(str->data);
                    }
                }
            }
            free(kv->value.arr.data);
        }
    }
}

GGML_CALL const char * ggml_type_name(enum ggml_type type) {
    return type_traits[type].type_name;
}

GGML_CALL int ggml_blck_size(enum ggml_type type) {
    return type_traits[type].blck_size;
}

void ggml_free(struct ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    // make this function thread safe
    ggml_critical_section_start();

    bool found = false;

    for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
        if (&g_state.contexts[i].context == ctx) {
            g_state.contexts[i].used = false;

            GGML_PRINT_DEBUG("%s: context %d has been freed. memory used = %zu\n",
                    __func__, i, ggml_used_mem(ctx));

            if (ctx->mem_buffer_owned) {
                GGML_ALIGNED_FREE(ctx->mem_buffer);
            }

            found = true;
            break;
        }
    }

    if (!found) {
        GGML_PRINT_DEBUG("%s: context not found\n", __func__);
    }

    ggml_critical_section_end();
}

void ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

struct ggml_tensor * ggml_set_name(struct ggml_tensor * tensor, const char * name) {
    strncpy(tensor->name, name, sizeof(tensor->name) - 1);
    tensor->name[sizeof(tensor->name) - 1] = '\0';
    return tensor;
}

inline static float ggml_gelu_f32(float x) {
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

struct ggml_context * ggml_init(struct ggml_init_params params) {
    // make this function thread safe
    ggml_critical_section_start();
    
    static bool is_first_call = true;

    if (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            //const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            //const uint64_t t_end = ggml_time_us(); //UNUSED(t_end);

            //GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

        // initialize g_state
        {
            //const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            g_state = (struct ggml_state) {
                /*.contexts =*/ { { 0 } },
                /*.numa =*/ {
                    .n_nodes = 0,
                    .total_cpus = 0,
                },
            };

            for (int i = 0; i < GGML_MAX_CONTEXTS; ++i) {
                g_state.contexts[i].used = false;
            }

            //const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            //GGML_PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

#if defined(GGML_USE_CLBLAST)
        ggml_cl_init();
#endif

        ggml_setup_op_has_task_pass();

        is_first_call = false;
    }

    // find non-used context in g_state
    struct ggml_context * ctx = NULL;

    for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
        if (!g_state.contexts[i].used) {
            g_state.contexts[i].used = true;
            ctx = &g_state.contexts[i].context;

            GGML_PRINT_DEBUG("%s: found unused context %d\n", __func__, i);
            break;
        }
    }

    if (ctx == NULL) {
        GGML_PRINT_DEBUG("%s: no unused context found\n", __func__);

        ggml_critical_section_end();

        return NULL;
    }

    // allow to call ggml_init with 0 size
    if (params.mem_size == 0) {
        params.mem_size = GGML_MEM_ALIGN;
    }

    const size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

    *ctx = (struct ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : GGML_ALIGNED_MALLOC(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.no_alloc_save      =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ NULL,
        /*.objects_end        =*/ NULL,
        /*.scratch            =*/ { 0, 0, NULL, },
        /*.scratch_save       =*/ { 0, 0, NULL, },
    };

    GGML_ASSERT(ctx->mem_buffer != NULL);

    ggml_assert_aligned(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    ggml_critical_section_end();

    return ctx;
}

static void gguf_tensor_info_sanitize(struct gguf_tensor_info * info) {
    GGML_ASSERT(info->n_dims <= GGML_MAX_DIMS);
    GGML_ASSERT(0 <= info->type && info->type < GGML_TYPE_COUNT);

    for (uint32_t i = 0; i < info->n_dims; ++i) {
        GGML_ASSERT(info->ne[i] > 0);
    }

    // prevent overflow for total number of elements
    GGML_ASSERT(INT64_MAX/info->ne[1] > info->ne[0]);
    GGML_ASSERT(INT64_MAX/info->ne[2] > info->ne[0]*info->ne[1]);
    GGML_ASSERT(INT64_MAX/info->ne[3] > info->ne[0]*info->ne[1]*info->ne[2]);}

void gguf_free(struct gguf_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->kv) {
        // free string memory - not great..
        for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
            gguf_free_kv(&ctx->kv[i]);
        }

        free(ctx->kv);
    }

    if (ctx->infos) {
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            if (info->name.data) {
                free(info->name.data);
            }
        }

        free(ctx->infos);
    }

    free(ctx);
}

inline static void * ggml_calloc(size_t num, size_t size) {
    if (num == 0 || size == 0) {
        //GGML_PRINT("WARNING: Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        printf("WARNING: Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        return NULL;
    }
    void * result = calloc(num, size);
    if (result == NULL) {
        //GGML_PRINT("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        printf("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        GGML_ASSERT(false);
    }
    return result;
}

int gguf_get_n_kv(const struct gguf_context * ctx) {
    return ctx->header.n_kv;
}

uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT32);
    return ctx->kv[key_id].value.uint32;
}

const char * gguf_get_key(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].key.data;
}

int gguf_find_key(const struct gguf_context * ctx, const char * key) {
    // return -1 if key not found
    int keyfound = -1;

    const int n_kv = gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
        if (strcmp(key, gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

static bool gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

static bool gguf_fread_str(FILE * file, struct gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && gguf_fread_el(file, &p->n, sizeof(p->n), offset);

    // early exit if string length is invalid, prevents from integer overflow
    if (p->n == SIZE_MAX) {
        fprintf(stderr, "%s: invalid string length (%" PRIu64 ")\n", __func__, p->n);
        return false;
    }

    p->data = (char*) ggml_calloc(p->n + 1, 1);


    ok = ok && gguf_fread_el(file,  p->data, p->n, offset);

    return ok;
}

const char * ggml_op_to_string(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:                 return "NONE";
        case GGML_OP_DUP:                  return "DUP";
        case GGML_OP_ADD:                  return "ADD";
        case GGML_OP_ADD1:                 return "ADD1";
        case GGML_OP_ACC:                  return "ACC";
        case GGML_OP_SUB:                  return "SUB";
        case GGML_OP_MUL:                  return "MUL";
        case GGML_OP_DIV:                  return "DIV";
        case GGML_OP_SQR:                  return "SQR";
        case GGML_OP_SQRT:                 return "SQRT";
        case GGML_OP_LOG:                  return "LOG";
        case GGML_OP_SUM:                  return "SUM";
        case GGML_OP_SUM_ROWS:             return "SUM_ROWS";
        case GGML_OP_MEAN:                 return "MEAN";
        case GGML_OP_ARGMAX:               return "ARGMAX";
        case GGML_OP_REPEAT:               return "REPEAT";
        case GGML_OP_REPEAT_BACK:          return "REPEAT_BACK";
        case GGML_OP_CONCAT:               return "CONCAT";
        case GGML_OP_SILU_BACK:            return "SILU_BACK";
        case GGML_OP_NORM:                 return "NORM";
        case GGML_OP_RMS_NORM:             return "RMS_NORM";
        case GGML_OP_RMS_NORM_BACK:        return "RMS_NORM_BACK";
        case GGML_OP_GROUP_NORM:           return "GROUP_NORM";
        case GGML_OP_MUL_MAT:              return "MUL_MAT";
        case GGML_OP_MUL_MAT_ID:           return "MUL_MAT_ID";
        case GGML_OP_OUT_PROD:             return "OUT_PROD";
        case GGML_OP_SCALE:                return "SCALE";
        case GGML_OP_SET:                  return "SET";
        case GGML_OP_CPY:                  return "CPY";
        case GGML_OP_CONT:                 return "CONT";
        case GGML_OP_RESHAPE:              return "RESHAPE";
        case GGML_OP_VIEW:                 return "VIEW";
        case GGML_OP_PERMUTE:              return "PERMUTE";
        case GGML_OP_TRANSPOSE:            return "TRANSPOSE";
        case GGML_OP_GET_ROWS:             return "GET_ROWS";
        case GGML_OP_GET_ROWS_BACK:        return "GET_ROWS_BACK";
        case GGML_OP_DIAG:                 return "DIAG";
        case GGML_OP_DIAG_MASK_INF:        return "DIAG_MASK_INF";
        case GGML_OP_DIAG_MASK_ZERO:       return "DIAG_MASK_ZERO";
        case GGML_OP_SOFT_MAX:             return "SOFT_MAX";
        case GGML_OP_SOFT_MAX_BACK:        return "SOFT_MAX_BACK";
        case GGML_OP_ROPE:                 return "ROPE";
        case GGML_OP_ROPE_BACK:            return "ROPE_BACK";
        case GGML_OP_CLAMP:                return "CLAMP";
        case GGML_OP_CONV_TRANSPOSE_1D:    return "CONV_TRANSPOSE_1D";
        case GGML_OP_IM2COL:               return "IM2COL";
        case GGML_OP_CONV_TRANSPOSE_2D:    return "CONV_TRANSPOSE_2D";
        case GGML_OP_POOL_1D:              return "POOL_1D";
        case GGML_OP_POOL_2D:              return "POOL_2D";
        case GGML_OP_UPSCALE:              return "UPSCALE";
        case GGML_OP_PAD:                  return "PAD";
        case GGML_OP_ARANGE:               return "ARANGE";
        case GGML_OP_TIMESTEP_EMBEDDING:   return "TIMESTEP_EMBEDDING";
        case GGML_OP_ARGSORT:              return "ARGSORT";
        case GGML_OP_LEAKY_RELU:           return "LEAKY_RELU";
        case GGML_OP_FLASH_ATTN_EXT:       return "FLASH_ATTN_EXT";
        case GGML_OP_FLASH_ATTN_BACK:      return "FLASH_ATTN_BACK";
        case GGML_OP_SSM_CONV:             return "SSM_CONV";
        case GGML_OP_SSM_SCAN:             return "SSM_SCAN";
        case GGML_OP_WIN_PART:             return "WIN_PART";
        case GGML_OP_WIN_UNPART:           return "WIN_UNPART";
        case GGML_OP_GET_REL_POS:          return "GET_REL_POS";
        case GGML_OP_ADD_REL_POS:          return "ADD_REL_POS";
        case GGML_OP_UNARY:                return "UNARY";
        case GGML_OP_MAP_UNARY:            return "MAP_UNARY";
        case GGML_OP_MAP_BINARY:           return "MAP_BINARY";
        case GGML_OP_MAP_CUSTOM1_F32:      return "MAP_CUSTOM1_F32";
        case GGML_OP_MAP_CUSTOM2_F32:      return "MAP_CUSTOM2_F32";
        case GGML_OP_MAP_CUSTOM3_F32:      return "MAP_CUSTOM3_F32";
        case GGML_OP_MAP_CUSTOM1:          return "MAP_CUSTOM1";
        case GGML_OP_MAP_CUSTOM2:          return "MAP_CUSTOM2";
        case GGML_OP_MAP_CUSTOM3:          return "MAP_CUSTOM3";
        case GGML_OP_CROSS_ENTROPY_LOSS:   return "CROSS_ENTROPY_LOSS";
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK: return "CROSS_ENTROPY_LOSS_BACK";
        case GGML_OP_COUNT:                return "COUNT";
        default:                           return "UNKNOWN";
    }
}

void print_operation(struct ggml_tensor * tensor) {
    if (tensor == NULL) {
        printf("Tensor is NULL.\n");
        return;
    }

    printf("\n *************************  OPERATION  ************************* \n");

    //printf("Tensor view_src: %s\n", tensor->view_src->op);

    printf("Tensor Name: %s\n", tensor->name);
    printf("Tensor Type: %d\n", tensor->type); // Tipo de tensor (e.g., GGML_TYPE_I8, GGML_TYPE_FLOAT32, etc.)
    printf("Operation: %d\n", tensor->op); // Tipo de operación realizada en el tensor
    printf("Operation Parameters: ");
    
    for (long unsigned int i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++i) {
    //for (int i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++i) {
        if (tensor->op_params[i] != 0) {  // Imprime solo los parámetros no nulos
            printf("%d ", tensor->op_params[i]);
        }
    }
    printf("\n");
    
    printf("Flags: %d\n", tensor->flags);
    printf("Number of Elements: ");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] != 0) {
            printf("%ld ", tensor->ne[i]);
        }
    }
    printf("\n");

    printf("Performance: Runs: %d, Cycles: %ld, Time: %ld us\n",
           tensor->perf_runs, tensor->perf_cycles, tensor->perf_time_us);

    if (tensor->grad != NULL) {
        printf("Gradient Tensor: %s\n", tensor->grad->name);
    }

    printf("Data Pointer: %p\n", tensor->data);
    printf("Extra Pointer: %p\n", tensor->extra);
    printf(" ------------------------------------------------------------ \n");
}

struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params) {
    FILE * file = fopen(fname, "rb");
    if (!file) {
        return NULL;
    }

    size_t offset = 0;
    char magic[4];
    
    // Leer el número mágico (4 bytes)
    {
        gguf_fread_el(file, magic, sizeof(magic), &offset);

        // Validar si el número mágico coincide con GGUF_MAGIC
        for (uint32_t i = 0; i < sizeof(magic); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                fprintf(stderr, "%s: invalid magic characters '%c%c%c%c'\n", __func__, magic[0], magic[1], magic[2], magic[3]);
                fclose(file);
                return NULL;
            }
        }
    }

    bool ok = true;

    // Crear el contexto de GGUF y asignar memoria
    //struct gguf_context * ctx = (gguf_context*) GGML_CALLOC(1, sizeof(struct gguf_context));
    //struct gguf_context * ctx = (struct gguf_context *) GGML_CALLOC(1, sizeof(struct gguf_context));
    struct gguf_context * ctx = (struct gguf_context *) calloc(1, sizeof(struct gguf_context));

   // read the header
    {
        strncpy(ctx->header.magic, magic, 4);

        ctx->kv    = NULL;
        ctx->infos = NULL;
        ctx->data  = NULL;

        ok = ok && gguf_fread_el(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);

        if (ctx->header.version == 1) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }

        // sanity-checks to prevent from integer/buffer overflows

        ok = ok && (ctx->header.n_tensors < (SIZE_MAX/2)/sizeof(struct gguf_tensor_info));
        //ok = ok && (ctx->header.n_tensors < (SIZE_MAX/2)/ggml_tensor_overhead());
        ok = ok && (ctx->header.n_tensors < (SIZE_MAX/2) / (GGML_OBJECT_SIZE + GGML_TENSOR_SIZE));
        ok = ok && (ctx->header.n_kv      < (SIZE_MAX/2)/sizeof(struct gguf_kv));

        if (!ok) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }
    }

    // read the kv pairs
    {
        const uint64_t n_kv = ctx->header.n_kv;

        // header.n_kv will hold the actual value of pairs that were successfully read in the loop below
        ctx->header.n_kv = 0;
        //ctx->kv = (struct gguf_kv *) ggml_calloc(n_kv, sizeof(struct gguf_kv));
        ctx->kv = (struct gguf_kv *) calloc(n_kv, sizeof(struct gguf_kv));

        for (uint64_t i = 0; i < n_kv; ++i) {
            struct gguf_kv * kv = &ctx->kv[i];

            //fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            ok = ok && gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && gguf_fread_el (file, &kv->type, sizeof(kv->type), &offset);

            //fprintf(stderr, "%s: reading kv with key %s\n", __func__, kv->key.data);

            switch (kv->type) {
                case GGUF_TYPE_UINT8:   ok = ok && gguf_fread_el (file, &kv->value.uint8,   sizeof(kv->value.uint8),   &offset); break;
                case GGUF_TYPE_INT8:    ok = ok && gguf_fread_el (file, &kv->value.int8,    sizeof(kv->value.int8),    &offset); break;
                case GGUF_TYPE_UINT16:  ok = ok && gguf_fread_el (file, &kv->value.uint16,  sizeof(kv->value.uint16),  &offset); break;
                case GGUF_TYPE_INT16:   ok = ok && gguf_fread_el (file, &kv->value.int16,   sizeof(kv->value.int16),   &offset); break;
                case GGUF_TYPE_UINT32:  ok = ok && gguf_fread_el (file, &kv->value.uint32,  sizeof(kv->value.uint32),  &offset); break;
                case GGUF_TYPE_INT32:   ok = ok && gguf_fread_el (file, &kv->value.int32,   sizeof(kv->value.int32),   &offset); break;
                case GGUF_TYPE_FLOAT32: ok = ok && gguf_fread_el (file, &kv->value.float32, sizeof(kv->value.float32), &offset); break;
                case GGUF_TYPE_UINT64:  ok = ok && gguf_fread_el (file, &kv->value.uint64,  sizeof(kv->value.uint64),  &offset); break;
                case GGUF_TYPE_INT64:   ok = ok && gguf_fread_el (file, &kv->value.int64,   sizeof(kv->value.int64),   &offset); break;
                case GGUF_TYPE_FLOAT64: ok = ok && gguf_fread_el (file, &kv->value.float64, sizeof(kv->value.float64), &offset); break;
                case GGUF_TYPE_BOOL:    ok = ok && gguf_fread_el (file, &kv->value.bool_,   sizeof(kv->value.bool_),   &offset); break;
                case GGUF_TYPE_STRING:  ok = ok && gguf_fread_str(file, &kv->value.str,                                &offset); break;
                case GGUF_TYPE_ARRAY:
                    {
                        ok = ok && gguf_fread_el(file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
                        ok = ok && gguf_fread_el(file, &kv->value.arr.n,    sizeof(kv->value.arr.n),    &offset);

                        switch (kv->value.arr.type) {
                            case GGUF_TYPE_UINT8:
                            case GGUF_TYPE_INT8:
                            case GGUF_TYPE_UINT16:
                            case GGUF_TYPE_INT16:
                            case GGUF_TYPE_UINT32:
                            case GGUF_TYPE_INT32:
                            case GGUF_TYPE_FLOAT32:
                            case GGUF_TYPE_UINT64:
                            case GGUF_TYPE_INT64:
                            case GGUF_TYPE_FLOAT64:
                            case GGUF_TYPE_BOOL:
                                {
                                    // prevent from integer overflow in the malloc below
                                    if (kv->value.arr.n >= SIZE_MAX/gguf_type_size(kv->value.arr.type)) {
                                        fprintf(stderr, "%s: array size is too large (%" PRIu64 ")\n", __func__, kv->value.arr.n);
                                        fclose(file);
                                        gguf_free(ctx);
                                        return NULL;
                                    }

                                    kv->value.arr.data = ggml_calloc(kv->value.arr.n, gguf_type_size(kv->value.arr.type));

                                    ok = ok && gguf_fread_el(file, kv->value.arr.data, kv->value.arr.n * gguf_type_size(kv->value.arr.type), &offset);
                                } break;
                            case GGUF_TYPE_STRING:
                                {
                                    // prevent from integer overflow in the malloc below
                                    if (kv->value.arr.n >= SIZE_MAX/sizeof(struct gguf_str)) {
                                        fprintf(stderr, "%s: array size is too large (%" PRIu64 ")\n", __func__, kv->value.arr.n);
                                        fclose(file);
                                        gguf_free(ctx);
                                        return NULL;
                                    }

                                    kv->value.arr.data = ggml_calloc(kv->value.arr.n, sizeof(struct gguf_str));

                                    for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                                        ok = ok && gguf_fread_str(file, &((struct gguf_str *) kv->value.arr.data)[j], &offset);
                                    }
                                } break;
                            case GGUF_TYPE_ARRAY:
                            default: GGML_ASSERT(false && "invalid type"); break;
                        }
                    } break;
                default: GGML_ASSERT(false && "invalid type");
            }

            if (!ok) {
                break;
            }

            ctx->header.n_kv++;
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }
    }
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // read the tensor infos
    if (ctx->header.n_tensors > 0) {
        ctx->infos = ggml_calloc(ctx->header.n_tensors, sizeof(struct gguf_tensor_info));

        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                info->ne[j] = 1;
            }

            ok = ok && gguf_fread_str(file, &info->name,                          &offset);
            ok = ok && gguf_fread_el (file, &info->n_dims, sizeof(info->n_dims),  &offset);

            ok = ok && (info->n_dims <= GGML_MAX_DIMS);

            for (uint32_t j = 0; j < info->n_dims; ++j) {
                ok = ok && gguf_fread_el(file, &info->ne[j], sizeof(info->ne[j]), &offset);
            }

            ok = ok && gguf_fread_el (file, &info->type,   sizeof(info->type),    &offset);
            ok = ok && gguf_fread_el (file, &info->offset, sizeof(info->offset),  &offset);

            // TODO: return an error instead of crashing with GGML_ASSERT
            gguf_tensor_info_sanitize(info);

            // make sure there is no duplicated tensor names
            for (uint64_t j = 0; j < i; ++j) {
                if (strcmp(info->name.data, ctx->infos[j].name.data) == 0) {
                    fprintf(stderr, "%s: duplicated tensor name %s\n", __func__, info->name.data);
                    ok = false;
                }
            }

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor info\n", __func__);
                fclose(file);
                gguf_free(ctx);
                return NULL;
            }
        }
    }
    
    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    int alignment_idx = gguf_find_key(ctx, "general.alignment");
    if (alignment_idx != -1) {
        ctx->alignment = gguf_get_val_u32(ctx, alignment_idx);
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset_pad = offset % ctx->alignment;

        if (offset_pad != 0) {
            offset += ctx->alignment - offset_pad;
            fseek(file, offset, SEEK_SET);
        }
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = offset;

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];
            
             const int64_t ne =
                (int64_t) info->ne[0] *
                (int64_t) info->ne[1] *
                (int64_t) info->ne[2] *
                (int64_t) info->ne[3];

            if (ne % ggml_blck_size(info->type) != 0) {
                fprintf(stderr, "%s: tensor '%s' of type %d (%s) number of elements (%" PRId64 ") is not a multiple of block size (%d)\n",
                        __func__, info->name.data, (int)info->type, ggml_type_name(info->type), ne, ggml_blck_size(info->type));
                fclose(file);
                gguf_free(ctx);
                return NULL;
            }

            const size_t size_cur = ggml_row_size(info->type, ne);

            ctx->size += GGML_PAD(size_cur, ctx->alignment);
        }
    }
    
    // load the tensor data only if requested
    if (params.ctx != NULL) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        // the ggml_tensor structs to the appropriate locations in the binary blob
        
        // compute the exact size needed for the new ggml_context
        const size_t mem_size =
            params.no_alloc ?
            (ctx->header.n_tensors    )*ggml_tensor_overhead() :
            (ctx->header.n_tensors + 1)*ggml_tensor_overhead() + ctx->size;

        struct ggml_init_params pdata = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc   = params.no_alloc,
        };
    
        *params.ctx = ggml_init(pdata);
        
        struct ggml_context * ctx_data = *params.ctx;
        
        struct ggml_tensor * data = NULL;
        
        if (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != NULL;

            // read the binary blob with the tensor data
            ok = ok && gguf_fread_el(file, data->data, ctx->size, &offset);

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor data\n", __func__);
                fclose(file);
                ggml_free(ctx_data);
                gguf_free(ctx);
                return NULL;
            }

            ctx->data = data->data;
        }

        //print_operation(data);

        ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            const int64_t ne[GGML_MAX_DIMS] = {
                ctx->infos[i].ne[0],
                ctx->infos[i].ne[1],
                ctx->infos[i].ne[2],
                ctx->infos[i].ne[3],
            };

            struct ggml_tensor * cur = ggml_new_tensor(ctx_data, ctx->infos[i].type, ctx->infos[i].n_dims, ne);

            ok = ok && cur != NULL;

            if (!ok) {
                break;
            }

            ggml_set_name(cur, ctx->infos[i].name.data);

            // point the data member to the appropriate location in the binary blob using the tensor infos
            if (!params.no_alloc) {
              //cur->data = (char *) data->data + ctx->infos[i].offset - ctx->offset; // offset from start of file
                cur->data = (char *) data->data + ctx->infos[i].offset;               // offset from data
            }
            // Imprimir la operación del tensor actual
            //print_operation(cur);
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read the tensor data\n", __func__);
            fclose(file);
            ggml_free(ctx_data);
            gguf_free(ctx);
            return NULL;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }
    
    fclose(file);
    return ctx;
}

void gguf_print_context(const struct gguf_context * ctx) {
    FILE *file = fopen("../gguf_context.txt", "w"); 
    if (file == NULL) {
        perror("Error al abrir el archivo");
        return;
    }

    fprintf(file, "\n *************************  HEADER  ************************* \n \n");
    fprintf(file, "Magic: %.4s\n", ctx->header.magic);
    fprintf(file, "Version: %u\n", ctx->header.version);
    fprintf(file, "Number of tensors: %lu\n", ctx->header.n_tensors);
    fprintf(file, "Number of key-value pairs: %lu\n", ctx->header.n_kv);
    
    fprintf(file, "\n *************************  METADATA  ************************* \n \n");

    for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
        fprintf(file, "Key: %.*s\n", (int)ctx->kv[i].key.n, ctx->kv[i].key.data);
        
        // Escribir el valor dependiendo del tipo
        switch (ctx->kv[i].type) {
            case GGUF_TYPE_UINT8:
                fprintf(file, "Value: %u (uint8)\n", ctx->kv[i].value.uint8);
                break;
            case GGUF_TYPE_INT8:
                fprintf(file, "Value: %d (int8)\n", ctx->kv[i].value.int8);
                break;
            case GGUF_TYPE_UINT16:
                fprintf(file, "Value: %u (uint16)\n", ctx->kv[i].value.uint16);
                break;
            case GGUF_TYPE_INT16:
                fprintf(file, "Value: %d (int16)\n", ctx->kv[i].value.int16);
                break;
            case GGUF_TYPE_UINT32:
                fprintf(file, "Value: %u (uint32)\n", ctx->kv[i].value.uint32);
                break;
            case GGUF_TYPE_INT32:
                fprintf(file, "Value: %d (int32)\n", ctx->kv[i].value.int32);
                break;
            case GGUF_TYPE_UINT64:
                fprintf(file, "Value: %lu (uint64)\n", ctx->kv[i].value.uint64);
                break;
            case GGUF_TYPE_INT64:
                fprintf(file, "Value: %ld (int64)\n", ctx->kv[i].value.int64);
                break;
            case GGUF_TYPE_FLOAT32:
                fprintf(file, "Value: %f (float32)\n", ctx->kv[i].value.float32);
                break;
            case GGUF_TYPE_FLOAT64:
                fprintf(file, "Value: %lf (float64)\n", ctx->kv[i].value.float64);
                break;
            case GGUF_TYPE_BOOL:
                fprintf(file, "Value: %s (bool)\n", ctx->kv[i].value.bool_ ? "true" : "false");
                break;
            case GGUF_TYPE_STRING:
                fprintf(file, "Value: %.*s (string)\n", (int)ctx->kv[i].value.str.n, ctx->kv[i].value.str.data);
                break;
            case GGUF_TYPE_ARRAY:
                fprintf(file, "Value: (array of type %d, size %lu)\n", ctx->kv[i].value.arr.type, ctx->kv[i].value.arr.n);
                // Añadir más detalles aquí si es necesario para los contenidos del array
                break;
            default:
                fprintf(file, "Value: Unknown type\n");
                break;
        }
        
        fprintf(file, "Type: %d\n", ctx->kv[i].type);
        fprintf(file, " ----------------------------------- \n");
    }

    fprintf(file, "\n *************************  TENSORS  ************************* \n \n");
    for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
        fprintf(file, "Tensor Name: %.*s\n", (int)ctx->infos[i].name.n, ctx->infos[i].name.data);
        fprintf(file, "Number of Dimensions: %u\n", ctx->infos[i].n_dims);
        fprintf(file, "Offsets: %lu\n", ctx->infos[i].offset);
        fprintf(file, "Data Size: %zu\n", ctx->infos[i].size);
        
        // Escribir las dimensiones
        fprintf(file, "Dimensions: ");
        for (uint32_t j = 0; j < ctx->infos[i].n_dims; ++j) {
            fprintf(file, "%lu ", ctx->infos[i].ne[j]);
        }
        fprintf(file, "\n");
        // Crear un tensor temporal y asignar valores necesarios
        struct ggml_tensor temp_tensor;
        temp_tensor.type = ctx->infos[i].type;
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            temp_tensor.ne[j] = ctx->infos[i].ne[j];
        }
        temp_tensor.data = (void *)((char *)ctx->data + ctx->infos[i].offset);
        
        //temp_tensor.op = ctx->infos[i].op; 

        
        

        // Escribir la operación asociada con el tensor
        fprintf(file, "Operation: ");
        
        // Dependiendo de la operación, imprimimos la descripción
        switch (temp_tensor.op) {
            case GGML_OP_NONE:
                fprintf(file, "None operation\n");
                break;
            case GGML_OP_ADD:
                fprintf(file, "Addition operation (element-wise addition)\n");
                break;
            case GGML_OP_SUB:
                fprintf(file, "Subtraction operation (element-wise subtraction)\n");
                break;
            case GGML_OP_MUL:
                fprintf(file, "Multiplication operation (element-wise multiplication)\n");
                break;
            case GGML_OP_DIV:
                fprintf(file, "Division operation (element-wise division)\n");
                break;
            case GGML_OP_SQR:
                fprintf(file, "Square operation (element-wise squaring)\n");
                break;
            case GGML_OP_SQRT:
                fprintf(file, "Square root operation (element-wise square root)\n");
                break;
            case GGML_OP_LOG:
                fprintf(file, "Logarithm operation (element-wise log)\n");
                break;
            case GGML_OP_SUM:
                fprintf(file, "Sum operation (sum of all elements)\n");
                break;
            case GGML_OP_SUM_ROWS:
                fprintf(file, "Sum operation (sum over rows)\n");
                break;
            case GGML_OP_MEAN:
                fprintf(file, "Mean operation (mean of all elements)\n");
                break;
            case GGML_OP_ARGMAX:
                fprintf(file, "Argmax operation (index of the maximum element)\n");
                break;
            case GGML_OP_REPEAT:
                fprintf(file, "Repeat operation (repeat tensor along specified axis)\n");
                break;
            case GGML_OP_CONCAT:
                fprintf(file, "Concatenation operation (concatenate tensors along an axis)\n");
                break;
            case GGML_OP_SCALE:
                fprintf(file, "Scale operation (scale tensor by a factor)\n");
                break;
            case GGML_OP_SET:
                fprintf(file, "Set operation (set tensor to a constant value)\n");
                break;
            case GGML_OP_CPY:
                fprintf(file, "Copy operation (copy tensor)\n");
                break;
            case GGML_OP_RESHAPE:
                fprintf(file, "Reshape operation (change tensor shape)\n");
                break;
            case GGML_OP_VIEW:
                fprintf(file, "View operation (create a view of the tensor)\n");
                break;
            case GGML_OP_PERMUTE:
                fprintf(file, "Permute operation (rearrange tensor dimensions)\n");
                break;
            case GGML_OP_TRANSPOSE:
                fprintf(file, "Transpose operation (transpose tensor)\n");
                break;
            case GGML_OP_SOFT_MAX:
                fprintf(file, "Softmax operation (apply softmax function to tensor)\n");
                break;
            case GGML_OP_CLAMP:
                fprintf(file, "Clamp operation (limit tensor values within a range)\n");
                break;
            case GGML_OP_POOL_2D:
                fprintf(file, "2D Pooling operation (e.g., max or average pooling)\n");
                break;
            case GGML_OP_UPSCALE:
                fprintf(file, "Upscale operation (nearest-neighbor interpolation)\n");
                break;
            case GGML_OP_ARGSORT:
                fprintf(file, "Argsort operation (sort tensor indices)\n");
                break;
            case GGML_OP_LEAKY_RELU:
                fprintf(file, "Leaky ReLU operation (apply Leaky ReLU activation)\n");
                break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
                fprintf(file, "Cross-entropy loss operation (calculate cross-entropy loss)\n");
                break;
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
                fprintf(file, "Cross-entropy loss backpropagation (calculate gradients)\n");
                break;
            default:
                fprintf(file, "Unknown operation (op value: %d)\n", temp_tensor.op);
                break;
        }

        // Imprimir detalles del tensor
        
        fprintf(file, "Tensor type: ");
        switch (temp_tensor.type) {
            case GGML_TYPE_F32:
                fprintf(file, "F32\n");
                break;
            case GGML_TYPE_F16:
                fprintf(file, "F16\n");
                break;
            case GGML_TYPE_Q4_0:
                fprintf(file, "Q4_0\n");
                break;
            case GGML_TYPE_Q4_1:
                fprintf(file, "Q4_1\n");
                break;
            case GGML_TYPE_Q5_0:
                fprintf(file, "Q5_0\n");
                break;
            case GGML_TYPE_Q5_1:
                fprintf(file, "Q5_1\n");
                break;
            case GGML_TYPE_Q8_0:
                fprintf(file, "Q8_0\n");
                break;
            case GGML_TYPE_Q8_1:
                fprintf(file, "Q8_1\n");
                break;
            case GGML_TYPE_Q2_K:
                fprintf(file, "Q2_K\n");
                break;
            case GGML_TYPE_Q3_K:
                fprintf(file, "Q3_K\n");
                break;
            case GGML_TYPE_Q4_K:
                fprintf(file, "Q4_K\n");
                break;
            case GGML_TYPE_Q5_K:
                fprintf(file, "Q5_K\n");
                break;
            case GGML_TYPE_Q6_K:
                fprintf(file, "Q6_K\n");
                break;
            case GGML_TYPE_Q8_K:
                fprintf(file, "Q8_K\n");
                break;
            case GGML_TYPE_IQ2_XXS:
                fprintf(file, "IQ2_XXS\n");
                break;
            case GGML_TYPE_IQ2_XS:
                fprintf(file, "IQ2_XS\n");
                break;
            case GGML_TYPE_IQ3_XXS:
                fprintf(file, "IQ3_XXS\n");
                break;
            case GGML_TYPE_IQ1_S:
                fprintf(file, "IQ1_S\n");
                break;
            case GGML_TYPE_IQ4_NL:
                fprintf(file, "IQ4_NL\n");
                break;
            case GGML_TYPE_IQ3_S:
                fprintf(file, "IQ3_S\n");
                break;
            case GGML_TYPE_IQ2_S:
                fprintf(file, "IQ2_S\n");
                break;
            case GGML_TYPE_IQ4_XS:
                fprintf(file, "IQ4_XS\n");
                break;
            case GGML_TYPE_I8:
                fprintf(file, "I8\n");
                break;
            case GGML_TYPE_I16:
                fprintf(file, "I16\n");
                break;
            case GGML_TYPE_I32:
                fprintf(file, "I32\n");
                break;
            case GGML_TYPE_I64:
                fprintf(file, "I64\n");
                break;
            case GGML_TYPE_F64:
                fprintf(file, "F64\n");
                break;
            case GGML_TYPE_IQ1_M:
                fprintf(file, "IQ1_M\n");
                break;
            case GGML_TYPE_BF16:
                fprintf(file, "BF16\n");
                break;
            default:
                fprintf(file, "Unknown type (%d)\n", temp_tensor.type);
                break;
        }


        fprintf(file, "Dimensions: ");
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            fprintf(file, "%lld ", (long long)temp_tensor.ne[i]);
        }
        fprintf(file, "\n");

        fprintf(file, "Strides (bytes): ");
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            fprintf(file, "%zu ", temp_tensor.nb[i]);
        }
        fprintf(file, "\n");

        fprintf(file, "Operation parameters: ");
        for (long unsigned int i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++i) {
        //for (int i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++i) {
            fprintf(file, "%d ", temp_tensor.op_params[i]);
        }
        fprintf(file, "\n");

        fprintf(file, "Flags: %d\n", temp_tensor.flags);
        fprintf(file, "Grad tensor: %p\n", (void *)temp_tensor.grad); // Dirección de la gradiente
        fprintf(file, "Source tensors: ");
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            fprintf(file, "%p ", (void *)temp_tensor.src[i]);
        }
        fprintf(file, "\n");

        fprintf(file, "Performance runs: %d\n", temp_tensor.perf_runs);
        fprintf(file, "Performance cycles: %lld\n", (long long)temp_tensor.perf_cycles);
        fprintf(file, "Performance time (us): %lld\n", (long long)temp_tensor.perf_time_us);

        fprintf(file, "View source: %p\n", (void *)temp_tensor.view_src);
        fprintf(file, "View offset: %zu\n", temp_tensor.view_offs);

        fprintf(file, "Data pointer: %p\n", (void *)temp_tensor.data);
        fprintf(file, "Tensor name: %s\n", temp_tensor.name);
        fprintf(file, "Extra data pointer: %p\n", temp_tensor.extra);

        // Asegúrate de manejar la estructura de padding si es necesario
        fprintf(file, "Padding: ");
        for (long unsigned int i = 0; i < sizeof(temp_tensor.padding); ++i) {
        //for (int i = 0; i < sizeof(temp_tensor.padding); ++i) {
            fprintf(file, "%02x ", (unsigned char)temp_tensor.padding[i]);
        }
        fprintf(file, "\n");

        fprintf(file, "\n ---------------------------------------------------------------------- \n");
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <nombre_archivo.gguf>\n", argv[0]);
        return 1;
    }

    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    struct gguf_context * meta = gguf_init_from_file(argv[1], params);
    if (!meta) {
        fprintf(stderr, "Error: no se pudo cargar el modelo desde %s\n", argv[1]);
        return 1;
    }

    // Imprimir el contexto
    gguf_print_context(meta);

    // Liberar la memoria
    free(meta->kv);
    free(meta->infos);
    free(meta);
    
    return 0;
}