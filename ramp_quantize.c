/**
 * RAMP-Local GGUF Quantizer v3 (No-Split)
 *
 * BF16/F32 GGUF → mixed-precision GGUF via ggml_quantize_chunk().
 * Keeps fused ffn_gate_up_exps tensors intact (ik_llama natively supports them).
 * Adds .weight suffix to match llama.cpp expected tensor names.
 * Streams tensor-by-tensor: peak RAM = 1 tensor F32 + quant buffer.
 *
 * Build:
 *   cd ~/ik_llama.cpp/build_sm120
 *   gcc -O3 -march=native -fopenmp -I../ggml/include \
 *       ./ramp_quantize.c \
 *       -Lggml/src -lggml -lm -lpthread -Wl,-rpath,$(pwd)/ggml/src \
 *       -o bin/ramp-quantize
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "ggml.h"

// -----------------------------------------------------------------------
// Quant type parsing
// -----------------------------------------------------------------------

static enum ggml_type parse_type(const char *name) {
    if (!name) return GGML_TYPE_COUNT;
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const char *tn = ggml_type_name((enum ggml_type)i);
        if (tn && strcasecmp(name, tn) == 0) return (enum ggml_type)i;
    }
    if (strcasecmp(name, "Q4_K_M") == 0 || strcasecmp(name, "Q4_K") == 0) return GGML_TYPE_Q4_K;
    if (strcasecmp(name, "Q5_K_M") == 0 || strcasecmp(name, "Q5_K") == 0) return GGML_TYPE_Q5_K;
    if (strcasecmp(name, "IQ3_S") == 0) return GGML_TYPE_IQ3_S;
    if (strcasecmp(name, "IQ3_XXS") == 0) return GGML_TYPE_IQ3_XXS;
    if (strcasecmp(name, "IQ4_XS") == 0) return GGML_TYPE_IQ4_XS;
    return GGML_TYPE_COUNT;
}

// -----------------------------------------------------------------------
// Config parser
// -----------------------------------------------------------------------

#define MAX_RULES 1024

struct quant_rule { char pattern[256]; enum ggml_type type; };

static int n_rules = 0;
static struct quant_rule rules[MAX_RULES];
static enum ggml_type base_type = GGML_TYPE_Q4_K;

static int load_config(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open config: %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    buf[sz] = 0; fclose(f);

    char *bt = strstr(buf, "\"base_type\"");
    if (bt) {
        bt = strchr(bt + 11, '"');
        if (bt) { bt++; char *end = strchr(bt, '"');
            if (end) { char tmp[64]; int len = (end-bt<63)?(int)(end-bt):63;
                strncpy(tmp, bt, len); tmp[len]=0; base_type = parse_type(tmp);
                printf("Base type: %s (%d)\n", tmp, base_type); } }
    }

    char *cfg = strstr(buf, "\"config\"");
    if (!cfg) { free(buf); return -1; }
    char *p = cfg;
    while (p && n_rules < MAX_RULES) {
        char *nl = strstr(p+1, "\"layer."); char *ng = strstr(p+1, "\"global.");
        if (!nl && !ng) break;
        p = (!nl)?ng:(!ng)?nl:(nl<ng)?nl:ng;
        char *ks=p+1, *ke=strchr(ks,'"'); if(!ke) break;
        char *co=strchr(ke+1,':'); if(!co) break;
        char *vs=strchr(co,'"'); if(!vs) break; vs++;
        char *ve=strchr(vs,'"'); if(!ve) break;
        int kl=(ke-ks<255)?(int)(ke-ks):255, vl=(ve-vs<63)?(int)(ve-vs):63;
        strncpy(rules[n_rules].pattern, ks, kl); rules[n_rules].pattern[kl]=0;
        char ts[64]; strncpy(ts, vs, vl); ts[vl]=0;
        rules[n_rules].type = parse_type(ts);
        if (rules[n_rules].type < GGML_TYPE_COUNT) n_rules++;
        p = ve+1;
    }
    free(buf);
    printf("Loaded %d quantization rules\n", n_rules);
    return 0;
}

// -----------------------------------------------------------------------
// Tensor name → target type
// -----------------------------------------------------------------------

static enum ggml_type match_tensor(const char *name) {
    for (int i = 0; i < n_rules; i++) {
        const char *pat = rules[i].pattern;
        if (strncmp(pat, "layer.", 6) == 0) {
            int li; char role[64];
            if (sscanf(pat, "layer.%d.%63s", &li, role) != 2) continue;
            char pfx[32]; snprintf(pfx, sizeof(pfx), "blk.%d.", li);
            if (!strstr(name, pfx)) continue;
            if (!strcmp(role,"experts") && (strstr(name,"ffn_gate_exps")||strstr(name,"ffn_up_exps")||
                strstr(name,"ffn_down_exps")||strstr(name,"ffn_gate_up_exps"))) return rules[i].type;
            if (!strcmp(role,"attn") && (strstr(name,"attn_q.")||strstr(name,"attn_k.")||
                strstr(name,"attn_v.")||strstr(name,"attn_output.")||
                // Also match names without .weight suffix (BF16 GGUF format)
                // Use end-of-name check to avoid matching attn_qkv, attn_q_norm etc.
                (strlen(name) > 6 && strcmp(name + strlen(name) - 6, "attn_q") == 0) ||
                (strlen(name) > 6 && strcmp(name + strlen(name) - 6, "attn_k") == 0) ||
                (strlen(name) > 6 && strcmp(name + strlen(name) - 6, "attn_v") == 0) ||
                (strlen(name) > 11 && strcmp(name + strlen(name) - 11, "attn_output") == 0)
                )) return rules[i].type;
            if (!strcmp(role,"gdn") && (strstr(name,"attn_qkv")||strstr(name,"attn_gate"))) return rules[i].type;
            if (!strcmp(role,"ssm") && strstr(name,"ssm_")) return rules[i].type;
            if (!strcmp(role,"shared") && (strstr(name,"ffn_gate_shexp")||strstr(name,"ffn_up_shexp")||
                strstr(name,"ffn_down_shexp")||strstr(name,"ffn_gate_up_shexp"))) return rules[i].type;
            if (!strcmp(role,"norms") && (strstr(name,"attn_norm")||strstr(name,"ffn_norm")||
                strstr(name,"post_attention_norm")||strstr(name,"attn_q_norm")||strstr(name,"attn_k_norm"))) return rules[i].type;
            if (!strcmp(role,"gates") && strstr(name,"ffn_gate_inp")) return rules[i].type;
        } else if (strncmp(pat, "global.", 7) == 0) {
            const char *r = pat+7;
            if (!strcmp(r,"embed") && strstr(name,"token_embd")) return rules[i].type;
            if (!strcmp(r,"output") && (!strcmp(name,"output.weight") || !strcmp(name,"output"))) return rules[i].type;
            if (!strcmp(r,"output_norm") && strstr(name,"output_norm")) return rules[i].type;
        }
    }
    return base_type;
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static int n_threads = 0;  // auto-detect

// Check if tensor should come from fallback (SSM + norm tensors)
static int needs_fallback(const char *name) {
    return strstr(name, "ssm_") || strstr(name, "attn_norm") ||
           strstr(name, "ffn_norm") || strstr(name, "post_attention_norm") ||
           strstr(name, "output_norm") || strstr(name, "attn_q_norm") ||
           strstr(name, "attn_k_norm");
}

// Check if tensor is a Qwen3_5MoeRMSNorm weight (needs +1.0 offset)
// HuggingFace Qwen3.5: weight = nn.Parameter(torch.zeros(dim))
//   forward: output * (1.0 + self.weight)
// ggml: output * weight (no +1.0 added at runtime)
// So GGUF must store (weight + 1.0) for these norms.
//
// EXCLUDES ssm_norm: uses Qwen3_5MoeRMSNormGated which inits to torch.ones
//   and uses weight directly (no +1.0 offset)
static int is_norm_needing_offset(const char *name) {
    // Must match: attn_norm, ffn_norm, output_norm, post_attention_norm,
    //             attn_q_norm, attn_k_norm
    // Must NOT match: ssm_norm (RMSNormGated, no offset needed)
    if (strstr(name, "ssm_norm")) return 0;  // exclude first
    return strstr(name, "attn_norm") || strstr(name, "ffn_norm") ||
           strstr(name, "post_attention_norm") || strstr(name, "output_norm") ||
           strstr(name, "attn_q_norm") || strstr(name, "attn_k_norm");
}

// Check if tensor is ssm_a (needs -exp() transform)
// HuggingFace stores A_log (log-space), ggml expects -exp(A_log)
// HF: g = -A_log.exp() * softplus(a + dt_bias)
// ggml: gate = softplus(alpha + dt_bias) * ssm_a   (ssm_a = -exp(A_log))
static int is_ssm_a_tensor(const char *name) {
    // Match "ssm_a" but NOT "ssm_alpha"
    const char *p = strstr(name, "ssm_a");
    if (!p) return 0;
    // Check next char after "ssm_a" is end, '.', or nothing (not "lpha")
    char next = p[5];  // char after "ssm_a"
    return (next == '\0' || next == '.');
}

// Add +1.0 offset to norm weights (HF format -> ggml format)
static void apply_norm_offset(float *data, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        data[i] += 1.0f;
    }
}

// Deinterleave SSM head groups along the dt_rank dimension.
// HF Qwen3.5 stores SSM tensors with interleaved head groups (n_group=16, dt_rank=32).
//
// For 1D tensors (ssm_a [32], ssm_dt.bias [32]):
//   reshape(n_group=16, per_group=2).T.flatten()
//   indices: [0,16,1,17,2,18,...] -> [0,1,2,...,15,16,17,...,31]
//
// For 2D tensors (ssm_alpha [ne0=2048, ne1=32], ssm_beta [ne0=2048, ne1=32]):
//   The dt_rank=32 is ne[1] (the row dimension in GGUF storage).
//   Stored as 32 rows of ne0 elements each.
//   Deinterleave the row indices: reshape(16, 2, ne0) -> transpose(1, 0, 2) -> reshape(32, ne0)
//   i.e., row reorder: [0,1,2,...,31] -> [0,2,4,...,30,1,3,...,31]
//
// ne0 = inner dimension (row length), nrows = number of rows to deinterleave
// For 1D: ne0=n, nrows=1 (deinterleave the single "row" of n elements)
// For 2D: ne0=ne[0], nrows=ne[1]=32
static void deinterleave_ssm(float *data, int64_t ne0, int64_t nrows) {
    int64_t n_group = 16;  // Qwen3.5 ssm_n_group

    if (nrows <= 1) {
        // 1D case: deinterleave ne0 elements as reshape(n_group, ne0/n_group).T.flatten()
        int64_t per_group = ne0 / n_group;
        float *tmp = malloc(ne0 * sizeof(float));
        for (int64_t j = 0; j < per_group; j++) {
            for (int64_t g = 0; g < n_group; g++) {
                tmp[j * n_group + g] = data[g * per_group + j];
            }
        }
        memcpy(data, tmp, ne0 * sizeof(float));
        free(tmp);
    } else {
        // 2D case: deinterleave along the row dimension (nrows = dt_rank = 32)
        // Row reorder: reshape(n_group, per_group, ne0).transpose(1,0,2).reshape(nrows, ne0)
        int64_t per_group = nrows / n_group;
        int64_t total = nrows * ne0;
        float *tmp = malloc(total * sizeof(float));
        for (int64_t j = 0; j < per_group; j++) {
            for (int64_t g = 0; g < n_group; g++) {
                int64_t src_row = g * per_group + j;
                int64_t dst_row = j * n_group + g;
                memcpy(tmp + dst_row * ne0, data + src_row * ne0, ne0 * sizeof(float));
            }
        }
        memcpy(data, tmp, total * sizeof(float));
        free(tmp);
    }
}

// Check if tensor needs head deinterleaving.
// Affected: ssm_a, ssm_dt.bias, ssm_alpha, ssm_beta — all have dt_rank=32 dimension
// NOT affected: ssm_conv1d (conv kernel), ssm_out (projection), ssm_norm (no dt_rank)
static int needs_deinterleave(const char *name) {
    if (strstr(name, "ssm_a") && !strstr(name, "ssm_alpha")) return 1;
    if (strstr(name, "ssm_dt")) return 1;
    if (strstr(name, "ssm_alpha")) return 1;
    if (strstr(name, "ssm_beta")) return 1;
    return 0;
}

static void apply_neg_exp(float *data, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        data[i] = -expf(data[i]);
    }
}

static void bf16_to_f32(const uint16_t *src, float *dst, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        uint32_t tmp = (uint32_t)src[i] << 16;
        memcpy(&dst[i], &tmp, 4);
    }
}

// Fix tensor name: add .weight suffix where needed
// BF16 GGUF may or may not have .weight suffix. ik_llama expects .weight on most tensors.
//   - ssm_a → ssm_a (NO suffix — it's a scalar parameter)
//   - ssm_dt.bias → ssm_dt.bias (already has suffix)
//   - *.weight → keep as-is
//   - everything else → add .weight
static void fix_tensor_name(const char *src, char *dst, size_t maxlen) {
    size_t len = strlen(src);
    // Already has .weight or .bias suffix → keep
    if ((len > 7 && strcmp(src + len - 7, ".weight") == 0) ||
        (len > 5 && strcmp(src + len - 5, ".bias") == 0)) {
        strncpy(dst, src, maxlen);
        dst[maxlen-1] = 0;
        return;
    }
    // ssm_a has no suffix in both formats
    if (strstr(src, "ssm_a") && !strstr(src, "ssm_alpha")) {
        strncpy(dst, src, maxlen);
        dst[maxlen-1] = 0;
        return;
    }
    // Everything else: add .weight
    snprintf(dst, maxlen, "%s.weight", src);
    dst[maxlen-1] = 0;
}

// Quantize a F32 buffer to target type — MULTI-THREADED via OpenMP
static size_t quantize_and_write(FILE *fout, const float *f32, int64_t n_elements,
                                  int64_t n_per_row, enum ggml_type tgt) {
    int64_t nrows = n_elements / n_per_row;
    size_t row_size = ggml_row_size(tgt, n_per_row);
    size_t out_size = row_size * nrows;
    size_t alloc = out_size + 4096;
    void *buf = malloc(alloc);

    int nt = n_threads > 0 ? n_threads : 1;

    if (nrows >= nt * 4 && n_elements > 100000) {
        // Parallel: split rows across threads
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int t = 0; t < nt; t++) {
            int64_t start = (nrows * t) / nt;
            int64_t end   = (nrows * (t + 1)) / nt;
            int64_t chunk_rows = end - start;
            if (chunk_rows <= 0) continue;

            const float *src = f32 + start * n_per_row;
            void *dst = (char *)buf + start * row_size;
            ggml_quantize_chunk(tgt, src, dst, 0, chunk_rows, n_per_row, NULL);
        }
    } else {
        // Single-thread for small tensors
        ggml_quantize_chunk(tgt, f32, buf, 0, nrows, n_per_row, NULL);
    }

    size_t actual = row_size * nrows;
    fwrite(buf, 1, actual, fout);
    free(buf);
    return actual;
}

// Dequantize any tensor to F32
static float *dequant_to_f32(const void *data, int64_t n_elements, enum ggml_type src_type) {
    float *f32 = malloc((size_t)n_elements * sizeof(float));
    if (!f32) return NULL;

    if (src_type == GGML_TYPE_F32) {
        memcpy(f32, data, n_elements * sizeof(float));
    } else if (src_type == GGML_TYPE_BF16) {
        bf16_to_f32((const uint16_t *)data, f32, n_elements);
    } else if (src_type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)data, f32, n_elements);
    } else {
        ggml_type_traits_t tt = ggml_internal_get_type_traits(src_type);
        if (tt.to_float) tt.to_float(data, f32, n_elements);
        else memset(f32, 0, n_elements * sizeof(float));
    }
    return f32;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <config.json> [--fallback <fallback.gguf>]\n", argv[0]);
        fprintf(stderr, "  --fallback: secondary GGUF for SSM/norm tensors (use when primary has rotated SSM)\n");
        return 1;
    }
    const char *input_path = argv[1], *output_path = argv[2], *config_path = argv[3];
    const char *fallback_path = NULL;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--fallback") == 0 && i + 1 < argc) fallback_path = argv[++i];
    }

    if (load_config(config_path) < 0) return 1;

    // Detect threads (i5-14600KF: 6P+8E = 14 threads)
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #else
    n_threads = 1;
    #endif
    printf("Threads: %d\n", n_threads);

    for (int t = 0; t < GGML_TYPE_COUNT; t++) ggml_quantize_init((enum ggml_type)t);

    // Open input
    struct ggml_context *ctx_meta = NULL;
    struct gguf_init_params gp = { .no_alloc = true, .ctx = &ctx_meta };
    struct gguf_context *ctx_in = gguf_init_from_file(input_path, gp);
    if (!ctx_in) { fprintf(stderr, "Failed to open: %s\n", input_path); return 1; }

    int n_tensors = gguf_get_n_tensors(ctx_in);
    size_t data_offset = gguf_get_data_offset(ctx_in);
    printf("Input: %s (%d tensors)\n", input_path, n_tensors);

    // mmap input
    int fd = open(input_path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    void *mm = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    madvise(mm, st.st_size, MADV_SEQUENTIAL);

    // Open fallback GGUF (for SSM/norm tensors from non-rotated model)
    struct ggml_context *ctx_fb_meta = NULL;
    struct gguf_context *ctx_fb = NULL;
    void *mm_fb = NULL;
    size_t fb_data_offset = 0;
    if (fallback_path) {
        struct gguf_init_params gp_fb = { .no_alloc = true, .ctx = &ctx_fb_meta };
        ctx_fb = gguf_init_from_file(fallback_path, gp_fb);
        if (!ctx_fb) { fprintf(stderr, "Failed to open fallback: %s\n", fallback_path); return 1; }
        fb_data_offset = gguf_get_data_offset(ctx_fb);
        int fd_fb = open(fallback_path, O_RDONLY);
        struct stat st_fb; fstat(fd_fb, &st_fb);
        mm_fb = mmap(NULL, st_fb.st_size, PROT_READ, MAP_PRIVATE, fd_fb, 0);
        madvise(mm_fb, st_fb.st_size, MADV_SEQUENTIAL);
        printf("Fallback: %s (%d tensors, for SSM/norm)\n", fallback_path, gguf_get_n_tensors(ctx_fb));
    }

    // Count output tensors (skip blk.40 MTP, keep fused gate_up intact)
    int n_out = 0;
    for (int i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(ctx_in, i);
        if (strstr(name, "blk.40.")) continue;  // skip MTP layer
        n_out += 1;
    }
    printf("Output tensors: %d (fused gate_up kept intact, MTP pruned)\n", n_out);

    // Build output GGUF header
    struct gguf_context *ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);

    // CRITICAL: Remove any general.alignment inherited from the input GGUF.
    // gguf_init_empty() sets ctx_out->alignment = 32 (GGUF_DEFAULT_ALIGNMENT).
    // gguf_set_kv() copies KV pairs including general.alignment from the input,
    // but does NOT update ctx_out->alignment. This causes a mismatch: the writer
    // computes tensor offsets and header padding with alignment=32, but the reader
    // (gguf_init_from_file) reads general.alignment from KV and uses it for
    // data_offset calculation. If the input had alignment != 32, ALL tensor
    // positions would be wrong in the output file.
    // Fix: force general.alignment to match the writer's alignment (32).
    gguf_set_val_u32(ctx_out, "general.alignment", GGUF_DEFAULT_ALIGNMENT);

    // Fix MTP: prune layer 40 (nextn_predict_layers=0, block_count=40)
    gguf_set_val_u32(ctx_out, "qwen35moe.block_count", 40);
    gguf_set_val_u32(ctx_out, "qwen35moe.nextn_predict_layers", 0);
    // Fix SSM inner_size: BF16 GGUF has 8192 but runtime needs 4096
    gguf_set_val_u32(ctx_out, "qwen35moe.ssm.inner_size", 4096);
    // Fix file_type to Q4_K_M (15) — the base quant type
    gguf_set_val_u32(ctx_out, "general.file_type", 15);
    // Keep expert_shared_count=1 from BF16 source (the model has 1 shared expert per layer)
    // Setting to 0 disables shared experts → garbage output
    // Fix tokenizer pre-type
    gguf_set_val_str(ctx_out, "tokenizer.ggml.pre", "qwen35");
    printf("Fixed metadata: block_count=40, nextn=0, ssm.inner_size=4096, file_type=15, tokenizer=qwen35\n");

    size_t ctx_size = ggml_tensor_overhead() * (n_out + 10) + 2*1024*1024;
    struct ggml_init_params ip = { .mem_size = ctx_size, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context *ctx_tmp = ggml_init(ip);

    // First pass: register all output tensors in header
    for (int i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(ctx_in, i);
        if (strstr(name, "blk.40.")) continue;  // skip MTP layer
        struct ggml_tensor *t = ggml_get_tensor(ctx_meta, name);
        if (!t) continue;

        int64_t nel = ggml_nelements(t);
        enum ggml_type tgt = match_tensor(name);
        // Small tensors or tensors with ne[0] < 32: must be F32 for runtime
        if (nel < 65536 || t->ne[0] < 32) tgt = GGML_TYPE_F32;

        char oname[512];

        // No split — ik_llama supports fused gate_up_exps natively
        {
            fix_tensor_name(name, oname, sizeof(oname));
            struct ggml_tensor *to = ggml_new_tensor(ctx_tmp, tgt, GGML_MAX_DIMS, t->ne);
            ggml_set_name(to, oname);
            gguf_add_tensor(ctx_out, to);
        }
    }

    // Write header (metadata + tensor infos + alignment padding)
    printf("Writing GGUF header (%d tensors) to %s ...\n", n_out, output_path);
    gguf_write_to_file(ctx_out, output_path, /*only_meta=*/true);

    // BUG FIX: gguf_get_data_offset(ctx_out) returns 0 for contexts created with
    // gguf_init_empty() because ctx->offset is only set during gguf_init_from_file().
    // Use gguf_get_meta_size() which computes the actual header size by doing a dry
    // run of the serialization (includes alignment padding to 32 bytes).
    size_t data_section_start = gguf_get_meta_size(ctx_out);

    // Second pass: quantize and write tensor data SEQUENTIALLY
    // Instead of seeking to pre-computed offsets (which could mismatch if ggml_nbytes
    // differs from actual quantized size), we write data sequentially and pad each
    // tensor to GGML_PAD(size, 32) — exactly as gguf_write_to_buf does.
    // This guarantees the actual file layout matches the offsets in the header.
    FILE *fout = fopen(output_path, "r+b");  // open for read+write (don't truncate)
    if (!fout) { fprintf(stderr, "Failed to open output for writing: %s\n", output_path); return 1; }
    fseek(fout, data_section_start, SEEK_SET);
    printf("Data section starts at offset: %zu (%.2f KiB)\n", data_section_start, data_section_start / 1024.0);

    static const uint8_t zero_pad[32] = {0};
    int64_t total_in = 0, total_out = 0;
    int tensor_idx = 0;
    int out_idx = 0;  // tracks position in ctx_out tensor list
    size_t cumulative_offset = 0;  // running offset within data section for verification

    for (int i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(ctx_in, i);
        if (strstr(name, "blk.40.")) continue;
        struct ggml_tensor *t = ggml_get_tensor(ctx_meta, name);
        if (!t) continue;

        // Route to fallback for SSM/norm tensors if available
        int use_fb = 0;
        struct ggml_tensor *t_fb = NULL;
        if (ctx_fb && needs_fallback(name)) {
            // Find this tensor in the fallback GGUF by index
            int fb_idx = gguf_find_tensor(ctx_fb, name);
            if (fb_idx >= 0) {
                t_fb = ggml_get_tensor(ctx_fb_meta, name);
                if (t_fb) {
                    use_fb = 1;
                } else {
                    fprintf(stderr, "WARN: fallback tensor %s found in gguf but not in ggml context\n", name);
                }
            }
        }

        struct ggml_tensor *t_src = use_fb ? t_fb : t;
        enum ggml_type src_type = t_src->type;
        int64_t nel = ggml_nelements(t_src);
        size_t src_size = ggml_nbytes(t_src);

        const void *src_data;
        if (use_fb) {
            int fb_idx = gguf_find_tensor(ctx_fb, name);
            size_t fb_off = fb_data_offset + gguf_get_tensor_offset(ctx_fb, fb_idx);
            src_data = (const char *)mm_fb + fb_off;
            printf("[%4d/%4d] %-45s [FALLBACK]\n", tensor_idx+1, n_out, name);
        } else {
            size_t tensor_off = data_offset + gguf_get_tensor_offset(ctx_in, i);
            src_data = (const char *)mm + tensor_off;
        }
        total_in += src_size;

        enum ggml_type tgt = match_tensor(name);
        if (nel < 65536 || t->ne[0] < 32) tgt = GGML_TYPE_F32;

        // Check if this tensor needs a data transformation (HF -> ggml convention)
        int need_norm_offset  = is_norm_needing_offset(name);
        int need_neg_exp      = is_ssm_a_tensor(name);
        int need_deinterleave = needs_deinterleave(name);
        int need_transform    = need_norm_offset || need_neg_exp || need_deinterleave;

        if (src_type == tgt && !need_transform) {
            // Copy as-is (no type change, no data transform needed)
            size_t expected_off = gguf_get_tensor_offset(ctx_out, out_idx);
            if (cumulative_offset != expected_off) {
                fprintf(stderr, "OFFSET MISMATCH tensor %d (copy): cumulative=%zu expected=%zu diff=%zd\n",
                        out_idx, cumulative_offset, expected_off, (ssize_t)(cumulative_offset - expected_off));
            }
            fwrite(src_data, 1, src_size, fout);
            size_t pad = GGML_PAD(src_size, 32) - src_size;
            if (pad > 0) fwrite(zero_pad, 1, pad, fout);
            cumulative_offset += src_size + pad;
            total_out += src_size;
            tensor_idx++; out_idx++;
            if (src_size > 1024*1024)
                printf("[%4d/%4d] %-45s %6s         %8.1f MiB [keep]\n",
                       tensor_idx, n_out, name, ggml_type_name(src_type), src_size/(1024.0*1024.0));
        } else {
            // Dequant + transform + quantize (or just dequant to F32)
            size_t expected_off = gguf_get_tensor_offset(ctx_out, out_idx);
            if (cumulative_offset != expected_off) {
                fprintf(stderr, "OFFSET MISMATCH tensor %d (quant): cumulative=%zu expected=%zu diff=%zd\n",
                        out_idx, cumulative_offset, expected_off, (ssize_t)(cumulative_offset - expected_off));
            }
            float *f32 = dequant_to_f32(src_data, nel, src_type);

            // Apply HF -> ggml data transformations
            if (need_norm_offset) {
                apply_norm_offset(f32, nel);
                printf("  [TRANSFORM] %s: +1.0 (RMSNorm offset, %ld elements)\n", name, (long)nel);
            }
            if (need_deinterleave) {
                int64_t ne0 = t_src->ne[0];
                int64_t nrows = (ggml_n_dims(t_src) >= 2) ? t_src->ne[1] : 1;
                deinterleave_ssm(f32, ne0, nrows);
                printf("  [TRANSFORM] %s: deinterleave (n_group=16, ne0=%ld, nrows=%ld)\n",
                       name, (long)ne0, (long)nrows);
            }
            if (need_neg_exp) {
                printf("  [TRANSFORM] %s: -exp() (A_log -> ssm_a, values before: [%.4f, %.4f, ...], ",
                       name, f32[0], nel > 1 ? f32[1] : 0.0f);
                apply_neg_exp(f32, nel);
                printf("after: [%.4f, %.4f, ...])\n", f32[0], nel > 1 ? f32[1] : 0.0f);
            }

            size_t actual;
            if (tgt == GGML_TYPE_F32) {
                // F32 target: just write the transformed floats directly
                actual = nel * sizeof(float);
                fwrite(f32, 1, actual, fout);
            } else {
                actual = quantize_and_write(fout, f32, nel, t->ne[0], tgt);
            }
            size_t pad = GGML_PAD(actual, 32) - actual;
            if (pad > 0) fwrite(zero_pad, 1, pad, fout);
            cumulative_offset += actual + pad;
            total_out += actual;
            tensor_idx++; out_idx++;
            const char *tag = need_transform ? (src_type == tgt ? " [transform]" : "") : "";
            printf("[%4d/%4d] %-45s %6s -> %-6s %8.1f -> %8.1f MiB%s\n",
                   tensor_idx, n_out, name,
                   ggml_type_name(src_type), ggml_type_name(tgt),
                   src_size/(1024.0*1024.0), actual/(1024.0*1024.0), tag);
            free(f32);
        }
    }

    // Truncate file to exact size (safety: prevents stale data from prior runs)
    size_t final_size = data_section_start + cumulative_offset;
    fflush(fout);
    if (ftruncate(fileno(fout), final_size) != 0) {
        perror("ftruncate");
    }
    fclose(fout);

    // Verify: expected size from header offsets
    size_t last_tensor_offset = gguf_get_tensor_offset(ctx_out, n_out - 1);
    // The last tensor's end = offset + GGML_PAD(size, 32) but we don't store size
    // separately. Use cumulative_offset which tracks the same thing.
    printf("\nData section: start=%zu, cumulative=%zu, last_tensor_offset=%zu\n",
           data_section_start, cumulative_offset, last_tensor_offset);
    printf("Final file size: %.2f GB (data section: %.2f GB)\n",
           final_size/(1024.0*1024.0*1024.0), cumulative_offset/(1024.0*1024.0*1024.0));

    printf("\n========================================\n");
    printf("Input:  %.2f GB\n", total_in/(1024.0*1024.0*1024.0));
    printf("Output: %.2f GB\n", total_out/(1024.0*1024.0*1024.0));
    printf("Ratio:  %.1f%%\n", 100.0*total_out/total_in);
    printf("Tensors: %d\n", tensor_idx);
    printf("========================================\n");

    // ----------------------------------------------------------------
    // Post-write validation: re-open the output GGUF and verify every
    // tensor's (data_offset + tensor_offset + nbytes) <= file_size.
    // This is the exact check llama-server does at load time.
    // ----------------------------------------------------------------
    printf("\nValidating output GGUF ...\n");
    {
        struct stat ost; stat(output_path, &ost);
        size_t file_sz = ost.st_size;

        struct ggml_context *vctx = NULL;
        struct gguf_init_params vp = { .no_alloc = true, .ctx = &vctx };
        struct gguf_context *vguf = gguf_init_from_file(output_path, vp);
        if (!vguf) {
            fprintf(stderr, "VALIDATION FAILED: cannot re-open %s\n", output_path);
        } else {
            int vn = gguf_get_n_tensors(vguf);
            size_t vdata_off = gguf_get_data_offset(vguf);
            printf("  Validation: %d tensors, data_offset=%zu, file_size=%zu\n", vn, vdata_off, file_sz);
            printf("  Writer data_section_start was: %zu (delta=%zd)\n",
                   data_section_start, (ssize_t)(vdata_off - data_section_start));
            int errors = 0;
            for (int vi = 0; vi < vn; vi++) {
                const char *vname = gguf_get_tensor_name(vguf, vi);
                struct ggml_tensor *vt = ggml_get_tensor(vctx, vname);
                if (!vt) { fprintf(stderr, "  MISSING tensor %s\n", vname); errors++; continue; }
                size_t voff = vdata_off + gguf_get_tensor_offset(vguf, vi);
                size_t vnb = ggml_nbytes(vt);
                if (voff + vnb > file_sz) {
                    fprintf(stderr, "  OUT-OF-BOUNDS: tensor %d '%s' offset=%zu nbytes=%zu end=%zu file=%zu (over by %zu)\n",
                            vi, vname, voff, vnb, voff + vnb, file_sz, (voff + vnb) - file_sz);
                    errors++;
                }
            }
            if (errors == 0) {
                printf("  Validation PASSED: all %d tensors within file bounds\n", vn);
            } else {
                fprintf(stderr, "  Validation FAILED: %d tensor(s) out of bounds\n", errors);
            }
            gguf_free(vguf);
            if (vctx) ggml_free(vctx);
        }
    }

    munmap(mm, st.st_size); close(fd);
    ggml_free(ctx_tmp); gguf_free(ctx_out); gguf_free(ctx_in);
    if (ctx_meta) ggml_free(ctx_meta);
    ggml_quantize_free();
    return 0;
}
