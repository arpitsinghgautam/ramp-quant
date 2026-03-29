# RAMP-Quant

**RAMP: RL-guided Adaptive Mixed-Precision quantization for GGUF models. Produces hardware-optimized quantized models for consumer GPUs.**

RAMP uses data-free sensitivity analysis (no calibration data needed), evolutionary search, and per-tensor type optimization to find the best mixed-precision configuration for your specific hardware and VRAM budget.

## Key result

RAMP v2 produced a **15.2 GB Qwen3.5-35B-A3B GGUF** (3.78 BPW) that runs at **90 tok/s** on RTX 5060 Ti 16GB with 30/30 on functional benchmarks. Custom imatrix + per-tensor overrides (IQ3_S base + Q8_0/Q6_K/Q5_K for SSM/attention critical paths).

## How it works

RAMP runs a 6-phase pipeline:

1. **GGUF Analysis** — parse tensor structure, identify layer types (GDN, attention, MoE experts, norms)
2. **NSDS Sensitivity** — data-free sensitivity scoring per tensor (no calibration data required)
3. **Proxy Model** — build quantization error database, estimate proxy loss for each configuration
4. **Search** — greedy + evolutionary search over mixed-precision configurations within VRAM budget
5. **Validate** — verify the configuration fits hardware constraints
6. **Build** — generate `llama-quantize` command with `--custom-q` tensor overrides

## What's in this repo

```
ramp_local.py           Main orchestrator (6-phase pipeline)
gguf_analyzer.py        GGUF structure analysis
nsds_sensitivity.py     Data-free per-tensor sensitivity (NSDS)
proxy_model.py          Quantization error DB + proxy loss
gguf_builder.py         GGUF construction from config
search_evo.py           Evolutionary + greedy search
quant_error.py          Quantization error estimation
validate.py             Result validation
ramp_quantize.c         C implementation for custom quantization

pipeline/               Extended pipeline modules
  run_pipeline.py       Full pipeline with monitoring
  allocator.py          Bit allocation logic
  sensitivity_analyzer.py  Layer sensitivity analysis
  benchmark.py          Performance/quality benchmark
  optrot_selective.py   Selective optimal rotation
  monitor.py            Pipeline monitoring

tools/                  Quantization utilities
  benchmark_gguf.py     Perplexity + functional domain tests
  kurtboost_bf16.py     Kurtosis-based per-tensor analysis
  build_mtp_gguf.py     Build GGUF with MTP tensors
  inject_mtp_tensors.py Inject MTP tensors into existing GGUF
  patch_gguf_mtp.py     Patch GGUF for MTP support
```

## Usage

```bash
# Run full RAMP pipeline
python ramp_local.py \
  --model path/to/model-BF16.gguf \
  --target-size 15G \
  --gpu-vram 16G

# KurtBoost sensitivity analysis
python tools/kurtboost_bf16.py path/to/model-BF16/ --output overrides.json

# Benchmark a GGUF model
python tools/benchmark_gguf.py path/to/model.gguf --test-functional --test-ppl
```

## The RAMP v2 model

The optimized Qwen3.5-35B-A3B-RAMP-v2-15G GGUF is available on HuggingFace: [chimere-ai/Qwen3.5-35B-A3B-RAMP-v2-15G](https://huggingface.co/chimere-ai/Qwen3.5-35B-A3B-RAMP-v2-15G) (coming soon).

## Related repos

- [chimere](https://github.com/AIdevsmartdata/chimere) — Rust inference runtime that uses RAMP-quantized models
- [chimere-odo](https://github.com/AIdevsmartdata/chimere-odo) — Inference orchestrator

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Author

**Kevin Remondiere** — Independent ML researcher, Bayonne, France
