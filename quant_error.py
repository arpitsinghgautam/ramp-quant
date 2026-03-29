#!/usr/bin/env python3
"""
RAMP-Local: Round-Trip Quantization Error Measurement

Measures actual quantization error per tensor group by:
1. Loading BF16 weights from safetensors (shard by shard, ~5 GB RAM peak)
2. Simulating quantization to each candidate GGUF type
3. Computing Frobenius relative error, MSE, max error, cosine similarity
4. Saving results for QuantErrorDB (proxy_model.py)

For MoE experts: samples K experts per layer to save time.

Usage:
    python3 quant_error.py ~/.chimere/models/Qwen3.5-35B-A3B-BF16-rotated/
    python3 quant_error.py model_dir/ --expert-sample-k 8 --output cache/quant_errors.json
"""

from __future__ import annotations
import argparse, json, time, sys, gc, re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# Quantization types to measure (same as SEARCH_QUANT_TYPES in gguf_analyzer)
CANDIDATE_TYPES = ["IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S",
                   "Q4_K", "Q5_K", "Q6_K", "Q8_0"]

# BPW for each type (for simulation)
TYPE_BPW = {
    "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125, "IQ2_S": 2.5625,
    "IQ3_XXS": 3.0625, "IQ3_S": 3.4375,
    "Q4_K": 4.5, "Q5_K": 5.5, "Q6_K": 6.5625, "Q8_0": 8.5,
}


# ---------------------------------------------------------------------------
# Quantization simulation
# ---------------------------------------------------------------------------

def simulate_q8_0(W: np.ndarray) -> np.ndarray:
    """Simulate Q8_0: 32-element blocks, int8 + f16 scale."""
    flat = W.flatten().astype(np.float32)
    n = len(flat)
    pad = (32 - n % 32) % 32
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    blocks = flat.reshape(-1, 32)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)
    # Quantize to int8 range [-127, 127]
    quantized = np.round(blocks / scales * 127.0).clip(-127, 127)
    dequantized = (quantized / 127.0 * scales).flatten()[:n]
    return dequantized


def simulate_q6_k(W: np.ndarray) -> np.ndarray:
    """Simulate Q6_K: 256-element superblocks, 6-bit quantization."""
    flat = W.flatten().astype(np.float32)
    n = len(flat)
    pad = (256 - n % 256) % 256
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    blocks = flat.reshape(-1, 256)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)
    # 6-bit: [-31, 31] range
    quantized = np.round(blocks / scales * 31.0).clip(-31, 31)
    dequantized = (quantized / 31.0 * scales).flatten()[:n]
    return dequantized


def simulate_q5_k(W: np.ndarray) -> np.ndarray:
    """Simulate Q5_K: 256-element superblocks, 5-bit with min."""
    flat = W.flatten().astype(np.float32)
    n = len(flat)
    pad = (256 - n % 256) % 256
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    blocks = flat.reshape(-1, 256)
    bmin = np.min(blocks, axis=1, keepdims=True)
    bmax = np.max(blocks, axis=1, keepdims=True)
    scale = bmax - bmin
    scale = np.where(scale == 0, 1.0, scale)
    # 5-bit: [0, 31] range
    quantized = np.round((blocks - bmin) / scale * 31.0).clip(0, 31)
    dequantized = (quantized / 31.0 * scale + bmin).flatten()[:n]
    return dequantized


def simulate_q4_k(W: np.ndarray) -> np.ndarray:
    """Simulate Q4_K: 256-element superblocks, 4-bit with min."""
    flat = W.flatten().astype(np.float32)
    n = len(flat)
    pad = (256 - n % 256) % 256
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    blocks = flat.reshape(-1, 256)
    bmin = np.min(blocks, axis=1, keepdims=True)
    bmax = np.max(blocks, axis=1, keepdims=True)
    scale = bmax - bmin
    scale = np.where(scale == 0, 1.0, scale)
    # 4-bit: [0, 15] range
    quantized = np.round((blocks - bmin) / scale * 15.0).clip(0, 15)
    dequantized = (quantized / 15.0 * scale + bmin).flatten()[:n]
    return dequantized


def simulate_iq3_s(W: np.ndarray) -> np.ndarray:
    """Simulate IQ3_S: ~3.44 BPW, approximate with 3-bit symmetric."""
    flat = W.flatten().astype(np.float32)
    n = len(flat)
    pad = (256 - n % 256) % 256
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    blocks = flat.reshape(-1, 256)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)
    # 3-bit symmetric: [-3, 3] with 7 levels + 0
    quantized = np.round(blocks / scales * 3.0).clip(-3, 3)
    dequantized = (quantized / 3.0 * scales).flatten()[:n]
    return dequantized


def simulate_iq3_xxs(W: np.ndarray) -> np.ndarray:
    """Simulate IQ3_XXS: ~3.06 BPW, coarser 3-bit."""
    flat = W.flatten().astype(np.float32)
    n = len(flat)
    pad = (256 - n % 256) % 256
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    blocks = flat.reshape(-1, 256)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)
    # Coarser 3-bit: fewer effective levels due to XXS overhead
    quantized = np.round(blocks / scales * 2.5).clip(-3, 3)
    dequantized = (quantized / 2.5 * scales).flatten()[:n]
    return dequantized


def simulate_iq2_xs(W: np.ndarray) -> np.ndarray:
    """Simulate IQ2_XS: ~2.31 BPW, 2-bit with extra scales."""
    flat = W.flatten().astype(np.float32)
    n = len(flat)
    pad = (256 - n % 256) % 256
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    blocks = flat.reshape(-1, 256)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)
    # 2-bit: [-1, 0, 1] with scale
    quantized = np.round(blocks / scales * 1.5).clip(-2, 1)
    dequantized = (quantized / 1.5 * scales).flatten()[:n]
    return dequantized


def simulate_iq2_xxs(W: np.ndarray) -> np.ndarray:
    """Simulate IQ2_XXS: ~2.06 BPW, aggressive 2-bit."""
    flat = W.flatten().astype(np.float32)
    n = len(flat)
    pad = (256 - n % 256) % 256
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    blocks = flat.reshape(-1, 256)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)
    quantized = np.round(blocks / scales).clip(-1, 1)
    dequantized = (quantized * scales).flatten()[:n]
    return dequantized


SIMULATORS = {
    "Q8_0": simulate_q8_0,
    "Q6_K": simulate_q6_k,
    "Q5_K": simulate_q5_k,
    "Q4_K": simulate_q4_k,
    "IQ3_S": simulate_iq3_s,
    "IQ3_XXS": simulate_iq3_xxs,
    "IQ2_XS": simulate_iq2_xs,
    "IQ2_XXS": simulate_iq2_xxs,
}


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def compute_errors(W_orig: np.ndarray, W_quant: np.ndarray) -> dict:
    """Compute error metrics between original and quantized weights."""
    diff = W_orig - W_quant
    frobenius_orig = np.linalg.norm(W_orig)
    frobenius_diff = np.linalg.norm(diff)

    frobenius_rel = float(frobenius_diff / frobenius_orig) if frobenius_orig > 1e-12 else 0.0
    mse = float(np.mean(diff ** 2))
    max_error = float(np.max(np.abs(diff)))

    # Cosine similarity
    dot = np.dot(W_orig.flatten(), W_quant.flatten())
    norm_orig = np.linalg.norm(W_orig)
    norm_quant = np.linalg.norm(W_quant)
    cosine_sim = float(dot / (norm_orig * norm_quant)) if (norm_orig > 1e-12 and norm_quant > 1e-12) else 1.0

    return {
        "frobenius_rel": frobenius_rel,
        "mse": mse,
        "max_error": max_error,
        "cosine_sim": cosine_sim,
    }


# ---------------------------------------------------------------------------
# HF tensor name -> decision group mapping
# ---------------------------------------------------------------------------

HF_ROLE_PATTERNS = [
    (re.compile(r'.*layers\.(\d+)\.self_attn\.(q|k|v|o)_proj'), 'attn'),
    (re.compile(r'.*layers\.(\d+)\.self_attn\..*'), 'attn'),
    (re.compile(r'.*layers\.(\d+)\.mlp\.shared_expert\.(gate|up|down)_proj'), 'shared'),
    (re.compile(r'.*layers\.(\d+)\.mlp\.experts\.(gate_up|down)_proj'), 'experts'),
    (re.compile(r'.*layers\.(\d+)\.mlp\.gate\.'), 'gates'),
    (re.compile(r'.*layers\.(\d+)\.(input_layernorm|post_attention_layernorm)'), 'norms'),
    (re.compile(r'.*embed_tokens\.'), 'global.embed'),
    (re.compile(r'.*\.norm\.weight'), 'global.output_norm'),
    (re.compile(r'lm_head\.'), 'global.output'),
]


def hf_name_to_group(name: str) -> str:
    """Map HuggingFace tensor name to RAMP-Local decision group."""
    for pattern, role in HF_ROLE_PATTERNS:
        m = pattern.match(name)
        if m:
            if role.startswith("global."):
                return role
            layer_idx = m.group(1)
            return f"layer.{layer_idx}.{role}"
    return "unknown"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_shard(shard_path: Path, qtypes: list[str],
                  expert_sample_k: int = 16,
                  verbose: bool = True) -> dict:
    """Analyze all tensors in a safetensors shard.

    Returns: {group_name: {qtype: {frobenius_rel, mse, max_error, cosine_sim}}}
    Aggregates within groups using element-weighted average.
    """
    import safetensors.torch as st

    tensors = st.load_file(str(shard_path))
    results = defaultdict(lambda: defaultdict(lambda: {
        "frobenius_rel": 0.0, "mse": 0.0, "max_error": 0.0,
        "cosine_sim": 0.0, "total_elements": 0, "n_tensors": 0,
    }))

    # Track expert tensors for sampling
    expert_tensors = defaultdict(list)
    non_expert_tensors = []

    for name in tensors:
        group = hf_name_to_group(name)
        if ".experts" in group:
            expert_tensors[group].append(name)
        else:
            non_expert_tensors.append(name)

    # Sample experts
    tensors_to_process = list(non_expert_tensors)
    rng = np.random.RandomState(42)
    for group, names in expert_tensors.items():
        if len(names) > expert_sample_k:
            sampled = rng.choice(names, size=expert_sample_k, replace=False).tolist()
            tensors_to_process.extend(sampled)
        else:
            tensors_to_process.extend(names)

    for i, name in enumerate(tensors_to_process):
        W = tensors[name]
        w_np = W.float().numpy()

        if w_np.size < 100:
            continue

        group = hf_name_to_group(name)
        w_flat = w_np.flatten()
        nel = len(w_flat)

        for qtype in qtypes:
            sim_fn = SIMULATORS.get(qtype)
            if sim_fn is None:
                continue
            w_q = sim_fn(w_flat)
            errs = compute_errors(w_flat, w_q)

            # Element-weighted accumulation
            entry = results[group][qtype]
            entry["frobenius_rel"] += errs["frobenius_rel"] * nel
            entry["mse"] += errs["mse"] * nel
            entry["max_error"] = max(entry["max_error"], errs["max_error"])
            entry["cosine_sim"] += errs["cosine_sim"] * nel
            entry["total_elements"] += nel
            entry["n_tensors"] += 1

        del w_np, w_flat

    del tensors
    gc.collect()

    # Normalize weighted averages
    final = {}
    for group, qtypes_data in results.items():
        final[group] = {}
        for qtype, data in qtypes_data.items():
            total = data["total_elements"]
            if total > 0:
                final[group][qtype] = {
                    "frobenius_rel": data["frobenius_rel"] / total,
                    "mse": data["mse"] / total,
                    "max_error": data["max_error"],
                    "cosine_sim": data["cosine_sim"] / total,
                }
    return final


def merge_results(all_results: list[dict]) -> dict:
    """Merge per-shard results into a single database.

    Element-weighted average across shards for the same group.
    """
    merged = defaultdict(lambda: defaultdict(lambda: {
        "frobenius_rel_sum": 0.0, "mse_sum": 0.0,
        "max_error": 0.0, "cosine_sim_sum": 0.0,
        "count": 0,
    }))

    for shard_result in all_results:
        for group, qtypes in shard_result.items():
            for qtype, metrics in qtypes.items():
                entry = merged[group][qtype]
                entry["frobenius_rel_sum"] += metrics["frobenius_rel"]
                entry["mse_sum"] += metrics["mse"]
                entry["max_error"] = max(entry["max_error"], metrics["max_error"])
                entry["cosine_sim_sum"] += metrics["cosine_sim"]
                entry["count"] += 1

    final = {}
    for group, qtypes in merged.items():
        final[group] = {}
        for qtype, data in qtypes.items():
            n = data["count"]
            if n > 0:
                final[group][qtype] = {
                    "frobenius_rel": data["frobenius_rel_sum"] / n,
                    "mse": data["mse_sum"] / n,
                    "max_error": data["max_error"],
                    "cosine_sim": data["cosine_sim_sum"] / n,
                }
    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAMP-Local: Measure round-trip quantization errors on BF16 safetensors")
    parser.add_argument("model_dir", type=Path,
                       help="Path to BF16 safetensors directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                       help="Output JSON path (default: cache/quant_errors.json)")
    parser.add_argument("--expert-sample-k", type=int, default=16,
                       help="Experts to sample per layer (default: 16)")
    parser.add_argument("--qtypes", nargs="+", default=None,
                       help="Quant types to test (default: all SEARCH_QUANT_TYPES)")
    args = parser.parse_args()

    qtypes = args.qtypes or CANDIDATE_TYPES
    print(f"RAMP-Local: Round-Trip Quantization Error Measurement", flush=True)
    print(f"  Model: {args.model_dir}", flush=True)
    print(f"  Types: {', '.join(qtypes)}", flush=True)
    print(f"  Expert sample: {args.expert_sample_k}", flush=True)

    shards = sorted(args.model_dir.glob("*.safetensors"))
    if not shards:
        print("ERROR: No safetensors files found", file=sys.stderr)
        sys.exit(1)
    print(f"  Shards: {len(shards)}", flush=True)

    all_results = []
    t0 = time.monotonic()

    for i, shard in enumerate(shards):
        sz_gb = shard.stat().st_size / (1024**3)
        print(f"\n  [{i+1}/{len(shards)}] {shard.name} ({sz_gb:.1f} GB)...", flush=True)

        t_shard = time.monotonic()
        result = analyze_shard(shard, qtypes, args.expert_sample_k)
        elapsed = time.monotonic() - t_shard

        n_groups = len(result)
        print(f"    {n_groups} groups analyzed in {elapsed:.0f}s", flush=True)
        all_results.append(result)
        gc.collect()

    # Merge
    print(f"\nMerging results...", flush=True)
    merged = merge_results(all_results)

    total_elapsed = time.monotonic() - t0
    print(f"Done: {len(merged)} groups, {total_elapsed:.0f}s total", flush=True)

    # Report top sensitive groups
    print(f"\n{'Group':<35} {'IQ2_XXS':>8} {'IQ3_S':>8} {'Q4_K':>8} {'Q8_0':>8}")
    print("-" * 75)
    sorted_groups = sorted(merged.keys())
    for group in sorted_groups[:30]:
        qtypes_data = merged[group]
        vals = []
        for qt in ["IQ2_XXS", "IQ3_S", "Q4_K", "Q8_0"]:
            if qt in qtypes_data:
                vals.append(f"{qtypes_data[qt]['frobenius_rel']:.5f}")
            else:
                vals.append("   N/A")
        print(f"{group:<35} {'  '.join(vals)}")
    if len(sorted_groups) > 30:
        print(f"  ... and {len(sorted_groups) - 30} more groups")

    # Save
    output = args.output or Path(__file__).parent / "cache" / "quant_errors.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2))
    print(f"\nSaved to: {output}", flush=True)


if __name__ == "__main__":
    main()
