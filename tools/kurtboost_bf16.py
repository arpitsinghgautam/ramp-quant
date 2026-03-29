#!/usr/bin/env python3
"""kurtboost_bf16.py — KurtBoost sensitivity analysis on BF16 safetensors

Fast shard-by-shard kurtosis analysis. No dequantization needed (BF16 = native float).
Produces --tensor-type overrides for llama-quantize.

Usage:
    python3 kurtboost_bf16.py ~/.chimere/models/Qwen3.5-35B-A3B-BF16-rotated/
    python3 kurtboost_bf16.py ~/.chimere/models/Qwen3.5-35B-A3B-BF16-rotated/ --budget 3.56
"""

from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import safetensors.torch as st
from scipy.stats import kurtosis as scipy_kurtosis


def analyze_shard(shard_path: Path) -> list[dict]:
    """Analyze all tensors in a safetensors shard."""
    tensors = st.load_file(str(shard_path))
    results = []
    for name, W in tensors.items():
        w = W.float().numpy().flatten()
        if len(w) < 100:
            continue
        k = float(scipy_kurtosis(w, fisher=True))
        std = float(np.std(w))
        rng = float(np.max(w) - np.min(w))
        results.append({
            "name": name,
            "kurtosis": k,
            "std": std,
            "range": rng,
            "elements": len(w),
            "shape": list(W.shape),
        })
    del tensors
    return results


# Mapping layer names to GGUF tensor names for --tensor-type
def to_gguf_name(hf_name: str) -> str:
    """Convert HuggingFace tensor name to GGUF tensor name pattern."""
    # model.language_model.layers.N.self_attn.q_proj.weight → blk.N.attn_q.weight
    name = hf_name.replace("model.language_model.layers.", "blk.")
    name = name.replace(".self_attn.q_proj.weight", ".attn_q.weight")
    name = name.replace(".self_attn.k_proj.weight", ".attn_k.weight")
    name = name.replace(".self_attn.v_proj.weight", ".attn_v.weight")
    name = name.replace(".self_attn.o_proj.weight", ".attn_output.weight")
    name = name.replace(".mlp.experts.gate_up_proj", ".ffn_gate_up_exps.weight")
    name = name.replace(".mlp.experts.down_proj", ".ffn_down_exps.weight")
    name = name.replace(".mlp.shared_expert.gate_proj.weight", ".ffn_gate.weight")
    name = name.replace(".mlp.shared_expert.up_proj.weight", ".ffn_up.weight")
    name = name.replace(".mlp.shared_expert.down_proj.weight", ".ffn_down.weight")
    name = name.replace(".mlp.gate.weight", ".ffn_gate_inp.weight")
    name = name.replace(".input_layernorm.weight", ".attn_norm.weight")
    name = name.replace(".post_attention_layernorm.weight", ".ffn_norm.weight")
    name = name.replace("model.language_model.embed_tokens.weight", "token_embd.weight")
    name = name.replace("model.language_model.norm.weight", "output_norm.weight")
    name = name.replace("lm_head.weight", "output.weight")
    return name


# GGUF quant types with approximate BPW
QUANT_TYPES = {
    "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.50,
    "IQ3_XXS": 3.06, "IQ3_S": 3.44, "IQ3_M": 3.70,
    "Q4_K_S": 4.50, "Q4_K_M": 4.85,
    "Q5_K_S": 5.50, "Q5_K_M": 5.69,
    "Q6_K": 6.56, "Q8_0": 8.50,
}


def allocate_bits(tensors: list[dict], budget_bpw: float = 3.56) -> list[dict]:
    """Allocate quant types based on kurtosis sensitivity.

    Higher kurtosis → more bits needed (outlier-heavy = quantization sensitive).
    """
    if not tensors:
        return []

    # Normalize kurtosis scores
    kurts = np.array([t["kurtosis"] for t in tensors])
    median_k = np.median(kurts)
    mad_k = np.median(np.abs(kurts - median_k))
    if mad_k < 0.01:
        mad_k = np.std(kurts)
    if mad_k < 0.01:
        # All tensors similar → uniform allocation
        return [{"name": t["name"], "quant": "IQ3_S", "bpw": 3.44} for t in tensors]

    # Z-score normalized by MAD
    z_scores = (kurts - median_k) / (1.4826 * mad_k)

    # Map z-score to quant type
    allocated = []
    for t, z in zip(tensors, z_scores):
        # Skip norms and gates (always Q8_0)
        if "norm" in t["name"] or "layernorm" in t["name"]:
            allocated.append({"name": t["name"], "quant": "Q8_0", "bpw": 8.5})
            continue
        if t["elements"] < 10000:  # tiny tensors
            allocated.append({"name": t["name"], "quant": "Q8_0", "bpw": 8.5})
            continue

        # Sensitivity zones based on z-score
        if z > 2.0:      quant = "Q6_K"     # Very high kurtosis → needs many bits
        elif z > 1.0:    quant = "Q5_K_S"   # High kurtosis
        elif z > 0.0:    quant = "Q4_K_M"   # Above median
        elif z > -1.0:   quant = "IQ3_S"    # Below median
        else:             quant = "IQ2_XS"   # Very low kurtosis → can compress more

        allocated.append({"name": t["name"], "quant": quant, "bpw": QUANT_TYPES[quant]})

    # Adjust to meet budget
    total_bits = sum(a["bpw"] * t["elements"] for a, t in zip(allocated, tensors))
    total_elements = sum(t["elements"] for t in tensors)
    current_bpw = total_bits / total_elements

    # Iteratively adjust if over budget
    attempts = 0
    while current_bpw > budget_bpw + 0.05 and attempts < 20:
        # Downgrade the least sensitive tensor that's above IQ2_XS
        candidates = [(i, z_scores[i]) for i, a in enumerate(allocated)
                      if a["quant"] not in ("IQ2_XS", "IQ2_XXS", "Q8_0") and "norm" not in tensors[i]["name"]]
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1])  # lowest z-score first
        idx = candidates[0][0]
        # Downgrade one step
        current = allocated[idx]["quant"]
        downgrade = {"Q6_K": "Q5_K_S", "Q5_K_S": "Q4_K_M", "Q4_K_M": "IQ3_S", "IQ3_S": "IQ3_XXS", "IQ3_XXS": "IQ2_XS"}
        if current in downgrade:
            new_q = downgrade[current]
            allocated[idx] = {"name": allocated[idx]["name"], "quant": new_q, "bpw": QUANT_TYPES[new_q]}

        total_bits = sum(a["bpw"] * t["elements"] for a, t in zip(allocated, tensors))
        current_bpw = total_bits / total_elements
        attempts += 1

    return allocated


def main():
    parser = argparse.ArgumentParser(description="KurtBoost on BF16 safetensors")
    parser.add_argument("model_dir", type=Path, help="Path to BF16 safetensors directory")
    parser.add_argument("--budget", type=float, default=3.56, help="Target BPW (default: 3.56)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    print(f"KurtBoost Sensitivity Analysis", flush=True)
    print(f"  Model: {args.model_dir}", flush=True)
    print(f"  Budget: {args.budget} BPW", flush=True)

    shards = sorted(args.model_dir.glob("*.safetensors"))
    print(f"  Shards: {len(shards)}", flush=True)

    all_tensors = []
    t0 = time.monotonic()
    for i, shard in enumerate(shards):
        sz = shard.stat().st_size / (1024**3)
        print(f"\n  [{i+1}/{len(shards)}] {shard.name} ({sz:.1f} GB)...", flush=True)
        results = analyze_shard(shard)
        all_tensors.extend(results)
        print(f"    {len(results)} tensors analyzed", flush=True)

    elapsed = time.monotonic() - t0
    print(f"\nTotal: {len(all_tensors)} tensors in {elapsed:.0f}s", flush=True)

    # Allocate bits
    print(f"\nAllocating bits (budget={args.budget} BPW)...", flush=True)
    allocation = allocate_bits(all_tensors, args.budget)

    # Summary
    from collections import Counter
    quant_dist = Counter(a["quant"] for a in allocation)
    print(f"\nAllocation distribution:", flush=True)
    for q, c in sorted(quant_dist.items(), key=lambda x: -QUANT_TYPES.get(x[0], 0)):
        print(f"  {q:12s} {c:4d} tensors", flush=True)

    total_bits = sum(a["bpw"] * t["elements"] for a, t in zip(allocation, all_tensors))
    total_elements = sum(t["elements"] for t in all_tensors)
    actual_bpw = total_bits / total_elements
    estimated_gb = total_bits / 8 / (1024**3)
    print(f"\nEstimated: {actual_bpw:.2f} BPW, ~{estimated_gb:.2f} GB", flush=True)

    # Generate llama-quantize overrides
    print(f"\n=== llama-quantize tensor-type overrides ===", flush=True)
    # Group by quant type and generate regex patterns
    by_quant = defaultdict(list)
    for a in allocation:
        gguf_name = to_gguf_name(a["name"])
        by_quant[a["quant"]].append(gguf_name)

    overrides = []
    for q, names in sorted(by_quant.items()):
        if q == "IQ3_S":
            continue  # Default, no override needed
        for n in names[:5]:  # Show first 5
            print(f'  --tensor-type "{n}={q}"', flush=True)
        if len(names) > 5:
            print(f"  ... (+{len(names)-5} more {q})", flush=True)
        overrides.extend([(n, q) for n in names])

    # Save
    output_path = args.output or (args.model_dir / "kurtboost_allocation.json")
    result = {
        "budget_bpw": args.budget,
        "actual_bpw": round(actual_bpw, 3),
        "estimated_gb": round(estimated_gb, 2),
        "n_tensors": len(all_tensors),
        "distribution": dict(quant_dist),
        "overrides": overrides,
        "tensors": [{**t, **a} for t, a in zip(all_tensors, allocation)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
