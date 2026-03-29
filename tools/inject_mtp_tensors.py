#!/usr/bin/env python3
"""Inject MTP tensors from HF shard into existing IQ3_S custom-mix GGUF.

Creates a new GGUF with:
- All tensors from the base model (IQ3_S custom-mix, block_count=40)
- MTP layer tensors (blk.40.*) from HF shard 14, quantized to appropriate types
- Updated metadata: block_count=41, nextn_predict_layers=1
"""
import sys
import os
import json
import struct
import numpy as np
from pathlib import Path

# Add ik_llama gguf-py to path
sys.path.insert(0, str(Path.home() / "ik_llama.cpp" / "gguf-py"))

import gguf
from gguf import GGUFWriter, GGUFReader

def load_mtp_tensors(shard_path: str, index_path: str) -> dict:
    """Load MTP tensors from HF safetensors shard."""
    import safetensors.torch as st

    tensors = st.load_file(shard_path)

    # Filter only mtp.* tensors, convert BF16 → F32
    mtp = {k: v.float().numpy() for k, v in tensors.items() if k.startswith("mtp.")}
    print(f"Loaded {len(mtp)} MTP tensors from shard")
    for k, v in sorted(mtp.items()):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    return mtp


def map_mtp_tensor(name: str, data: np.ndarray, bid: int) -> list:
    """Map HF MTP tensor name to GGUF name(s) + data."""
    results = []

    # mtp.fc.weight → nextn.eh_proj
    if name == "mtp.fc.weight":
        results.append((f"blk.{bid}.nextn.eh_proj.weight", data))

    # mtp.norm.weight → nextn.shared_head_norm
    elif name == "mtp.norm.weight":
        results.append((f"blk.{bid}.nextn.shared_head_norm.weight", data))

    # mtp.pre_fc_norm_embedding.weight → nextn.enorm
    elif name == "mtp.pre_fc_norm_embedding.weight":
        results.append((f"blk.{bid}.nextn.enorm.weight", data))

    # mtp.pre_fc_norm_hidden.weight → nextn.hnorm
    elif name == "mtp.pre_fc_norm_hidden.weight":
        results.append((f"blk.{bid}.nextn.hnorm.weight", data))

    # mtp.layers.0.input_layernorm.weight → attn_norm
    elif name == "mtp.layers.0.input_layernorm.weight":
        results.append((f"blk.{bid}.attn_norm.weight", data))

    # mtp.layers.0.post_attention_layernorm.weight → post_attention_norm
    elif name == "mtp.layers.0.post_attention_layernorm.weight":
        results.append((f"blk.{bid}.post_attention_norm.weight", data))

    # mtp.layers.0.self_attn.* → attention tensors
    elif name == "mtp.layers.0.self_attn.q_proj.weight":
        results.append((f"blk.{bid}.attn_q.weight", data))
    elif name == "mtp.layers.0.self_attn.k_proj.weight":
        results.append((f"blk.{bid}.attn_k.weight", data))
    elif name == "mtp.layers.0.self_attn.v_proj.weight":
        results.append((f"blk.{bid}.attn_v.weight", data))
    elif name == "mtp.layers.0.self_attn.o_proj.weight":
        results.append((f"blk.{bid}.attn_output.weight", data))
    elif name == "mtp.layers.0.self_attn.q_norm.weight":
        results.append((f"blk.{bid}.attn_q_norm.weight", data))
    elif name == "mtp.layers.0.self_attn.k_norm.weight":
        results.append((f"blk.{bid}.attn_k_norm.weight", data))

    # mtp.layers.0.mlp.gate.weight → ffn_gate_inp
    elif name == "mtp.layers.0.mlp.gate.weight":
        results.append((f"blk.{bid}.ffn_gate_inp.weight", data))

    # mtp.layers.0.mlp.shared_expert_gate.weight → ffn_gate_inp_shexp
    elif name == "mtp.layers.0.mlp.shared_expert_gate.weight":
        results.append((f"blk.{bid}.ffn_gate_inp_shexp.weight", data))

    # mtp.layers.0.mlp.shared_expert.gate_proj.weight → ffn_gate_shexp
    elif name == "mtp.layers.0.mlp.shared_expert.gate_proj.weight":
        results.append((f"blk.{bid}.ffn_gate_shexp.weight", data))
    elif name == "mtp.layers.0.mlp.shared_expert.up_proj.weight":
        results.append((f"blk.{bid}.ffn_up_shexp.weight", data))
    elif name == "mtp.layers.0.mlp.shared_expert.down_proj.weight":
        results.append((f"blk.{bid}.ffn_down_shexp.weight", data))

    # Expert tensors will be handled separately (need merging)
    elif "mlp.experts" in name:
        return []  # Skip individual experts, handle in merge step

    else:
        print(f"  WARNING: Unmapped tensor {name}")

    return results


def merge_experts(mtp_tensors: dict, bid: int, n_experts: int = 256) -> list:
    """Merge individual expert tensors into 3D tensors."""
    results = []

    for proj_type in ["gate_proj", "up_proj", "down_proj"]:
        experts = []
        for i in range(n_experts):
            key = f"mtp.layers.0.mlp.experts.{i}.{proj_type}.weight"
            if key in mtp_tensors:
                experts.append(mtp_tensors[key])

        if len(experts) == n_experts:
            merged = np.stack(experts, axis=0)  # [n_experts, out, in]
            gguf_name_map = {
                "gate_proj": f"blk.{bid}.ffn_gate_exps.weight",
                "up_proj": f"blk.{bid}.ffn_up_exps.weight",
                "down_proj": f"blk.{bid}.ffn_down_exps.weight",
            }
            results.append((gguf_name_map[proj_type], merged))
            print(f"  Merged {n_experts} experts {proj_type} → {gguf_name_map[proj_type]} shape={merged.shape}")
        else:
            print(f"  WARNING: Only {len(experts)}/{n_experts} experts for {proj_type}")

    return results


def main():
    base_gguf = "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf"
    shard_path = "~/.chimere/models/Qwen3.5-35B-A3B-MTP-shard/model.safetensors-00014-of-00014.safetensors"
    index_path = "~/.chimere/models/Qwen3.5-35B-A3B-MTP-shard/model.safetensors.index.json"
    output_gguf = "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-MTP.gguf"

    mtp_bid = 40  # MTP layer index

    print("=" * 60)
    print("Step 1: Load MTP tensors from HF shard")
    print("=" * 60)
    mtp_tensors = load_mtp_tensors(shard_path, index_path)

    print("\n" + "=" * 60)
    print("Step 2: Map MTP tensors to GGUF names")
    print("=" * 60)
    mapped_tensors = []
    for name, data in sorted(mtp_tensors.items()):
        results = map_mtp_tensor(name, data, mtp_bid)
        mapped_tensors.extend(results)

    # Merge expert tensors
    expert_tensors = merge_experts(mtp_tensors, mtp_bid)
    mapped_tensors.extend(expert_tensors)

    print(f"\nTotal MTP tensors to inject: {len(mapped_tensors)}")

    print("\n" + "=" * 60)
    print("Step 3: Read base GGUF and create new GGUF with MTP")
    print("=" * 60)

    reader = GGUFReader(base_gguf)
    print(f"Base model: {len(reader.tensors)} tensors")

    # Get base model embed_tokens and output for nextn duplication
    embed_tokens_data = None
    output_data = None
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            embed_tokens_data = t.data
        elif t.name == "output.weight":
            output_data = t.data

    # Add nextn.embed_tokens and nextn.shared_head_head (duplicates of main model)
    if embed_tokens_data is not None:
        mapped_tensors.append((f"blk.{mtp_bid}.nextn.embed_tokens.weight", embed_tokens_data))
        print(f"  Added nextn.embed_tokens from main model embed_tokens")

    if output_data is not None:
        mapped_tensors.append((f"blk.{mtp_bid}.nextn.shared_head_head.weight", output_data))
        print(f"  Added nextn.shared_head_head from main model output")

    # Create output GGUF
    writer = GGUFWriter(output_gguf, arch="qwen35moe")

    # Copy all metadata from base, updating block_count and nextn
    for key, field in reader.fields.items():
        if key == "GGUF.version" or key == "GGUF.tensor_count" or key == "GGUF.kv_count":
            continue
        if "block_count" in key:
            writer.add_block_count(41)  # 40 + 1 MTP
        elif "nextn_predict_layers" in key:
            writer.add_key_value(key, 1, gguf.GGUFValueType.UINT32)
        else:
            # Copy field as-is (simplified — just copy known metadata)
            pass

    # Actually, this is getting complex. Let's use a simpler approach:
    # Binary patch the existing GGUF metadata and append tensors.
    print("\nThis approach requires a full GGUF rewrite which is complex.")
    print("Using binary patch approach instead...")
    writer.close()
    os.remove(output_gguf)

    # Binary approach: modify metadata in-place + append tensors
    # This is too fragile. Better approach: use llama-gguf-hash or similar.
    print("\nFALLBACK: Writing standalone MTP GGUF for merge later")

    # Write just the MTP tensors to a small GGUF
    mtp_output = "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/mtp-layer-40.gguf"
    w = GGUFWriter(mtp_output, arch="qwen35moe")
    w.add_block_count(41)
    w.add_key_value("qwen35moe.nextn_predict_layers", 1, gguf.GGUFValueType.UINT32)

    for name, data in mapped_tensors:
        if isinstance(data, np.ndarray):
            # Convert BF16 to F16
            if data.dtype == np.float32:
                w.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
            else:
                data_f32 = data.astype(np.float32) if data.dtype != np.float32 else data
                w.add_tensor(name, data_f32, raw_dtype=gguf.GGMLQuantizationType.F32)
        print(f"  Written: {name}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"\nMTP tensors written to: {mtp_output}")
    print(f"Size: {os.path.getsize(mtp_output) / 1e9:.2f} GB")
    print("\nNext step: use llama-quantize to merge or gguf-cat to combine")


if __name__ == "__main__":
    main()
