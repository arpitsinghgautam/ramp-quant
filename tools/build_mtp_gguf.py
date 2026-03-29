#!/usr/bin/env python3
"""Build IQ3_S GGUF with MTP by copying base model and adding MTP tensors.

Reads the base custom-mix IQ3_S (block_count=40) and creates a new GGUF
with block_count=41 and MTP layer tensors from HF shard 14.
"""
import sys
import os
import struct
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ik_llama.cpp" / "gguf-py"))
import gguf

BASE = "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf"
SHARD = "~/.chimere/models/Qwen3.5-35B-A3B-MTP-shard/model.safetensors-00014-of-00014.safetensors"
OUTPUT = "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-MTP.gguf"
MTP_BID = 40


def load_and_map_mtp(shard_path: str) -> list:
    """Load MTP tensors from HF shard, map to GGUF names."""
    import safetensors.torch as st
    raw = st.load_file(shard_path)
    mtp = {k: v for k, v in raw.items() if k.startswith("mtp.")}
    print(f"Loaded {len(mtp)} MTP tensors")

    mapped = []
    experts = {}  # Collect for merging

    for name, tensor in sorted(mtp.items()):
        data = tensor.float().numpy()
        bid = MTP_BID

        if name == "mtp.fc.weight":
            mapped.append((f"blk.{bid}.nextn.eh_proj.weight", data))
        elif name == "mtp.norm.weight":
            mapped.append((f"blk.{bid}.nextn.shared_head_norm.weight", data))
        elif name == "mtp.pre_fc_norm_embedding.weight":
            mapped.append((f"blk.{bid}.nextn.enorm.weight", data))
        elif name == "mtp.pre_fc_norm_hidden.weight":
            mapped.append((f"blk.{bid}.nextn.hnorm.weight", data))
        elif name == "mtp.layers.0.input_layernorm.weight":
            mapped.append((f"blk.{bid}.attn_norm.weight", data))
        elif name == "mtp.layers.0.post_attention_layernorm.weight":
            mapped.append((f"blk.{bid}.post_attention_norm.weight", data))
        elif name == "mtp.layers.0.self_attn.q_proj.weight":
            mapped.append((f"blk.{bid}.attn_q.weight", data))
        elif name == "mtp.layers.0.self_attn.k_proj.weight":
            mapped.append((f"blk.{bid}.attn_k.weight", data))
        elif name == "mtp.layers.0.self_attn.v_proj.weight":
            mapped.append((f"blk.{bid}.attn_v.weight", data))
        elif name == "mtp.layers.0.self_attn.o_proj.weight":
            mapped.append((f"blk.{bid}.attn_output.weight", data))
        elif name == "mtp.layers.0.self_attn.q_norm.weight":
            mapped.append((f"blk.{bid}.attn_q_norm.weight", data))
        elif name == "mtp.layers.0.self_attn.k_norm.weight":
            mapped.append((f"blk.{bid}.attn_k_norm.weight", data))
        elif name == "mtp.layers.0.mlp.gate.weight":
            mapped.append((f"blk.{bid}.ffn_gate_inp.weight", data))
        elif name == "mtp.layers.0.mlp.shared_expert_gate.weight":
            mapped.append((f"blk.{bid}.ffn_gate_inp_shexp.weight", data))
        elif name == "mtp.layers.0.mlp.shared_expert.gate_proj.weight":
            mapped.append((f"blk.{bid}.ffn_gate_shexp.weight", data))
        elif name == "mtp.layers.0.mlp.shared_expert.up_proj.weight":
            mapped.append((f"blk.{bid}.ffn_up_shexp.weight", data))
        elif name == "mtp.layers.0.mlp.shared_expert.down_proj.weight":
            mapped.append((f"blk.{bid}.ffn_down_shexp.weight", data))
        elif "mlp.experts" in name:
            experts[name] = data
        # else: skip unknown

    # Merge 256 experts into 3D tensors
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        exp_list = []
        for i in range(256):
            k = f"mtp.layers.0.mlp.experts.{i}.{proj}.weight"
            if k in experts:
                exp_list.append(experts[k])
        if len(exp_list) == 256:
            merged = np.stack(exp_list, axis=0)
            gguf_map = {"gate_proj": "ffn_gate_exps", "up_proj": "ffn_up_exps", "down_proj": "ffn_down_exps"}
            mapped.append((f"blk.{bid}.{gguf_map[proj]}.weight", merged))
            print(f"  Merged experts {proj} → shape {merged.shape}")
            del exp_list  # Free memory

    return mapped


def main():
    print("=" * 60)
    print("Building IQ3_S-MTP GGUF")
    print("=" * 60)

    # Step 1: Load MTP tensors
    print("\n[1/3] Loading MTP tensors from HF shard...")
    mtp_tensors = load_and_map_mtp(SHARD)
    print(f"  {len(mtp_tensors)} mapped tensors")

    # Step 2: Read base model
    print("\n[2/3] Reading base model...")
    reader = gguf.GGUFReader(BASE)
    print(f"  {len(reader.tensors)} tensors, arch=qwen35moe")

    # Get embed_tokens and output for nextn duplication
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            mtp_tensors.append((f"blk.{MTP_BID}.nextn.embed_tokens.weight", t.data))
            print(f"  Added nextn.embed_tokens (duplicated from token_embd)")
        elif t.name == "output.weight":
            mtp_tensors.append((f"blk.{MTP_BID}.nextn.shared_head_head.weight", t.data))
            print(f"  Added nextn.shared_head_head (duplicated from output)")

    # Step 3: Write new GGUF
    print(f"\n[3/3] Writing {OUTPUT}...")
    print(f"  Total tensors: {len(reader.tensors)} base + {len(mtp_tensors)} MTP")

    # Open output for binary writing
    with open(OUTPUT, "wb") as fout:
        # We'll write a complete new GGUF by:
        # 1. Copying the base file entirely
        # 2. Using gguf-py to patch metadata and append tensors

        # Actually, the simplest approach: use GGUFWriter to create from scratch
        pass

    os.remove(OUTPUT)  # Remove empty file

    # Use GGUFWriter
    writer = gguf.GGUFWriter(OUTPUT, arch="qwen35moe")

    # Copy metadata from base, patching block_count and nextn
    print("  Writing metadata...")
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue

        if "block_count" in key:
            writer.add_block_count(41)
        elif "nextn_predict_layers" in key:
            writer.add_uint32(key, 1)
        else:
            # Copy raw KV data
            # field.types[0] is the value type
            try:
                val_type = field.types[-1] if field.types else None
                if val_type == gguf.GGUFValueType.STRING:
                    writer.add_key_value(key, str(bytes(field.parts[-1]), 'utf-8'), val_type)
                elif val_type == gguf.GGUFValueType.UINT32:
                    writer.add_key_value(key, int(field.parts[-1][0]), val_type)
                elif val_type == gguf.GGUFValueType.INT32:
                    writer.add_key_value(key, int(field.parts[-1][0]), val_type)
                elif val_type == gguf.GGUFValueType.FLOAT32:
                    writer.add_key_value(key, float(field.parts[-1][0]), val_type)
                elif val_type == gguf.GGUFValueType.BOOL:
                    writer.add_key_value(key, bool(field.parts[-1][0]), val_type)
                elif val_type == gguf.GGUFValueType.UINT64:
                    writer.add_key_value(key, int(field.parts[-1][0]), val_type)
                elif val_type == gguf.GGUFValueType.ARRAY:
                    # Skip arrays (complex) - handle known ones explicitly
                    pass
                else:
                    pass
            except Exception as e:
                pass  # Skip problematic metadata

    # Copy all tensors from base model
    print("  Copying base tensors...")
    for i, t in enumerate(reader.tensors):
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(reader.tensors)}")

    # Add MTP tensors
    print(f"  Adding {len(mtp_tensors)} MTP tensors...")
    for name, data in mtp_tensors:
        if isinstance(data, np.ndarray) and data.dtype == np.float32:
            # Store small tensors as F32, large ones as F16
            if data.size > 1_000_000:
                data_f16 = data.astype(np.float16)
                writer.add_tensor(name, data_f16, raw_dtype=gguf.GGMLQuantizationType.F16)
            else:
                writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            # Already quantized data from base model (for nextn duplicates)
            writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F16)
        print(f"    {name}: {data.shape if hasattr(data, 'shape') else 'raw'}")

    print("  Writing to disk...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = os.path.getsize(OUTPUT) / 1e9
    print(f"\nDone! Output: {OUTPUT}")
    print(f"Size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
