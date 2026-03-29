#!/usr/bin/env python3
"""Patch existing IQ3_S GGUF to add MTP: binary approach.

Strategy:
1. Use gguf-py to create a NEW GGUF with ONLY the MTP tensors + updated metadata
2. Use a merge step that copies the base GGUF's KV data byte-for-byte
   but patches block_count and nextn_predict_layers

Actually, simplest approach:
- Read base GGUF entirely as bytes
- Find and patch block_count (40→41) and nextn_predict_layers (0→1)
- Reconstruct with GGUFWriter using proper API calls
- The key insight: use the base model's GGUFReader to get TOKEN data properly
"""
import sys, os, struct, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.home() / "ik_llama.cpp" / "gguf-py"))
import gguf

BASE = os.path.expanduser("~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf")
SHARD13 = os.path.expanduser("~/.chimere/models/Qwen3.5-35B-A3B-MTP-shard/model.safetensors-00013-of-00014.safetensors")
SHARD14 = os.path.expanduser("~/.chimere/models/Qwen3.5-35B-A3B-MTP-shard/model.safetensors-00014-of-00014.safetensors")
OUTPUT = os.path.expanduser("~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-MTP.gguf")
MTP_BID = 40


def load_mtp_tensors():
    from safetensors import safe_open
    all_tensors = {}
    for shard in [SHARD13, SHARD14]:
        with safe_open(shard, framework='pt') as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    all_tensors[k] = f.get_tensor(k).float().numpy()
    return all_tensors


def map_and_merge(raw: dict) -> list:
    bid = MTP_BID
    result = []
    simple = {
        "mtp.fc.weight": f"blk.{bid}.nextn.eh_proj.weight",
        "mtp.norm.weight": f"blk.{bid}.nextn.shared_head_norm.weight",
        "mtp.pre_fc_norm_embedding.weight": f"blk.{bid}.nextn.enorm.weight",
        "mtp.pre_fc_norm_hidden.weight": f"blk.{bid}.nextn.hnorm.weight",
        "mtp.layers.0.input_layernorm.weight": f"blk.{bid}.attn_norm.weight",
        "mtp.layers.0.post_attention_layernorm.weight": f"blk.{bid}.post_attention_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight": f"blk.{bid}.attn_q.weight",
        "mtp.layers.0.self_attn.k_proj.weight": f"blk.{bid}.attn_k.weight",
        "mtp.layers.0.self_attn.v_proj.weight": f"blk.{bid}.attn_v.weight",
        "mtp.layers.0.self_attn.o_proj.weight": f"blk.{bid}.attn_output.weight",
        "mtp.layers.0.self_attn.q_norm.weight": f"blk.{bid}.attn_q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight": f"blk.{bid}.attn_k_norm.weight",
        "mtp.layers.0.mlp.gate.weight": f"blk.{bid}.ffn_gate_inp.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight": f"blk.{bid}.ffn_gate_inp_shexp.weight",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": f"blk.{bid}.ffn_gate_shexp.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": f"blk.{bid}.ffn_up_shexp.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": f"blk.{bid}.ffn_down_shexp.weight",
    }
    for hf, gg in simple.items():
        if hf in raw:
            result.append((gg, raw[hf]))

    for proj, suffix in [("gate_proj","ffn_gate_exps"),("up_proj","ffn_up_exps"),("down_proj","ffn_down_exps")]:
        exps = [raw[f"mtp.layers.0.mlp.experts.{i}.{proj}.weight"] for i in range(256)
                if f"mtp.layers.0.mlp.experts.{i}.{proj}.weight" in raw]
        if len(exps) == 256:
            result.append((f"blk.{bid}.{suffix}.weight", np.stack(exps, axis=0)))
    return result


def main():
    print("[1] Loading MTP tensors...")
    raw = load_mtp_tensors()
    mapped = map_and_merge(raw)
    del raw
    print(f"  {len(mapped)} MTP tensors")

    print("[2] Reading base GGUF for binary token copy...")
    reader = gguf.GGUFReader(BASE)

    # Extract tokenizer data properly using the reader's raw binary access
    # The GGUFReader maps the file, so we can read the token arrays via field.parts
    # For string arrays: field.parts contains individual string bytes
    # field.data contains the indices into parts for each element

    # Tokens
    tf = reader.fields["tokenizer.ggml.tokens"]
    # For ARRAY of STRING: parts[0]=array_type, parts[1]=array_len, parts[2..]=string lengths+data
    # Actually field.data gives us the part indices for each string value
    tokens = []
    for idx in tf.data:
        raw_bytes = bytes(tf.parts[idx])
        tokens.append(raw_bytes.decode('utf-8', errors='replace'))
    print(f"  Tokens: {len(tokens)}")

    # Token types
    ttf = reader.fields["tokenizer.ggml.token_type"]
    token_types = [int(ttf.parts[idx][0]) for idx in ttf.data]
    print(f"  Token types: {len(token_types)}")

    # Merges
    mf = reader.fields["tokenizer.ggml.merges"]
    merges = []
    for idx in mf.data:
        raw_bytes = bytes(mf.parts[idx])
        merges.append(raw_bytes.decode('utf-8', errors='replace'))
    print(f"  Merges: {len(merges)}")

    # Get nextn vocab tensors from base
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            mapped.append((f"blk.{MTP_BID}.nextn.embed_tokens.weight", t.data, t.tensor_type))
        elif t.name == "output.weight":
            mapped.append((f"blk.{MTP_BID}.nextn.shared_head_head.weight", t.data, t.tensor_type))

    print(f"\n[3] Writing new GGUF...")
    writer = gguf.GGUFWriter(OUTPUT, arch="qwen35moe")

    # Copy scalar metadata (skip arrays and overrides)
    array_keys = {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
                  "qwen35moe.rope.dimension_sections"}
    skip_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}

    for key, field in reader.fields.items():
        if key in skip_keys or key in array_keys:
            continue
        if "block_count" in key:
            writer.add_block_count(41)
            continue
        if "nextn_predict_layers" in key:
            writer.add_uint32(key, 1)
            continue

        # Copy scalar values
        val_type = field.types[-1] if field.types else None
        try:
            if val_type == gguf.GGUFValueType.STRING:
                writer.add_key_value(key, bytes(field.parts[-1]).decode('utf-8'), val_type)
            elif val_type == gguf.GGUFValueType.UINT32:
                writer.add_key_value(key, int(field.parts[-1][0]), val_type)
            elif val_type == gguf.GGUFValueType.INT32:
                writer.add_key_value(key, int(field.parts[-1][0]), val_type)
            elif val_type == gguf.GGUFValueType.FLOAT32:
                writer.add_key_value(key, float(field.parts[-1][0]), val_type)
            elif val_type == gguf.GGUFValueType.BOOL:
                writer.add_key_value(key, bool(field.parts[-1][0]), val_type)
        except:
            pass

    # Write tokenizer arrays using proper API
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)

    # Write rope sections
    if "qwen35moe.rope.dimension_sections" in reader.fields:
        sf = reader.fields["qwen35moe.rope.dimension_sections"]
        sections = [int(sf.parts[idx][0]) for idx in sf.data]
        writer.add_rope_dimension_sections(sections)
        print(f"  Rope sections: {sections}")

    # Copy base tensors
    print(f"  Copying {len(reader.tensors)} base tensors...")
    for i, t in enumerate(reader.tensors):
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    # Add MTP tensors
    print(f"  Adding {len(mapped)} MTP tensors...")
    for item in mapped:
        if len(item) == 3:
            name, data, raw_type = item
            writer.add_tensor(name, data, raw_dtype=raw_type)
        else:
            name, data = item
            if isinstance(data, np.ndarray) and data.dtype == np.float32 and data.size > 100_000:
                writer.add_tensor(name, data.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
            else:
                writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
        print(f"    {name}")

    print("\n[4] Flushing...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nDONE: {OUTPUT} ({os.path.getsize(OUTPUT)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
