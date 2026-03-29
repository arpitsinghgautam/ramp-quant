#!/usr/bin/env python3
"""Build IQ3_S-MTP GGUF v3 — proper metadata copy including ARRAY fields."""
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
    """Load all MTP tensors from both shards."""
    from safetensors import safe_open
    tensors = {}

    for shard in [SHARD13, SHARD14]:
        if not os.path.exists(shard):
            print(f"  SKIP {shard} (not found)")
            continue
        with safe_open(shard, framework='pt') as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    tensors[k] = f.get_tensor(k).float().numpy()
        print(f"  Loaded from {os.path.basename(shard)}: {sum(1 for k in tensors if k.startswith('mtp.'))} MTP tensors")

    return tensors


def map_mtp(tensors: dict) -> list:
    """Map HF tensor names to GGUF names and merge experts."""
    bid = MTP_BID
    mapped = []
    simple_map = {
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

    for hf_name, gguf_name in simple_map.items():
        if hf_name in tensors:
            mapped.append((gguf_name, tensors[hf_name], None))

    # Merge 256 experts
    for proj, gguf_suffix in [("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")]:
        exps = [tensors[f"mtp.layers.0.mlp.experts.{i}.{proj}.weight"] for i in range(256) if f"mtp.layers.0.mlp.experts.{i}.{proj}.weight" in tensors]
        if len(exps) == 256:
            merged = np.stack(exps, axis=0)
            mapped.append((f"blk.{bid}.{gguf_suffix}.weight", merged, None))
            print(f"  Merged 256 experts {proj} → {merged.shape}")

    return mapped


def read_field_value(reader, key):
    """Extract a field value from GGUFReader in a format suitable for GGUFWriter."""
    field = reader.fields[key]
    val_type = field.types[-1] if field.types else None

    if val_type == gguf.GGUFValueType.STRING:
        return str(bytes(field.parts[-1]), 'utf-8'), val_type
    elif val_type == gguf.GGUFValueType.UINT32:
        return int(field.parts[-1][0]), val_type
    elif val_type == gguf.GGUFValueType.INT32:
        return int(field.parts[-1][0]), val_type
    elif val_type == gguf.GGUFValueType.FLOAT32:
        return float(field.parts[-1][0]), val_type
    elif val_type == gguf.GGUFValueType.BOOL:
        return bool(field.parts[-1][0]), val_type
    elif val_type == gguf.GGUFValueType.UINT64:
        return int(field.parts[-1][0]), val_type
    elif val_type == gguf.GGUFValueType.UINT8:
        return int(field.parts[-1][0]), val_type
    elif val_type == gguf.GGUFValueType.INT64:
        return int(field.parts[-1][0]), val_type
    elif val_type == gguf.GGUFValueType.FLOAT64:
        return float(field.parts[-1][0]), val_type
    else:
        return None, val_type


def main():
    print("=" * 60)
    print("[1/4] Loading MTP tensors")
    print("=" * 60)
    mtp_raw = load_mtp_tensors()
    mtp_mapped = map_mtp(mtp_raw)
    del mtp_raw  # Free ~4 GB
    print(f"  {len(mtp_mapped)} MTP tensors mapped")

    print("\n" + "=" * 60)
    print("[2/4] Reading base model")
    print("=" * 60)
    reader = gguf.GGUFReader(BASE)
    print(f"  {len(reader.tensors)} tensors")

    # Add nextn.embed_tokens and shared_head_head from base model (keep quantization type)
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            mtp_mapped.append((f"blk.{MTP_BID}.nextn.embed_tokens.weight", t.data, t.tensor_type))
        elif t.name == "output.weight":
            mtp_mapped.append((f"blk.{MTP_BID}.nextn.shared_head_head.weight", t.data, t.tensor_type))
    print(f"  Total MTP tensors (with nextn vocab): {len(mtp_mapped)}")

    print("\n" + "=" * 60)
    print("[3/4] Writing GGUF with proper metadata")
    print("=" * 60)

    writer = gguf.GGUFWriter(OUTPUT, arch="qwen35moe")

    # Copy ALL metadata properly
    skip_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in reader.fields.items():
        if key in skip_keys:
            continue

        # Override block_count and nextn
        if "block_count" in key:
            writer.add_block_count(41)
            continue
        if "nextn_predict_layers" in key:
            writer.add_uint32(key, 1)
            continue

        # Handle ARRAY fields specially
        is_array = any(t == gguf.GGUFValueType.ARRAY for t in field.types)

        if is_array:
            arr_type = field.types[1] if len(field.types) > 1 else None
            if key == "tokenizer.ggml.tokens":
                tokens = [str(bytes(field.parts[i]), 'utf-8', errors='replace') for i in range(len(field.data))]
                writer.add_token_list(tokens)
                print(f"  Copied {key}: {len(tokens)} tokens")
            elif key == "tokenizer.ggml.token_type":
                # field.data contains indices into field.parts
                types_arr = [int(field.parts[i][0]) for i in range(len(field.data))]
                writer.add_token_types(types_arr)
                print(f"  Copied {key}: {len(types_arr)} types")
            elif key == "tokenizer.ggml.merges":
                merges = [str(bytes(field.parts[i]), 'utf-8', errors='replace') for i in range(len(field.data))]
                writer.add_token_merges(merges)
                print(f"  Copied {key}: {len(merges)} merges")
            elif key == "qwen35moe.rope.dimension_sections":
                # Int32 array
                sections = [int(field.parts[i][0]) for i in range(len(field.data))]
                writer.add_key_value(key, sections, gguf.GGUFValueType.ARRAY)
                print(f"  Copied {key}: {sections}")
            else:
                print(f"  SKIP ARRAY: {key}")
            continue

        # Handle scalar fields
        val, val_type = read_field_value(reader, key)
        if val is not None:
            writer.add_key_value(key, val, val_type)

    # Copy all base tensors
    print(f"\n  Copying {len(reader.tensors)} base tensors...")
    for i, t in enumerate(reader.tensors):
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(reader.tensors)}")

    # Add MTP tensors
    print(f"\n  Adding {len(mtp_mapped)} MTP tensors...")
    for item in mtp_mapped:
        if len(item) == 3:
            name, data, raw_type = item
        else:
            name, data = item
            raw_type = None

        if raw_type is not None:
            # Quantized tensor from base model — pass raw_dtype
            writer.add_tensor(name, data, raw_dtype=raw_type)
        elif isinstance(data, np.ndarray) and data.dtype == np.float32 and data.size > 100_000:
            writer.add_tensor(name, data.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
        elif isinstance(data, np.ndarray) and data.dtype == np.float32:
            writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
        print(f"    {name}")

    print("\n" + "=" * 60)
    print("[4/4] Writing to disk")
    print("=" * 60)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = os.path.getsize(OUTPUT) / 1e9
    print(f"\nSUCCESS: {OUTPUT}")
    print(f"Size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
