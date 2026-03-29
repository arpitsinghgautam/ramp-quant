#!/usr/bin/env python3
"""
RAMP-Local: GGUF Analyzer
Parses GGUF metadata, enumerates tensors, groups by layer/role,
computes byte sizes for any quant type configuration.

Reuses GGUF parsing patterns from analyze_expert_similarity.py.
"""

import struct
import mmap
import os
import json
import re
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# GGUF constants
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF" LE
GGUF_VERSION = 3

# Quantization type IDs from ggml-common.h
GGML_TYPE_F32    = 0
GGML_TYPE_F16    = 1
GGML_TYPE_Q4_0   = 2
GGML_TYPE_Q4_1   = 3
GGML_TYPE_Q5_0   = 6
GGML_TYPE_Q5_1   = 7
GGML_TYPE_Q8_0   = 8
GGML_TYPE_Q8_1   = 9
GGML_TYPE_Q2_K   = 10
GGML_TYPE_Q3_K   = 11
GGML_TYPE_Q4_K   = 12
GGML_TYPE_Q5_K   = 13
GGML_TYPE_Q6_K   = 14
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS  = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S   = 19
GGML_TYPE_IQ4_NL  = 20
GGML_TYPE_IQ3_S   = 21
GGML_TYPE_IQ2_S   = 22
GGML_TYPE_IQ4_XS  = 23
GGML_TYPE_IQ1_M   = 29
GGML_TYPE_BF16    = 30

# Quantization type name <-> ID mapping
QTYPE_NAME_TO_ID = {
    "F32": GGML_TYPE_F32, "F16": GGML_TYPE_F16, "BF16": GGML_TYPE_BF16,
    "Q4_0": GGML_TYPE_Q4_0, "Q4_1": GGML_TYPE_Q4_1,
    "Q5_0": GGML_TYPE_Q5_0, "Q5_1": GGML_TYPE_Q5_1,
    "Q8_0": GGML_TYPE_Q8_0, "Q8_1": GGML_TYPE_Q8_1,
    "Q2_K": GGML_TYPE_Q2_K, "Q3_K": GGML_TYPE_Q3_K,
    "Q4_K": GGML_TYPE_Q4_K, "Q5_K": GGML_TYPE_Q5_K, "Q6_K": GGML_TYPE_Q6_K,
    "IQ2_XXS": GGML_TYPE_IQ2_XXS, "IQ2_XS": GGML_TYPE_IQ2_XS,
    "IQ2_S": GGML_TYPE_IQ2_S,
    "IQ3_XXS": GGML_TYPE_IQ3_XXS, "IQ3_S": GGML_TYPE_IQ3_S,
    "IQ4_NL": GGML_TYPE_IQ4_NL, "IQ4_XS": GGML_TYPE_IQ4_XS,
    "IQ1_S": GGML_TYPE_IQ1_S, "IQ1_M": GGML_TYPE_IQ1_M,
}
QTYPE_ID_TO_NAME = {v: k for k, v in QTYPE_NAME_TO_ID.items()}

# Block sizes (elements per quantization block)
BLOCK_SIZES = {
    GGML_TYPE_F32: 1, GGML_TYPE_F16: 1, GGML_TYPE_BF16: 1,
    GGML_TYPE_Q4_0: 32, GGML_TYPE_Q4_1: 32,
    GGML_TYPE_Q5_0: 32, GGML_TYPE_Q5_1: 32,
    GGML_TYPE_Q8_0: 32, GGML_TYPE_Q8_1: 32 + 4,  # 32 int8 + 4 byte sum
    GGML_TYPE_Q2_K: 256, GGML_TYPE_Q3_K: 256,
    GGML_TYPE_Q4_K: 256, GGML_TYPE_Q5_K: 256, GGML_TYPE_Q6_K: 256,
    GGML_TYPE_IQ2_XXS: 256, GGML_TYPE_IQ2_XS: 256, GGML_TYPE_IQ2_S: 256,
    GGML_TYPE_IQ3_XXS: 256, GGML_TYPE_IQ3_S: 256,
    GGML_TYPE_IQ4_NL: 32, GGML_TYPE_IQ4_XS: 256,
    GGML_TYPE_IQ1_S: 256, GGML_TYPE_IQ1_M: 256,
}

# Bytes per block
BLOCK_BYTES = {
    GGML_TYPE_F32: 4, GGML_TYPE_F16: 2, GGML_TYPE_BF16: 2,
    GGML_TYPE_Q4_0: 18,    # 32 * 0.5 + 2
    GGML_TYPE_Q4_1: 20,    # 32 * 0.5 + 2 + 2
    GGML_TYPE_Q5_0: 22,    # 32 * 0.625 + 2
    GGML_TYPE_Q5_1: 24,
    GGML_TYPE_Q8_0: 34,    # 32 * 1 + 2
    GGML_TYPE_Q8_1: 40,
    GGML_TYPE_Q2_K: 84,    # 256 elems
    GGML_TYPE_Q3_K: 110,
    GGML_TYPE_Q4_K: 144,
    GGML_TYPE_Q5_K: 176,
    GGML_TYPE_Q6_K: 210,
    GGML_TYPE_IQ2_XXS: 66,  # 256 elems
    GGML_TYPE_IQ2_XS: 74,
    GGML_TYPE_IQ2_S: 82,
    GGML_TYPE_IQ3_XXS: 98,
    GGML_TYPE_IQ3_S: 110,
    GGML_TYPE_IQ4_NL: 18,   # 32 elems
    GGML_TYPE_IQ4_XS: 136,  # 256 elems
    GGML_TYPE_IQ1_S: 50,
    GGML_TYPE_IQ1_M: 56,
}

# Bits per weight (for display / BPW calculation)
BPW = {name: (BLOCK_BYTES[qid] * 8.0) / BLOCK_SIZES[qid]
       for name, qid in QTYPE_NAME_TO_ID.items()
       if qid in BLOCK_BYTES and qid in BLOCK_SIZES}

# GGUF candidate quant types for search (ordered by BPW ascending)
SEARCH_QUANT_TYPES = ["IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S",
                      "Q4_K", "Q5_K", "Q6_K", "Q8_0"]

# GGUF metadata value types
GGUF_VTYPE_UINT8   = 0
GGUF_VTYPE_INT8    = 1
GGUF_VTYPE_UINT16  = 2
GGUF_VTYPE_INT16   = 3
GGUF_VTYPE_UINT32  = 4
GGUF_VTYPE_INT32   = 5
GGUF_VTYPE_FLOAT32 = 6
GGUF_VTYPE_BOOL    = 7
GGUF_VTYPE_STRING  = 8
GGUF_VTYPE_ARRAY   = 9
GGUF_VTYPE_UINT64  = 10
GGUF_VTYPE_INT64   = 11
GGUF_VTYPE_FLOAT64 = 12


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TensorInfo:
    name: str
    shape: tuple
    n_elements: int
    qtype_id: int
    qtype_name: str
    offset: int        # byte offset in file (from data start)
    byte_size: int     # actual byte size in current GGUF
    layer_idx: int = -1
    role: str = "unknown"  # attn_q, attn_k, attn_v, attn_o, shared_gate,
                           # shared_up, shared_down, expert_gate, expert_up,
                           # expert_down, gate, norm, embed, output

    @property
    def bpw(self) -> float:
        return (self.byte_size * 8.0) / self.n_elements if self.n_elements > 0 else 0.0


@dataclass
class DecisionGroup:
    """A group of tensors that share the same quantization decision."""
    name: str           # e.g., "layer.5.attn", "layer.5.experts"
    tensor_names: list  # list of tensor names in this group
    total_elements: int
    role: str
    layer_idx: int

    def byte_size_for_qtype(self, qtype_name: str) -> int:
        """Compute total byte size if all tensors in group use given qtype."""
        qid = QTYPE_NAME_TO_ID[qtype_name]
        bs = BLOCK_SIZES[qid]
        bb = BLOCK_BYTES[qid]
        return sum(
            (nel + bs - 1) // bs * bb
            for nel in [self.total_elements]  # simplified: use total
        )

    def byte_size_for_qtype_exact(self, qtype_name: str,
                                   tensors: dict) -> int:
        """Exact byte size using per-tensor element counts."""
        qid = QTYPE_NAME_TO_ID[qtype_name]
        bs = BLOCK_SIZES[qid]
        bb = BLOCK_BYTES[qid]
        total = 0
        for tname in self.tensor_names:
            nel = tensors[tname].n_elements
            total += (nel + bs - 1) // bs * bb
        return total


# ---------------------------------------------------------------------------
# GGUF Parser
# ---------------------------------------------------------------------------

def _read_string(buf, offset):
    length = struct.unpack_from('<Q', buf, offset)[0]
    offset += 8
    s = buf[offset:offset + length].decode('utf-8', errors='replace')
    return s, offset + length


def _skip_value(buf, offset, vtype):
    if vtype in (GGUF_VTYPE_UINT8, GGUF_VTYPE_INT8, GGUF_VTYPE_BOOL):
        return offset + 1
    elif vtype in (GGUF_VTYPE_UINT16, GGUF_VTYPE_INT16):
        return offset + 2
    elif vtype in (GGUF_VTYPE_UINT32, GGUF_VTYPE_INT32, GGUF_VTYPE_FLOAT32):
        return offset + 4
    elif vtype in (GGUF_VTYPE_UINT64, GGUF_VTYPE_INT64, GGUF_VTYPE_FLOAT64):
        return offset + 8
    elif vtype == GGUF_VTYPE_STRING:
        _, new_off = _read_string(buf, offset)
        return new_off
    elif vtype == GGUF_VTYPE_ARRAY:
        elem_type = struct.unpack_from('<I', buf, offset)[0]
        offset += 4
        n_elems = struct.unpack_from('<Q', buf, offset)[0]
        offset += 8
        for _ in range(n_elems):
            offset = _skip_value(buf, offset, elem_type)
        return offset
    raise ValueError(f"Unknown GGUF value type: {vtype}")


def _read_value(buf, offset, vtype):
    """Read a metadata value, return (value, new_offset)."""
    if vtype == GGUF_VTYPE_UINT32:
        return struct.unpack_from('<I', buf, offset)[0], offset + 4
    elif vtype == GGUF_VTYPE_INT32:
        return struct.unpack_from('<i', buf, offset)[0], offset + 4
    elif vtype == GGUF_VTYPE_FLOAT32:
        return struct.unpack_from('<f', buf, offset)[0], offset + 4
    elif vtype == GGUF_VTYPE_UINT64:
        return struct.unpack_from('<Q', buf, offset)[0], offset + 8
    elif vtype == GGUF_VTYPE_STRING:
        return _read_string(buf, offset)
    else:
        # Skip unsupported types
        return None, _skip_value(buf, offset, vtype)


# ---------------------------------------------------------------------------
# Tensor role classification
# ---------------------------------------------------------------------------

# Regex patterns for Qwen3.5-35B-A3B tensor names
ROLE_PATTERNS = [
    # Full attention layers (10/40)
    (re.compile(r'blk\.(\d+)\.attn_q\.weight'), 'attn_q'),
    (re.compile(r'blk\.(\d+)\.attn_k\.weight'), 'attn_k'),
    (re.compile(r'blk\.(\d+)\.attn_v\.weight'), 'attn_v'),
    (re.compile(r'blk\.(\d+)\.attn_output\.weight'), 'attn_o'),
    # GDN (Gated DeltaNet) layers (30/40) — fused QKV + gate + SSM
    (re.compile(r'blk\.(\d+)\.attn_qkv\.weight'), 'gdn_qkv'),
    (re.compile(r'blk\.(\d+)\.attn_gate\.weight'), 'gdn_gate'),
    (re.compile(r'blk\.(\d+)\.attn_q_norm\.weight'), 'gdn_q_norm'),
    (re.compile(r'blk\.(\d+)\.attn_k_norm\.weight'), 'gdn_k_norm'),
    (re.compile(r'blk\.(\d+)\.ssm_a'), 'ssm_a'),
    (re.compile(r'blk\.(\d+)\.ssm_alpha\.weight'), 'ssm_alpha'),
    (re.compile(r'blk\.(\d+)\.ssm_beta\.weight'), 'ssm_beta'),
    (re.compile(r'blk\.(\d+)\.ssm_conv1d\.weight'), 'ssm_conv'),
    (re.compile(r'blk\.(\d+)\.ssm_dt\.bias'), 'ssm_dt'),
    (re.compile(r'blk\.(\d+)\.ssm_norm\.weight'), 'ssm_norm'),
    (re.compile(r'blk\.(\d+)\.ssm_out\.weight'), 'ssm_out'),
    (re.compile(r'blk\.(\d+)\.post_attention_norm\.weight'), 'post_attn_norm'),
    # MoE FFN
    (re.compile(r'blk\.(\d+)\.ffn_gate\.weight'), 'shared_gate'),
    (re.compile(r'blk\.(\d+)\.ffn_up\.weight'), 'shared_up'),
    (re.compile(r'blk\.(\d+)\.ffn_down\.weight'), 'shared_down'),
    (re.compile(r'blk\.(\d+)\.ffn_gate_shexp\.weight'), 'shared_gate'),
    (re.compile(r'blk\.(\d+)\.ffn_up_shexp\.weight'), 'shared_up'),
    (re.compile(r'blk\.(\d+)\.ffn_down_shexp\.weight'), 'shared_down'),
    (re.compile(r'blk\.(\d+)\.ffn_gate_exps\.weight'), 'expert_gate'),
    (re.compile(r'blk\.(\d+)\.ffn_up_exps\.weight'), 'expert_up'),
    (re.compile(r'blk\.(\d+)\.ffn_down_exps\.weight'), 'expert_down'),
    (re.compile(r'blk\.(\d+)\.ffn_gate_inp\.weight'), 'moe_gate'),
    (re.compile(r'blk\.(\d+)\.ffn_gate_inp_shexp\.weight'), 'shared_expert_gate'),
    # Norms
    (re.compile(r'blk\.(\d+)\.attn_norm\.weight'), 'attn_norm'),
    (re.compile(r'blk\.(\d+)\.ffn_norm\.weight'), 'ffn_norm'),
    # Global
    (re.compile(r'output_norm\.weight'), 'output_norm'),
    (re.compile(r'output\.weight'), 'output'),
    (re.compile(r'token_embd\.weight'), 'embed'),
]


def classify_tensor(name: str) -> tuple:
    """Return (layer_idx, role) for a tensor name."""
    for pattern, role in ROLE_PATTERNS:
        m = pattern.match(name)
        if m:
            layer_idx = int(m.group(1)) if m.lastindex and m.lastindex >= 1 else -1
            return layer_idx, role
    return -1, "unknown"


def decision_group_for_tensor(layer_idx: int, role: str) -> str:
    """Map tensor to its decision group name."""
    if layer_idx < 0:
        # Global tensors: embed, output, output_norm
        return f"global.{role}"

    if role.startswith("attn_"):
        return f"layer.{layer_idx}.attn"
    elif role in ("gdn_qkv", "gdn_gate"):
        return f"layer.{layer_idx}.gdn"
    elif role in ("gdn_q_norm", "gdn_k_norm"):
        return f"layer.{layer_idx}.norms"  # small norms → norms group
    elif role.startswith("ssm_"):
        return f"layer.{layer_idx}.ssm"
    elif role == "post_attn_norm":
        return f"layer.{layer_idx}.norms"
    elif role.startswith("shared_"):
        return f"layer.{layer_idx}.shared"
    elif role.startswith("expert_"):
        return f"layer.{layer_idx}.experts"
    elif role in ("moe_gate", "shared_expert_gate"):
        return f"layer.{layer_idx}.gates"
    elif role.endswith("_norm"):
        return f"layer.{layer_idx}.norms"
    else:
        return f"layer.{layer_idx}.{role}"


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class GGUFAnalyzer:
    """Parse GGUF file, enumerate tensors, build decision groups."""

    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        self.file_size = os.path.getsize(gguf_path)
        self.tensors: dict[str, TensorInfo] = {}
        self.groups: dict[str, DecisionGroup] = {}
        self.metadata: dict = {}
        self.n_layers: int = 0
        self._data_offset: int = 0  # byte offset where tensor data begins

        self._parse()
        self._build_groups()

    def _parse(self):
        """Parse GGUF header: metadata + tensor info entries."""
        with open(self.gguf_path, 'rb') as f:
            buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Header
            magic = struct.unpack_from('<I', buf, 0)[0]
            assert magic == GGUF_MAGIC, f"Not a GGUF file (magic={magic:#x})"
            version = struct.unpack_from('<I', buf, 4)[0]
            n_tensors = struct.unpack_from('<Q', buf, 8)[0]
            n_kv = struct.unpack_from('<Q', buf, 16)[0]

            offset = 24

            # Read metadata key-value pairs
            for _ in range(n_kv):
                key, offset = _read_string(buf, offset)
                vtype = struct.unpack_from('<I', buf, offset)[0]
                offset += 4
                val, offset = _read_value(buf, offset, vtype)
                if val is not None:
                    self.metadata[key] = val

            # Read tensor info entries
            tensor_infos_raw = []
            for _ in range(n_tensors):
                name, offset = _read_string(buf, offset)
                n_dims = struct.unpack_from('<I', buf, offset)[0]
                offset += 4
                dims = []
                for _ in range(n_dims):
                    dims.append(struct.unpack_from('<Q', buf, offset)[0])
                    offset += 8
                qtype = struct.unpack_from('<I', buf, offset)[0]
                offset += 4
                data_offset = struct.unpack_from('<Q', buf, offset)[0]
                offset += 8

                shape = tuple(dims) if dims else (1,)
                n_elements = 1
                for d in shape:
                    n_elements *= d

                qtype_name = QTYPE_ID_TO_NAME.get(qtype, f"UNKNOWN_{qtype}")
                bs = BLOCK_SIZES.get(qtype, 1)
                bb = BLOCK_BYTES.get(qtype, 1)
                byte_size = (n_elements + bs - 1) // bs * bb

                layer_idx, role = classify_tensor(name)

                ti = TensorInfo(
                    name=name, shape=shape, n_elements=n_elements,
                    qtype_id=qtype, qtype_name=qtype_name,
                    offset=data_offset, byte_size=byte_size,
                    layer_idx=layer_idx, role=role,
                )
                self.tensors[name] = ti

            # Align data offset
            self._data_offset = (offset + 31) & ~31

            buf.close()

        # Extract layer count from metadata
        self.n_layers = self.metadata.get("llama.block_count",
                        self.metadata.get("qwen2moe.block_count", 0))
        if self.n_layers == 0:
            # Infer from tensor names
            max_layer = max((t.layer_idx for t in self.tensors.values()
                            if t.layer_idx >= 0), default=-1)
            self.n_layers = max_layer + 1

    def _build_groups(self):
        """Build decision groups from tensor classification."""
        group_map: dict[str, list[str]] = {}
        for name, ti in self.tensors.items():
            gname = decision_group_for_tensor(ti.layer_idx, ti.role)
            if gname not in group_map:
                group_map[gname] = []
            group_map[gname].append(name)

        for gname, tensor_names in group_map.items():
            total_el = sum(self.tensors[t].n_elements for t in tensor_names)
            # Determine role and layer from first tensor
            first = self.tensors[tensor_names[0]]
            self.groups[gname] = DecisionGroup(
                name=gname,
                tensor_names=tensor_names,
                total_elements=total_el,
                role=first.role,
                layer_idx=first.layer_idx,
            )

    def list_decision_groups(self) -> list[str]:
        """Return sorted list of decision group names."""
        return sorted(self.groups.keys())

    def group_byte_size(self, group_name: str, qtype_name: str) -> int:
        """Compute byte size for a group at given quant type."""
        group = self.groups[group_name]
        return group.byte_size_for_qtype_exact(qtype_name, self.tensors)

    def total_size_for_config(self, config: dict[str, str]) -> int:
        """Compute total byte size for a full configuration.
        config: {group_name -> qtype_name}
        """
        total = 0
        for gname, group in self.groups.items():
            qtype = config.get(gname, "IQ3_S")  # default fallback
            total += group.byte_size_for_qtype_exact(qtype, self.tensors)
        return total

    def current_config(self) -> dict[str, str]:
        """Extract current quant type configuration from GGUF."""
        config = {}
        for gname, group in self.groups.items():
            # Use majority quant type in group
            type_counts = {}
            for tname in group.tensor_names:
                qt = self.tensors[tname].qtype_name
                type_counts[qt] = type_counts.get(qt, 0) + 1
            config[gname] = max(type_counts, key=type_counts.get)
        return config

    def summary(self) -> str:
        """Print summary of GGUF structure."""
        lines = [
            f"GGUF: {self.gguf_path}",
            f"File size: {self.file_size / (1024**3):.2f} GB",
            f"Tensors: {len(self.tensors)}",
            f"Layers: {self.n_layers}",
            f"Decision groups: {len(self.groups)}",
            "",
            "Groups by type:",
        ]

        # Summarize by group pattern
        pattern_stats = {}
        for gname, group in sorted(self.groups.items()):
            # Extract pattern: "layer.X.attn" -> "layer.*.attn"
            pattern = re.sub(r'\.\d+\.', '.*.', gname)
            if pattern not in pattern_stats:
                pattern_stats[pattern] = {"count": 0, "elements": 0,
                                          "tensors": 0}
            pattern_stats[pattern]["count"] += 1
            pattern_stats[pattern]["elements"] += group.total_elements
            pattern_stats[pattern]["tensors"] += len(group.tensor_names)

        for pattern, stats in sorted(pattern_stats.items()):
            el_m = stats["elements"] / 1e6
            lines.append(f"  {pattern}: {stats['count']} groups, "
                        f"{stats['tensors']} tensors, {el_m:.1f}M elements")

        return "\n".join(lines)

    def read_tensor_data(self, tensor_name: str) -> bytes:
        """Read raw bytes for a tensor from GGUF file."""
        ti = self.tensors[tensor_name]
        with open(self.gguf_path, 'rb') as f:
            f.seek(self._data_offset + ti.offset)
            return f.read(ti.byte_size)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: gguf_analyzer.py <gguf_path> [--json]")
        sys.exit(1)

    gguf_path = sys.argv[1]
    analyzer = GGUFAnalyzer(gguf_path)
    print(analyzer.summary())

    if "--json" in sys.argv:
        config = analyzer.current_config()
        print("\nCurrent configuration:")
        print(json.dumps(config, indent=2))

    if "--groups" in sys.argv:
        print("\nAll decision groups:")
        for gname in analyzer.list_decision_groups():
            g = analyzer.groups[gname]
            current_qt = analyzer.current_config().get(gname, "?")
            size_mb = g.byte_size_for_qtype_exact(current_qt, analyzer.tensors) / (1024**2)
            print(f"  {gname}: {len(g.tensor_names)} tensors, "
                  f"{g.total_elements/1e6:.1f}M elem, "
                  f"{current_qt} ({size_mb:.1f} MB)")
