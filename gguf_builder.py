#!/usr/bin/env python3
"""
RAMP-Local: GGUF Builder

Converts a search result (group_name -> quant_type mapping) into a
llama-quantize command with --tensor-type overrides, ready to execute.

Also supports generating a human-readable summary of the mixed-precision plan.
"""

import json
import re
import os
from typing import Optional

from gguf_analyzer import (GGUFAnalyzer, SEARCH_QUANT_TYPES, BPW,
                           QTYPE_NAME_TO_ID)


# ---------------------------------------------------------------------------
# Group name -> llama-quantize regex pattern mapping
# ---------------------------------------------------------------------------

# Qwen3.5-35B-A3B GGUF tensor name patterns:
#   blk.{N}.attn_q.weight
#   blk.{N}.attn_k.weight
#   blk.{N}.attn_v.weight
#   blk.{N}.attn_output.weight
#   blk.{N}.ffn_gate_shexp.weight       (shared expert)
#   blk.{N}.ffn_up_shexp.weight
#   blk.{N}.ffn_down_shexp.weight
#   blk.{N}.ffn_gate_exps.weight         (routed experts, 3D tensor)
#   blk.{N}.ffn_up_exps.weight
#   blk.{N}.ffn_down_exps.weight
#   blk.{N}.ffn_gate_inp.weight          (MoE router)
#   blk.{N}.ffn_gate_inp_shexp.weight    (shared expert scalar gate)
#   blk.{N}.attn_norm.weight
#   blk.{N}.ffn_norm.weight
#   output_norm.weight
#   output.weight
#   token_embd.weight


def group_to_regex(group_name: str) -> str:
    """Convert a decision group name to a llama-quantize --tensor-type regex.

    Examples:
        "layer.5.attn"    -> "blk\\.5\\.attn_[qkvo]"
        "layer.5.shared"  -> "blk\\.5\\.ffn_(gate|up|down)_shexp"
        "layer.5.experts" -> "blk\\.5\\.ffn_(gate|up|down)_exps"
        "layer.5.norms"   -> "blk\\.5\\.(attn|ffn)_norm"
        "layer.5.gates"   -> "blk\\.5\\.ffn_gate_inp"
        "global.embed"    -> "token_embd"
        "global.output"   -> "output\\.weight"
    """
    parts = group_name.split(".")

    if parts[0] == "global":
        role = parts[1]
        if role == "embed":
            return "token_embd"
        elif role == "output":
            return "^output\\.weight$"
        elif role == "output_norm":
            return "output_norm"
        else:
            return role

    elif parts[0] == "layer":
        layer_idx = parts[1]
        role = parts[2]

        if role == "attn":
            return f"blk\\.{layer_idx}\\.attn_(q|k|v|output)\\.weight"
        elif role == "gdn":
            return f"blk\\.{layer_idx}\\.(attn_qkv|attn_gate|attn_q_norm|attn_k_norm)"
        elif role == "ssm":
            return f"blk\\.{layer_idx}\\.ssm_"
        elif role == "shared":
            return f"blk\\.{layer_idx}\\.ffn_(gate|up|down|gate_up)_shexp"
        elif role == "experts":
            # Match both separate (ffn_gate_exps, ffn_up_exps, ffn_down_exps)
            # and fused (ffn_gate_up_exps) tensor layouts
            return f"blk\\.{layer_idx}\\.ffn_(gate|up|down|gate_up)_exps"
        elif role == "norms":
            return f"blk\\.{layer_idx}\\.(attn|ffn|post_attention)_norm"
        elif role == "gates":
            return f"blk\\.{layer_idx}\\.ffn_gate_inp"
        else:
            return f"blk\\.{layer_idx}\\.{role}"

    return group_name  # fallback


class GGUFBuilder:
    """Generate llama-quantize commands from RAMP-local search results."""

    def __init__(self, analyzer: GGUFAnalyzer, config: dict):
        """
        Args:
            analyzer: parsed GGUF structure
            config: {group_name -> quant_type_name} from search
        """
        self.analyzer = analyzer
        self.config = config

    def find_base_type(self) -> str:
        """Find the most common quant type in config to use as base.

        Minimizes the number of --tensor-type overrides needed.
        """
        type_counts = {}
        for gname, qtype in self.config.items():
            # Weight by number of elements (prefer base type for bulk data)
            group = self.analyzer.groups.get(gname)
            elements = group.total_elements if group else 1
            type_counts[qtype] = type_counts.get(qtype, 0) + elements

        return max(type_counts, key=type_counts.get)

    def generate_command(self, input_bf16: str, output_gguf: str,
                         base_type: str = None,
                         imatrix_path: str = None,
                         llama_quantize: str = None) -> str:
        """Generate the llama-quantize command.

        Args:
            input_bf16: path to BF16/FP16 source GGUF
            output_gguf: path for output mixed-precision GGUF
            base_type: base quantization type (auto-detected if None)
            imatrix_path: optional imatrix file for IQ types
            llama_quantize: path to llama-quantize binary
        """
        if base_type is None:
            base_type = self.find_base_type()

        if llama_quantize is None:
            llama_quantize = os.path.expanduser(
                "~/ik_llama.cpp/build_sm120/bin/llama-quantize")

        # ggml_type_name() uses specific mixed-case names
        GGML_TYPE_NAMES = {
            "IQ2_XXS": "iq2_xxs", "IQ2_XS": "iq2_xs", "IQ2_S": "iq2_s",
            "IQ3_XXS": "iq3_xxs", "IQ3_S": "iq3_s",
            "IQ4_NL": "iq4_nl", "IQ4_XS": "iq4_xs",
            "Q2_K": "q2_K", "Q3_K": "q3_K",
            "Q4_K": "q4_K", "Q4_K_M": "q4_K", "Q4_K_S": "q4_K_S",
            "Q5_K": "q5_K", "Q5_K_M": "q5_K", "Q5_K_S": "q5_K_S",
            "Q6_K": "q6_K",
            "Q8_0": "q8_0",
            "F16": "f16", "F32": "f32", "BF16": "bf16",
        }

        # Build --custom-q rules (regex=type pairs)
        custom_rules = []
        for gname, qtype in sorted(self.config.items()):
            if qtype.upper() == base_type.upper():
                continue  # skip base type
            regex = group_to_regex(gname)
            ggml_name = GGML_TYPE_NAMES.get(qtype, qtype.lower())
            custom_rules.append(f"{regex}={ggml_name}")

        parts = [f"{llama_quantize}"]

        if imatrix_path:
            parts.append(f"  --imatrix {imatrix_path}")

        # Use --custom-q for compact override specification
        if custom_rules:
            # Split into chunks to keep lines reasonable
            chunk_size = 5
            for i in range(0, len(custom_rules), chunk_size):
                chunk = ",".join(custom_rules[i:i + chunk_size])
                parts.append(f'  --custom-q "{chunk}"')

        # Input, output, base type
        parts.append(f"  {input_bf16}")
        parts.append(f"  {output_gguf}")
        parts.append(f"  {base_type}")

        return " \\\n".join(parts)

    def generate_summary(self) -> str:
        """Generate human-readable summary of the mixed-precision plan."""
        lines = [
            "RAMP-Local Mixed-Precision Plan",
            "=" * 60,
            "",
            f"{'Layer':<8} {'Attn':>8} {'Shared':>8} {'Experts':>8} {'Norms':>6} {'Gates':>6}",
            "-" * 60,
        ]

        for layer_idx in range(self.analyzer.n_layers):
            attn = self.config.get(f"layer.{layer_idx}.attn", "?")
            shared = self.config.get(f"layer.{layer_idx}.shared", "?")
            experts = self.config.get(f"layer.{layer_idx}.experts", "?")
            norms = self.config.get(f"layer.{layer_idx}.norms", "?")
            gates = self.config.get(f"layer.{layer_idx}.gates", "?")
            lines.append(f"{layer_idx:>5}    {attn:>8} {shared:>8} "
                        f"{experts:>8} {norms:>6} {gates:>6}")

        # Global tensors
        lines.append("")
        lines.append("Global tensors:")
        for gname, qtype in sorted(self.config.items()):
            if gname.startswith("global."):
                lines.append(f"  {gname}: {qtype}")

        # Statistics
        lines.append("")
        lines.append("Type distribution:")
        type_counts = {}
        type_elements = {}
        for gname, qtype in self.config.items():
            group = self.analyzer.groups.get(gname)
            el = group.total_elements if group else 0
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
            type_elements[qtype] = type_elements.get(qtype, 0) + el

        total_el = sum(type_elements.values())
        for qtype in sorted(type_counts, key=lambda q: BPW.get(q, 0)):
            n = type_counts[qtype]
            el = type_elements[qtype]
            pct = 100 * el / total_el if total_el > 0 else 0
            lines.append(f"  {qtype:>10}: {n:>4} groups, {el/1e6:>8.1f}M params ({pct:>5.1f}%)")

        return "\n".join(lines)

    def save_config(self, path: str, metadata: dict = None) -> None:
        """Save configuration to JSON for reproducibility."""
        data = {
            "config": self.config,
            "base_type": self.find_base_type(),
            "n_layers": self.analyzer.n_layers,
            "total_groups": len(self.config),
            "gguf_source": self.analyzer.gguf_path,
        }
        if metadata:
            data.update(metadata)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate llama-quantize command from RAMP-local config")
    parser.add_argument("config_json", help="Path to search result JSON")
    parser.add_argument("--input-bf16", required=True,
                       help="Path to BF16/FP16 source GGUF")
    parser.add_argument("--output-gguf", required=True,
                       help="Path for output mixed-precision GGUF")
    parser.add_argument("--imatrix", default=None,
                       help="Path to imatrix file")
    parser.add_argument("--gguf-path", default=None,
                       help="Path to GGUF for analysis (default: from config)")
    args = parser.parse_args()

    with open(args.config_json) as f:
        result = json.load(f)

    config = result["config"]
    gguf_path = args.gguf_path or result.get("gguf_source", "")

    if gguf_path:
        analyzer = GGUFAnalyzer(gguf_path)
        builder = GGUFBuilder(analyzer, config)

        print(builder.generate_summary())
        print()
        print("=" * 60)
        print("llama-quantize command:")
        print("=" * 60)
        print(builder.generate_command(args.input_bf16, args.output_gguf,
                                       imatrix_path=args.imatrix))
    else:
        print("ERROR: need --gguf-path to generate command")
