#!/usr/bin/env python3
"""
Intelligent Bit Allocator for hybrid attention/GDN/MoE models.

Given sensitivity scores and a size budget, finds the optimal per-tensor
quantization type assignment using greedy + evolutionary search.

Key constraints for Qwen3.5-35B-A3B:
  - SSM tensors: minimum Q5_K for ssm_out, Q6_K for ssm_alpha/beta, F32 for ssm_a/ssm_dt
  - Attention layers: minimum Q4_K for Q/K/V/O
  - GDN layers: minimum Q5_K for gdn_qkv/gdn_gate
  - Norms: always Q8_0 (tiny, critical)
  - MoE router: always Q8_0 (tiny, critical)
  - MoE experts: IQ2_XXS to Q4_K range (this is the budget lever)
  - Shared experts: minimum Q4_K (always active, more sensitive)

The allocator respects the RAMP config JSON format used by ramp-quantize.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from copy import deepcopy

import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gguf_analyzer import (GGUFAnalyzer, SEARCH_QUANT_TYPES, QTYPE_NAME_TO_ID,
                            BLOCK_SIZES, BLOCK_BYTES, BPW)
from proxy_model import ProxyModel, QuantErrorDB
from nsds_sensitivity import NSDSSensitivity
from search_evo import EvoSearch, GreedySearch

# Import enhanced sensitivity if available
try:
    from pipeline.sensitivity_analyzer import EnhancedSensitivityAnalyzer
except ImportError:
    EnhancedSensitivityAnalyzer = None

# -------------------------------------------------------------------------
# Architecture constraints
# -------------------------------------------------------------------------

# Minimum allowed quant types per role (hardware safety / quality floor)
# Keys match group roles in gguf_analyzer.py
ROLE_CONSTRAINTS = {
    # SSM: recurrent state, error accumulates exponentially
    "ssm_a":     {"min": "F32",  "max": "F32"},   # Must stay F32 (-exp transform)
    "ssm_dt":    {"min": "F32",  "max": "F32"},   # Must stay F32 (bias)
    "ssm_alpha": {"min": "Q6_K", "max": "Q8_0"},  # State mixing
    "ssm_beta":  {"min": "Q6_K", "max": "Q8_0"},  # Output mixing
    "ssm_out":   {"min": "Q5_K", "max": "Q8_0"},  # Output projection
    "ssm_conv":  {"min": "Q5_K", "max": "Q8_0"},  # Short convolution
    "ssm_norm":  {"min": "Q8_0", "max": "Q8_0"},  # Gated norm

    # GDN (linear attention)
    "gdn_qkv":   {"min": "Q5_K", "max": "Q8_0"},
    "gdn_gate":  {"min": "Q5_K", "max": "Q8_0"},
    "gdn_q_norm": {"min": "Q8_0", "max": "Q8_0"},
    "gdn_k_norm": {"min": "Q8_0", "max": "Q8_0"},

    # Full attention
    "attn_q":    {"min": "Q4_K", "max": "Q8_0"},
    "attn_k":    {"min": "Q4_K", "max": "Q8_0"},
    "attn_v":    {"min": "Q4_K", "max": "Q8_0"},
    "attn_o":    {"min": "Q4_K", "max": "Q8_0"},

    # Norms (tiny, critical)
    "attn_norm":       {"min": "Q8_0", "max": "Q8_0"},
    "ffn_norm":        {"min": "Q8_0", "max": "Q8_0"},
    "post_attn_norm":  {"min": "Q8_0", "max": "Q8_0"},
    "output_norm":     {"min": "Q8_0", "max": "Q8_0"},

    # MoE router (tiny, critical for routing decisions)
    "moe_gate":            {"min": "Q8_0", "max": "Q8_0"},
    "shared_expert_gate":  {"min": "Q8_0", "max": "Q8_0"},

    # Shared experts (always active, more sensitive than routed)
    "shared_gate": {"min": "Q4_K", "max": "Q8_0"},
    "shared_up":   {"min": "Q4_K", "max": "Q8_0"},
    "shared_down": {"min": "Q4_K", "max": "Q8_0"},

    # Routed experts (the budget lever: 256 experts, 8 active)
    "expert_gate": {"min": "IQ2_XXS", "max": "Q4_K"},
    "expert_up":   {"min": "IQ2_XXS", "max": "Q4_K"},
    "expert_down": {"min": "IQ2_XXS", "max": "Q4_K"},

    # Global tensors
    "embed":  {"min": "Q4_K", "max": "Q6_K"},
    "output": {"min": "Q5_K", "max": "Q8_0"},
}

# Group-level constraints (for group names like "layer.5.ssm")
GROUP_ROLE_MAP = {
    "ssm": ["ssm_a", "ssm_dt", "ssm_alpha", "ssm_beta", "ssm_out",
            "ssm_conv", "ssm_norm"],
    "gdn": ["gdn_qkv", "gdn_gate"],
    "attn": ["attn_q", "attn_k", "attn_v", "attn_o"],
    "shared": ["shared_gate", "shared_up", "shared_down"],
    "experts": ["expert_gate", "expert_up", "expert_down"],
    "norms": ["attn_norm", "ffn_norm", "post_attn_norm",
              "gdn_q_norm", "gdn_k_norm"],
    "gates": ["moe_gate", "shared_expert_gate"],
}

# Ordered quant types by BPW (ascending)
ALL_QUANT_TYPES = ["IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S",
                    "Q4_K", "Q5_K", "Q6_K", "Q8_0", "F32"]

QUANT_BPW = {}
for qt in ALL_QUANT_TYPES:
    if qt == "F32":
        QUANT_BPW[qt] = 32.0
    else:
        qid = QTYPE_NAME_TO_ID[qt]
        QUANT_BPW[qt] = BLOCK_BYTES[qid] * 8.0 / BLOCK_SIZES[qid]

BPW_ORDER = sorted(ALL_QUANT_TYPES, key=lambda qt: QUANT_BPW[qt])


def get_group_constraint(group_name: str, role: str) -> dict:
    """Get min/max quant type constraint for a group.

    Uses role to look up constraints. For groups containing mixed roles
    (e.g., "layer.5.ssm" containing ssm_a + ssm_alpha + ...), returns
    the most restrictive constraint (highest minimum).
    """
    # Direct role match
    if role in ROLE_CONSTRAINTS:
        return ROLE_CONSTRAINTS[role]

    # Group-level: find the role pattern in the group name
    parts = group_name.split(".")
    if len(parts) >= 3:
        group_type = parts[2]  # "ssm", "gdn", "attn", etc.
        if group_type in GROUP_ROLE_MAP:
            # Use the most restrictive minimum from constituent roles
            roles = GROUP_ROLE_MAP[group_type]
            constraints = [ROLE_CONSTRAINTS.get(r, {"min": "IQ2_XXS", "max": "Q8_0"})
                           for r in roles]
            if constraints:
                min_idx = max(BPW_ORDER.index(c["min"]) for c in constraints)
                max_idx = min(BPW_ORDER.index(c["max"]) for c in constraints)
                return {"min": BPW_ORDER[min_idx],
                        "max": BPW_ORDER[max(min_idx, max_idx)]}

    # Global tensors
    if group_name.startswith("global."):
        global_role = parts[1] if len(parts) >= 2 else "unknown"
        if global_role in ROLE_CONSTRAINTS:
            return ROLE_CONSTRAINTS[global_role]

    # Default: wide range
    return {"min": "IQ2_XXS", "max": "Q8_0"}


def constrained_quant_types(group_name: str, role: str) -> list:
    """Get allowed quant types for a group, respecting constraints."""
    constraint = get_group_constraint(group_name, role)
    min_idx = BPW_ORDER.index(constraint["min"])
    max_idx = BPW_ORDER.index(constraint["max"])
    return BPW_ORDER[min_idx:max_idx + 1]


# -------------------------------------------------------------------------
# Fixed groups (cannot be changed by search)
# -------------------------------------------------------------------------

def build_fixed_groups(analyzer: GGUFAnalyzer) -> dict:
    """Identify groups that have exactly one allowed quant type (fixed).

    Returns: {group_name: quant_type} for groups that cannot be changed.
    """
    fixed = {}
    for gname, group in analyzer.groups.items():
        allowed = constrained_quant_types(gname, group.role)
        if len(allowed) == 1:
            fixed[gname] = allowed[0]
    return fixed


def build_search_space(analyzer: GGUFAnalyzer) -> dict:
    """Build per-group allowed quant types for search.

    Returns: {group_name: [quant_types]} for groups that can be searched.
    """
    space = {}
    for gname, group in analyzer.groups.items():
        allowed = constrained_quant_types(gname, group.role)
        if len(allowed) > 1:
            space[gname] = allowed
    return space


# -------------------------------------------------------------------------
# Enhanced proxy with architecture-aware weights
# -------------------------------------------------------------------------

class HybridProxyModel(ProxyModel):
    """Proxy model enhanced with hybrid architecture awareness.

    Overrides the base ProxyModel to use enhanced sensitivity scores
    and architecture-specific role weights.
    """

    def __init__(self, analyzer: GGUFAnalyzer, sensitivity: NSDSSensitivity,
                 error_db: QuantErrorDB,
                 enhanced: Optional[object] = None):
        super().__init__(analyzer, sensitivity, error_db)
        self.enhanced = enhanced

        # Override group weights with enhanced scores if available
        if enhanced is not None and hasattr(enhanced, 'group_scores'):
            for gname in analyzer.groups:
                egs = enhanced.group_scores.get(gname)
                if egs:
                    self._group_weight[gname] = (
                        egs.combined_score *
                        self._size_weights.get(gname, 1.0) *
                        self._depth_weights.get(gname, 1.0)
                    )


# -------------------------------------------------------------------------
# Constrained evolutionary search
# -------------------------------------------------------------------------

class ConstrainedEvoSearch(EvoSearch):
    """Evolutionary search with per-group quant type constraints."""

    def __init__(self, proxy: ProxyModel, budget_bytes: int,
                 search_space: dict, **kwargs):
        """
        Args:
            search_space: {group_name: [allowed_quant_types]}
        """
        self.search_space = search_space
        # Override quant_types to union of all allowed types
        all_types = set()
        for types in search_space.values():
            all_types.update(types)
        kwargs["quant_types"] = sorted(all_types,
                                        key=lambda qt: QUANT_BPW.get(qt, 0))
        super().__init__(proxy, budget_bytes, **kwargs)

    def random_config(self) -> dict:
        """Generate random config respecting per-group constraints."""
        config = {}
        for dp in self.decision_points:
            allowed = self.search_space.get(dp, self.quant_types)
            config[dp] = self.rng.choice(allowed)
        return config

    def mutate(self, config: dict) -> dict:
        """Mutate respecting per-group constraints."""
        new = dict(config)
        for dp in self.decision_points:
            if self.rng.random() < self.mutation_rate:
                allowed = self.search_space.get(dp, self.quant_types)
                new[dp] = self.rng.choice(allowed)
        return self.repair(new)


class ConstrainedGreedySearch(GreedySearch):
    """Greedy search with per-group quant type constraints."""

    def __init__(self, proxy: ProxyModel, budget_bytes: int,
                 search_space: dict, **kwargs):
        self.search_space = search_space
        all_types = set()
        for types in search_space.values():
            all_types.update(types)
        kwargs["quant_types"] = sorted(all_types,
                                        key=lambda qt: QUANT_BPW.get(qt, 0))
        super().__init__(proxy, budget_bytes, **kwargs)

    def search(self, verbose: bool = True) -> tuple:
        """Greedy search respecting per-group constraints."""
        t0 = time.time()

        # Start at minimum allowed quant for each group
        config = {}
        for dp in self.decision_points:
            allowed = self.search_space.get(dp, self._bpw_order)
            # Start at lowest allowed
            min_type = min(allowed, key=lambda qt: QUANT_BPW.get(qt, 0))
            config[dp] = min_type
        config.update(self.fixed_groups)

        step = 0
        while True:
            step += 1
            best_upgrade = None
            best_ratio = -float('inf')

            for dp in self.decision_points:
                allowed = self.search_space.get(dp, self._bpw_order)
                allowed_sorted = sorted(allowed, key=lambda qt: QUANT_BPW.get(qt, 0))
                current_idx = allowed_sorted.index(config[dp]) if config[dp] in allowed_sorted else 0
                if current_idx >= len(allowed_sorted) - 1:
                    continue

                new_qt = allowed_sorted[current_idx + 1]
                old_loss = self.proxy.proxy_loss_single(dp, config[dp])
                new_loss = self.proxy.proxy_loss_single(dp, new_qt)
                improvement = old_loss - new_loss

                old_size = self.proxy.analyzer.group_byte_size(dp, config[dp])
                new_size = self.proxy.analyzer.group_byte_size(dp, new_qt)
                cost = new_size - old_size
                if cost <= 0:
                    continue

                test_config = dict(config)
                test_config[dp] = new_qt
                if self.proxy.total_size(test_config) > self.budget:
                    continue

                ratio = improvement / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_upgrade = (dp, new_qt)

            if best_upgrade is None:
                break

            config[best_upgrade[0]] = best_upgrade[1]

            if verbose and step % 50 == 0:
                size_gb = self.proxy.total_size(config) / (1024**3)
                score = self.proxy.proxy_loss(config)
                print(f"  Step {step}: score={score:.6f} size={size_gb:.2f}GB")

        score = self.proxy.proxy_loss(config)
        elapsed = time.time() - t0

        if verbose:
            size_gb = self.proxy.total_size(config) / (1024**3)
            bpw = self.proxy.config_bpw(config)
            print(f"  Greedy: {step} steps in {elapsed:.2f}s")
            print(f"  Result: score={score:.6f} size={size_gb:.2f}GB bpw={bpw:.2f}")

        return config, score


# -------------------------------------------------------------------------
# Main allocator
# -------------------------------------------------------------------------

def allocate(analyzer: GGUFAnalyzer,
             sensitivity: NSDSSensitivity,
             budget_gb: float,
             enhanced: Optional[object] = None,
             error_cache: Optional[str] = None,
             method: str = "both",
             pop_size: int = 128,
             generations: int = 200,
             seed: int = 42,
             verbose: bool = True) -> dict:
    """Run the full allocation pipeline.

    Returns: complete config dict {group_name: quant_type} for ramp-quantize.
    """
    budget_bytes = int(budget_gb * 1024**3)

    # Build error database
    error_db = QuantErrorDB()
    if error_cache and os.path.exists(error_cache):
        error_db.populate_from_measurements(error_cache)
        if verbose:
            print(f"Loaded measured errors from {error_cache}")
    else:
        error_db.populate_approximate(analyzer, sensitivity)
        if verbose:
            print("Using approximate quantization error estimates")

    # Build proxy model
    proxy = HybridProxyModel(analyzer, sensitivity, error_db, enhanced)

    # Build constraints
    fixed = build_fixed_groups(analyzer)
    search_space = build_search_space(analyzer)

    if verbose:
        print(f"\nAllocation constraints:")
        print(f"  Fixed groups: {len(fixed)} (norms, SSM_a/dt, gates)")
        print(f"  Searchable groups: {len(search_space)}")
        print(f"  Budget: {budget_gb:.2f} GB ({budget_bytes:,} bytes)")

        # Show constraint summary
        n_ssm = sum(1 for g in fixed if "ssm" in g)
        n_norm = sum(1 for g in fixed if "norm" in g)
        n_gate = sum(1 for g in fixed if "gate" in g)
        print(f"  Fixed breakdown: {n_ssm} SSM, {n_norm} norms, {n_gate} gates")

    results = {}

    if method in ("greedy", "both"):
        if verbose:
            print("\n--- Constrained Greedy Search ---")
        greedy = ConstrainedGreedySearch(
            proxy, budget_bytes, search_space, fixed_groups=fixed)
        g_config, g_score = greedy.search(verbose=verbose)
        results["greedy"] = {"config": g_config, "score": g_score}

    if method in ("evo", "both"):
        if verbose:
            print("\n--- Constrained Evolutionary Search ---")
        evo = ConstrainedEvoSearch(
            proxy, budget_bytes, search_space,
            population_size=pop_size, generations=generations,
            seed=seed, fixed_groups=fixed)
        e_config, e_score, _ = evo.search(verbose=verbose)
        results["evo"] = {"config": e_config, "score": e_score}

    # Pick best
    best_method = min(results, key=lambda m: results[m]["score"])
    best_config = results[best_method]["config"]
    best_score = results[best_method]["score"]

    if verbose:
        print(f"\nBest method: {best_method} (score={best_score:.6f})")
        size_gb = proxy.total_size(best_config) / (1024**3)
        bpw = proxy.config_bpw(best_config)
        print(f"Size: {size_gb:.2f} GB, BPW: {bpw:.2f}")
        print(proxy.report_config(best_config, f"Allocation ({best_method})"))

    return best_config


def save_ramp_config(config: dict, output_path: str,
                     analyzer: GGUFAnalyzer,
                     metadata: Optional[dict] = None):
    """Save allocation result as RAMP config JSON for ramp-quantize.

    Format matches what ramp_quantize.c expects:
    {
      "base_type": "IQ3_S",
      "config": {"layer.0.experts": "IQ3_S", ...}
    }
    """
    from gguf_builder import GGUFBuilder
    builder = GGUFBuilder(analyzer, config)
    base_type = builder.find_base_type()

    data = {
        "base_type": base_type,
        "config": config,
    }
    if metadata:
        data.update(metadata)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intelligent bit allocation for hybrid attention/GDN/MoE models")
    parser.add_argument("gguf_path", help="Path to GGUF file")
    parser.add_argument("--budget-gb", type=float, required=True,
                        help="Target size in GB")
    parser.add_argument("--method", choices=["evo", "greedy", "both"],
                        default="both")
    parser.add_argument("--sensitivity-cache", default=None,
                        help="Load NSDS sensitivity cache")
    parser.add_argument("--enhanced-cache", default=None,
                        help="Load enhanced sensitivity cache")
    parser.add_argument("--error-cache", default=None,
                        help="Load measured quant errors")
    parser.add_argument("--output", "-o", required=True,
                        help="Output RAMP config JSON")
    parser.add_argument("--pop-size", type=int, default=128)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    # Analyze GGUF
    analyzer = GGUFAnalyzer(args.gguf_path)
    if verbose:
        print(analyzer.summary())

    # Load or compute sensitivity
    nsds = NSDSSensitivity(analyzer)
    if args.sensitivity_cache and os.path.exists(args.sensitivity_cache):
        nsds.load(args.sensitivity_cache)
        if verbose:
            print(f"Loaded NSDS from {args.sensitivity_cache}")
    else:
        nsds.compute_all(verbose=verbose)

    # Load enhanced sensitivity if available
    enhanced = None
    if args.enhanced_cache and os.path.exists(args.enhanced_cache):
        if EnhancedSensitivityAnalyzer is not None:
            enhanced = EnhancedSensitivityAnalyzer(analyzer, nsds)
            enhanced.load(args.enhanced_cache)
            if verbose:
                print(f"Loaded enhanced sensitivity from {args.enhanced_cache}")

    # Run allocation
    config = allocate(
        analyzer, nsds, args.budget_gb,
        enhanced=enhanced,
        error_cache=args.error_cache,
        method=args.method,
        pop_size=args.pop_size,
        generations=args.generations,
        seed=args.seed,
        verbose=verbose)

    # Save
    save_ramp_config(config, args.output, analyzer, metadata={
        "budget_gb": args.budget_gb,
        "method": args.method,
        "gguf_source": args.gguf_path,
    })
    if verbose:
        print(f"\nConfig saved to {args.output}")
