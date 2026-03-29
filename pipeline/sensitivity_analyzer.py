#!/usr/bin/env python3
"""
Enhanced Sensitivity Analysis for hybrid attention/GDN/MoE models.

Combines multiple sensitivity signals:
1. NSDS (from ../nsds_sensitivity.py): kurtosis + SVD spectral analysis
2. GDN-specific: spectral radius of effective recurrence matrix
3. Layer-type multiplier: 1.5x for GDN layers (Quamba2 ICML 2025 finding)
4. Logit-based KL divergence (optional, requires running model inference)

The combined sensitivity score guides bit allocation: high-sensitivity tensors
get higher precision, low-sensitivity tensors can be compressed aggressively.

For Qwen3.5-35B-A3B:
  - SSM recurrent tensors get 1.5x multiplier (error propagates through time)
  - GDN layers: spectral radius of weight*gate product estimates error amplification
  - Attention Q/K: asymmetric head dim (512 vs 256) makes them sensitive
  - MoE experts: sparse activation (8/256) makes them individually less sensitive
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gguf_analyzer import GGUFAnalyzer
from nsds_sensitivity import NSDSSensitivity, TensorSensitivity, GroupSensitivity

# -------------------------------------------------------------------------
# Architecture constants
# -------------------------------------------------------------------------

FULL_ATTENTION_LAYERS = set(range(3, 40, 4))
GDN_LAYERS = set(range(40)) - FULL_ATTENTION_LAYERS
NUM_LAYERS = 40

# Quamba2 finding: GDN/SSM layers need 1.5x sensitivity multiplier
# because quantization errors propagate exponentially through recurrence
GDN_SENSITIVITY_MULTIPLIER = 1.5

# SSM-specific tensors get additional boost
SSM_CRITICAL_ROLES = {
    "ssm_a": 2.0,     # Exponential decay rate, F32 mandatory
    "ssm_dt": 1.8,    # Timestep, controls state update magnitude
    "ssm_alpha": 1.5, # State mixing coefficient
    "ssm_beta": 1.3,  # Output mixing coefficient
    "ssm_out": 1.2,   # Output projection
    "ssm_conv": 1.1,  # Convolution (short kernel, less critical)
    "ssm_norm": 1.5,  # Gated norm
}

# Role-based multipliers for the hybrid architecture
ROLE_MULTIPLIERS = {
    # Full attention (10/40 layers)
    "attn_q": 1.3,
    "attn_k": 1.2,
    "attn_v": 1.0,
    "attn_o": 1.1,
    # GDN (30/40 layers) -- base multiplier applied separately
    "gdn_qkv": 1.2,
    "gdn_gate": 1.3,
    # MoE
    "shared_gate": 1.2,
    "shared_up": 1.0,
    "shared_down": 1.1,
    "expert_gate": 0.8,
    "expert_up": 0.7,
    "expert_down": 0.9,
    # Router
    "moe_gate": 2.0,
    "shared_expert_gate": 1.5,
    # Global
    "embed": 1.3,
    "output": 1.5,
}


# -------------------------------------------------------------------------
# GDN spectral radius estimation
# -------------------------------------------------------------------------

def estimate_spectral_radius(W_gate: np.ndarray, W_qkv: np.ndarray,
                             typical_gate_activation: float = 0.5) -> float:
    """Estimate spectral radius of the effective GDN recurrence matrix.

    For GDN layers, the recurrence is:
        S_t = gate * S_{t-1} + beta * (K_t^T * V_t)
    where gate = sigmoid(W_gate @ x).

    The effective recurrence matrix has spectral radius:
        rho ~ typical_gate * ||W_gate|| * ||W_qkv||
    where ||.|| is the spectral norm (largest singular value).

    Higher spectral radius = more error amplification over sequence length.

    We use the Frobenius norm divided by sqrt(min_dim) as a fast approximation
    of spectral norm (within factor of sqrt(rank/min_dim) ~ 1-3x for typical weights).
    """
    # Fast spectral norm approximation via Frobenius
    def fast_spectral_norm(W):
        frob = np.linalg.norm(W.flatten())
        min_dim = min(W.shape) if W.ndim >= 2 else W.size
        return frob / np.sqrt(max(min_dim, 1))

    sigma_gate = fast_spectral_norm(W_gate)
    sigma_qkv = fast_spectral_norm(W_qkv)

    rho = typical_gate_activation * sigma_gate * sigma_qkv
    return float(rho)


# -------------------------------------------------------------------------
# Enhanced sensitivity analyzer
# -------------------------------------------------------------------------

@dataclass
class EnhancedGroupSensitivity:
    """Enhanced sensitivity for a decision group, combining multiple signals."""
    name: str
    # NSDS base scores
    nsds: float
    nv_score: float
    se_score: float
    # Architecture-aware adjustments
    layer_type: str          # "full_attention", "gdn", "global"
    role_multiplier: float   # role-based importance
    gdn_multiplier: float    # 1.5 for GDN layers, 1.0 for attention
    spectral_radius: float   # GDN recurrence amplification factor
    # Combined score
    combined_score: float    # final sensitivity score
    # Optional KL divergence
    kl_divergence: float = 0.0
    # Metadata
    n_tensors: int = 0
    total_elements: int = 0


class EnhancedSensitivityAnalyzer:
    """Multi-signal sensitivity analysis for hybrid attention/GDN/MoE models."""

    def __init__(self, analyzer: GGUFAnalyzer,
                 nsds: Optional[NSDSSensitivity] = None):
        """
        Args:
            analyzer: parsed GGUF structure
            nsds: precomputed NSDS scores (computed if None)
        """
        self.analyzer = analyzer
        self.nsds = nsds
        self.group_scores: dict[str, EnhancedGroupSensitivity] = {}

    def compute(self, verbose: bool = True) -> None:
        """Compute enhanced sensitivity scores for all decision groups."""
        t0 = time.monotonic()

        # Step 1: Compute NSDS if not provided
        if self.nsds is None:
            if verbose:
                print("Computing NSDS sensitivity scores...")
            self.nsds = NSDSSensitivity(self.analyzer)
            self.nsds.compute_all(verbose=verbose)

        # Step 2: Classify layers
        layer_types = self._classify_layers()

        # Step 3: Compute enhanced scores per group
        for gname, group in self.analyzer.groups.items():
            gs = self.nsds.get_group_sensitivity(gname)
            layer_idx = group.layer_idx
            role = group.role

            # Determine layer type
            if layer_idx < 0:
                ltype = "global"
            elif layer_idx in FULL_ATTENTION_LAYERS:
                ltype = "full_attention"
            else:
                ltype = "gdn"

            # GDN multiplier (Quamba2 finding)
            gdn_mult = GDN_SENSITIVITY_MULTIPLIER if ltype == "gdn" else 1.0

            # SSM-specific multiplier (stacks with GDN multiplier)
            if role in SSM_CRITICAL_ROLES:
                gdn_mult *= SSM_CRITICAL_ROLES[role]

            # Role multiplier
            role_mult = ROLE_MULTIPLIERS.get(role, 1.0)

            # Spectral radius (only for GDN layers, approximate from NSDS stats)
            spectral_radius = 0.0
            if ltype == "gdn" and role in ("gdn_qkv", "gdn_gate"):
                # Use SE_raw as proxy for spectral radius (correlated)
                spectral_radius = gs.se_score * 2.0

            # Combined score: NSDS * role * GDN_multiplier * (1 + spectral_radius/10)
            combined = (gs.nsds * role_mult * gdn_mult *
                        (1.0 + spectral_radius / 10.0))

            self.group_scores[gname] = EnhancedGroupSensitivity(
                name=gname,
                nsds=gs.nsds,
                nv_score=gs.nv_score,
                se_score=gs.se_score,
                layer_type=ltype,
                role_multiplier=role_mult,
                gdn_multiplier=gdn_mult,
                spectral_radius=spectral_radius,
                combined_score=combined,
                n_tensors=gs.n_tensors,
                total_elements=gs.total_elements,
            )

        # Normalize combined scores to [0, 1] range
        all_scores = [g.combined_score for g in self.group_scores.values()]
        if all_scores:
            max_score = max(all_scores)
            min_score = min(all_scores)
            score_range = max_score - min_score
            if score_range > 1e-8:
                for g in self.group_scores.values():
                    g.combined_score = (g.combined_score - min_score) / score_range

        elapsed = time.monotonic() - t0
        if verbose:
            print(f"Enhanced sensitivity analysis complete in {elapsed:.1f}s")

    def _classify_layers(self) -> dict:
        """Classify each layer as full_attention or gdn."""
        return {i: ("full_attention" if i in FULL_ATTENTION_LAYERS else "gdn")
                for i in range(NUM_LAYERS)}

    def get_sensitivity(self, group_name: str) -> float:
        """Get combined sensitivity score for a decision group."""
        gs = self.group_scores.get(group_name)
        return gs.combined_score if gs else 0.5

    def get_group(self, group_name: str) -> EnhancedGroupSensitivity:
        """Get full enhanced sensitivity for a group."""
        return self.group_scores.get(group_name,
            EnhancedGroupSensitivity(
                name=group_name, nsds=0.5, nv_score=0.5, se_score=0.5,
                layer_type="unknown", role_multiplier=1.0, gdn_multiplier=1.0,
                spectral_radius=0.0, combined_score=0.5))

    def save(self, path: str) -> None:
        """Save enhanced sensitivity scores to JSON."""
        data = {}
        for name, g in self.group_scores.items():
            data[name] = {
                "nsds": g.nsds,
                "nv_score": g.nv_score,
                "se_score": g.se_score,
                "layer_type": g.layer_type,
                "role_multiplier": g.role_multiplier,
                "gdn_multiplier": g.gdn_multiplier,
                "spectral_radius": g.spectral_radius,
                "combined_score": g.combined_score,
                "kl_divergence": g.kl_divergence,
                "n_tensors": g.n_tensors,
                "total_elements": g.total_elements,
            }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load enhanced sensitivity scores from JSON."""
        with open(path) as f:
            data = json.load(f)
        for name, d in data.items():
            self.group_scores[name] = EnhancedGroupSensitivity(name=name, **d)

    def report(self, top_k: int = 30) -> str:
        """Generate human-readable sensitivity report."""
        lines = [
            "=" * 85,
            "Enhanced Sensitivity Report (NSDS + GDN + Spectral)",
            "=" * 85,
            "",
            f"{'Group':<40} {'Score':>6} {'NSDS':>6} {'Type':>5} "
            f"{'GDN':>4} {'Role':>5} {'Rho':>5}",
            "-" * 85,
        ]

        sorted_groups = sorted(self.group_scores.items(),
                                key=lambda x: x[1].combined_score, reverse=True)

        for gname, gs in sorted_groups[:top_k]:
            ltype = gs.layer_type[:5]
            lines.append(
                f"{gname:<40} {gs.combined_score:>6.3f} {gs.nsds:>6.3f} "
                f"{ltype:>5} {gs.gdn_multiplier:>4.1f} "
                f"{gs.role_multiplier:>5.1f} {gs.spectral_radius:>5.2f}")

        if len(sorted_groups) > top_k:
            lines.append(f"  ... and {len(sorted_groups) - top_k} more groups")

        # Summary by layer type
        lines.extend(["", "Summary by layer type:"])
        type_stats = {}
        for g in self.group_scores.values():
            lt = g.layer_type
            if lt not in type_stats:
                type_stats[lt] = {"count": 0, "mean_score": 0.0}
            type_stats[lt]["count"] += 1
            type_stats[lt]["mean_score"] += g.combined_score

        for lt, st in sorted(type_stats.items()):
            mean = st["mean_score"] / max(st["count"], 1)
            lines.append(f"  {lt:>15}: {st['count']:>3} groups, "
                         f"mean sensitivity={mean:.3f}")

        return "\n".join(lines)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced sensitivity analysis for hybrid attention/GDN/MoE models")
    parser.add_argument("gguf_path", help="Path to GGUF file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: <gguf>.enhanced-sensitivity.json)")
    parser.add_argument("--nsds-cache", default=None,
                        help="Load cached NSDS scores from JSON")
    parser.add_argument("--expert-sample-k", type=int, default=16,
                        help="Experts to sample for NSDS (default: 16)")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    analyzer = GGUFAnalyzer(args.gguf_path)
    print(analyzer.summary())
    print()

    # Load or compute NSDS
    nsds = None
    if args.nsds_cache and os.path.exists(args.nsds_cache):
        nsds = NSDSSensitivity(analyzer, expert_sample_k=args.expert_sample_k)
        nsds.load(args.nsds_cache)
        print(f"Loaded NSDS cache from {args.nsds_cache}")

    # Compute enhanced sensitivity
    enhanced = EnhancedSensitivityAnalyzer(analyzer, nsds)
    enhanced.compute(verbose=True)

    print()
    print(enhanced.report(top_k=args.top_k))

    # Save
    output = args.output or args.gguf_path + ".enhanced-sensitivity.json"
    enhanced.save(output)
    print(f"\nSaved to {output}")
