#!/usr/bin/env python3
"""
RAMP-Local: Proxy Model for Quality Loss Estimation

Combines NSDS sensitivity scores with quantization error estimates to produce
a fast proxy for perplexity change. This proxy can evaluate millions of
configurations per second, replacing the need for actual model inference.

The proxy loss function:
  proxy_loss(config) = sum over groups G of:
      w_role(G) * w_depth(G) * w_size(G) * quant_error(G, config[G]) * sensitivity(G)

Optionally calibratable: run 20-30 actual PPL evaluations, fit a linear model
from proxy_loss -> actual_delta_PPL, then use the calibrated version.
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import Optional

from gguf_analyzer import (GGUFAnalyzer, DecisionGroup, SEARCH_QUANT_TYPES,
                           QTYPE_NAME_TO_ID, BLOCK_SIZES, BLOCK_BYTES, BPW)
from nsds_sensitivity import NSDSSensitivity, GroupSensitivity


# ---------------------------------------------------------------------------
# Quantization error estimation
# ---------------------------------------------------------------------------

# Approximate relative quantization error for each quant type.
# These are empirical estimates based on the BPW and quantization scheme.
# More accurate values can be computed by actual round-trip quantization
# (see quant_error.py), but these suffice for initial search.
#
# Key insight: for RANKING configurations (not predicting absolute PPL),
# we only need the error to be monotonically correct (lower BPW = higher error).
# The exact magnitude doesn't matter for search.

APPROX_RELATIVE_ERROR = {
    # type_name: approximate ||W - W_q||_F / ||W||_F
    "IQ1_S":    0.35,
    "IQ1_M":    0.30,
    "IQ2_XXS":  0.22,
    "IQ2_XS":   0.18,
    "IQ2_S":    0.15,
    "IQ3_XXS":  0.10,
    "IQ3_S":    0.08,
    "IQ4_NL":   0.055,
    "IQ4_XS":   0.050,
    "Q4_K":     0.040,
    "Q5_K":     0.025,
    "Q6_K":     0.015,
    "Q8_0":     0.005,
    "F16":      0.0001,
    "BF16":     0.0002,
    "F32":      0.0,
}


@dataclass
class QuantErrorEntry:
    """Per-group per-quant-type error metrics."""
    qtype: str
    frobenius_rel: float  # ||W - W_q||_F / ||W||_F
    mse: float = 0.0
    max_error: float = 0.0
    cosine_sim: float = 1.0


class QuantErrorDB:
    """Database of quantization errors per group per quant type.

    Can be populated with:
    1. Approximate estimates (fast, default)
    2. Actual round-trip measurements (accurate, via quant_error.py)
    """

    def __init__(self):
        self.errors: dict[str, dict[str, QuantErrorEntry]] = {}

    def populate_approximate(self, analyzer: GGUFAnalyzer,
                            sensitivity: NSDSSensitivity) -> None:
        """Populate with approximate errors based on BPW + kurtosis correction.

        Layers with higher kurtosis have more outliers, which means
        quantization error is amplified. We scale the base error by
        a kurtosis factor.
        """
        for gname, group in analyzer.groups.items():
            gs = sensitivity.get_group_sensitivity(gname)
            # Kurtosis correction: layers with high kurtosis have
            # disproportionately more error at low bit-widths
            kurt_factor = 1.0 + 0.1 * max(0, gs.nv_score - 0.5)

            self.errors[gname] = {}
            for qtype in SEARCH_QUANT_TYPES:
                base_err = APPROX_RELATIVE_ERROR.get(qtype, 0.1)
                adjusted_err = base_err * kurt_factor
                self.errors[gname][qtype] = QuantErrorEntry(
                    qtype=qtype,
                    frobenius_rel=adjusted_err,
                )

    def populate_from_measurements(self, path: str) -> None:
        """Load actual measured errors from quant_error.py output."""
        with open(path) as f:
            data = json.load(f)
        for gname, qtypes in data.items():
            self.errors[gname] = {}
            for qtype, metrics in qtypes.items():
                self.errors[gname][qtype] = QuantErrorEntry(
                    qtype=qtype,
                    frobenius_rel=metrics.get("frobenius_rel", 0.1),
                    mse=metrics.get("mse", 0.0),
                    max_error=metrics.get("max_error", 0.0),
                    cosine_sim=metrics.get("cosine_sim", 1.0),
                )

    def get_error(self, group_name: str, qtype: str) -> float:
        """Get relative Frobenius error for group at given quant type."""
        if group_name in self.errors and qtype in self.errors[group_name]:
            return self.errors[group_name][qtype].frobenius_rel
        return APPROX_RELATIVE_ERROR.get(qtype, 0.1)

    def save(self, path: str) -> None:
        """Save error database to JSON."""
        data = {}
        for gname, qtypes in self.errors.items():
            data[gname] = {
                qt: {"frobenius_rel": e.frobenius_rel, "mse": e.mse,
                     "max_error": e.max_error, "cosine_sim": e.cosine_sim}
                for qt, e in qtypes.items()
            }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Proxy model
# ---------------------------------------------------------------------------

class ProxyModel:
    """Fast proxy for quantization quality loss.

    Evaluates a configuration (group_name -> quant_type) and returns
    an estimated quality loss score. Lower score = better quality.

    Can evaluate ~100K configurations per second (pure Python dict lookups).
    """

    def __init__(self, analyzer: GGUFAnalyzer, sensitivity: NSDSSensitivity,
                 error_db: QuantErrorDB):
        self.analyzer = analyzer
        self.sensitivity = sensitivity
        self.error_db = error_db
        self.n_layers = analyzer.n_layers

        # Pre-compute static weights for each group
        self._role_weights = self._build_role_weights()
        self._depth_weights = self._build_depth_weights()
        self._size_weights = self._build_size_weights()
        self._sensitivity_scores = self._build_sensitivity_scores()

        # Combined static weight per group (product of role * depth * size * sensitivity)
        self._group_weight: dict[str, float] = {}
        for gname in analyzer.groups:
            self._group_weight[gname] = (
                self._role_weights.get(gname, 1.0) *
                self._depth_weights.get(gname, 1.0) *
                self._size_weights.get(gname, 1.0) *
                self._sensitivity_scores.get(gname, 0.5)
            )

        # Optional calibration parameters
        self._calibrated = False
        self._calib_slope = 1.0
        self._calib_intercept = 0.0

    def _build_role_weights(self) -> dict[str, float]:
        """Role-based importance weights.

        Based on empirical findings from RAMP, NSDS, and llama.cpp community:
        - Attention Q/K are critical (asymmetric head dim in Qwen3.5)
        - Shared experts are always-active, more important than routed
        - Norms are tiny but critical (always Q8_0 in practice)
        - Routed experts are numerous but individually less important
        """
        ROLE_IMPORTANCE = {
            # Full attention (10/40 layers)
            "attn_q": 1.8, "attn_k": 1.5, "attn_v": 1.2, "attn_o": 1.3,
            # GDN (Gated DeltaNet, 30/40 layers) — recurrent, critical
            "gdn_qkv": 1.6, "gdn_gate": 1.5,
            "gdn_q_norm": 2.5, "gdn_k_norm": 2.5,
            # SSM components — recurrent state, error accumulates over sequence
            "ssm_a": 2.0, "ssm_alpha": 1.8, "ssm_beta": 1.5,
            "ssm_conv": 1.3, "ssm_dt": 1.8, "ssm_norm": 2.5,
            "ssm_out": 1.4,
            "post_attn_norm": 2.5,
            # MoE FFN
            "shared_gate": 1.4, "shared_up": 1.1, "shared_down": 1.3,
            "expert_gate": 0.7, "expert_up": 0.6, "expert_down": 0.8,
            "moe_gate": 2.0,           # router is tiny but critical
            "shared_expert_gate": 1.5,  # scalar gate
            # Norms and globals
            "attn_norm": 2.5, "ffn_norm": 2.5,
            "output_norm": 3.0,
            "output": 2.0,             # lm_head
            "embed": 1.8,              # token embeddings
        }

        weights = {}
        for gname, group in self.analyzer.groups.items():
            role = group.role
            weights[gname] = ROLE_IMPORTANCE.get(role, 1.0)
        return weights

    def _build_depth_weights(self) -> dict[str, float]:
        """Depth-based importance weights.

        U-shaped curve: first and last layers are more sensitive.
        Based on RAMP transfer results (r>0.93 correlation across models).

        For Qwen3.5 with GDN layers (30/40 are GDN = recurrent, no KV cache),
        only 10 have full attention. The full-attention layers at network
        boundaries are especially critical.
        """
        weights = {}
        for gname, group in self.analyzer.groups.items():
            if group.layer_idx < 0:
                weights[gname] = 1.5  # global tensors
                continue
            d = group.layer_idx / max(self.n_layers - 1, 1)
            # U-shaped: 1.0 at center, ~1.5 at edges
            weights[gname] = 1.0 + 0.5 * (4 * (d - 0.5) ** 2)
        return weights

    def _build_size_weights(self) -> dict[str, float]:
        """Size-based weights: larger groups contribute more to overall quality.
        Normalized so average weight is ~1.0."""
        total_el = sum(g.total_elements for g in self.analyzer.groups.values())
        avg_el = total_el / max(len(self.analyzer.groups), 1)

        weights = {}
        for gname, group in self.analyzer.groups.items():
            # Log-scale to avoid extreme weights from 256-expert groups
            weights[gname] = np.log1p(group.total_elements / avg_el)
        return weights

    def _build_sensitivity_scores(self) -> dict[str, float]:
        """Extract NSDS scores for each group."""
        scores = {}
        for gname in self.analyzer.groups:
            gs = self.sensitivity.get_group_sensitivity(gname)
            scores[gname] = gs.nsds
        return scores

    def proxy_loss(self, config: dict[str, str]) -> float:
        """Compute proxy quality loss for a configuration.

        config: {group_name -> quant_type_name}
        Returns: estimated quality loss (lower = better)
        """
        total = 0.0
        for gname, qtype in config.items():
            w = self._group_weight.get(gname, 1.0)
            err = self.error_db.get_error(gname, qtype)
            total += w * err

        if self._calibrated:
            total = self._calib_slope * total + self._calib_intercept
        return total

    def proxy_loss_single(self, group_name: str, qtype: str) -> float:
        """Compute proxy loss contribution from a single group."""
        w = self._group_weight.get(group_name, 1.0)
        err = self.error_db.get_error(group_name, qtype)
        return w * err

    def total_size(self, config: dict[str, str]) -> int:
        """Total byte size for configuration."""
        return self.analyzer.total_size_for_config(config)

    def config_bpw(self, config: dict[str, str]) -> float:
        """Average bits per weight for configuration."""
        total_bits = 0
        total_elements = 0
        for gname, qtype in config.items():
            group = self.analyzer.groups[gname]
            qid = QTYPE_NAME_TO_ID[qtype]
            bpw = BLOCK_BYTES[qid] * 8.0 / BLOCK_SIZES[qid]
            total_bits += bpw * group.total_elements
            total_elements += group.total_elements
        return total_bits / max(total_elements, 1)

    def calibrate(self, proxy_scores: list, actual_ppls: list) -> None:
        """Calibrate proxy using actual perplexity measurements.

        Fits: actual_ppl = slope * proxy_score + intercept

        Args:
            proxy_scores: list of proxy_loss values for N configurations
            actual_ppls: list of actual perplexity values for same configs
        """
        if len(proxy_scores) < 3:
            print("WARNING: need at least 3 data points for calibration")
            return

        x = np.array(proxy_scores)
        y = np.array(actual_ppls)

        # Simple linear regression
        slope, intercept = np.polyfit(x, y, 1)
        residuals = y - (slope * x + intercept)
        r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

        self._calibrated = True
        self._calib_slope = float(slope)
        self._calib_intercept = float(intercept)

        print(f"Calibration: PPL = {slope:.4f} * proxy + {intercept:.4f}")
        print(f"R-squared: {r_squared:.4f}")
        if r_squared < 0.7:
            print("WARNING: low R-squared, proxy may not correlate well "
                  "with actual perplexity. Consider adding more measurements.")

    def report_config(self, config: dict[str, str], label: str = "") -> str:
        """Pretty-print a configuration with loss breakdown."""
        lines = []
        if label:
            lines.append(f"Configuration: {label}")
        lines.append(f"  Total size: {self.total_size(config) / (1024**3):.2f} GB")
        lines.append(f"  Average BPW: {self.config_bpw(config):.2f}")
        lines.append(f"  Proxy loss: {self.proxy_loss(config):.6f}")
        lines.append(f"")
        lines.append(f"  {'Group':<35} {'QType':<10} {'Loss':>8} {'Weight':>8}")
        lines.append(f"  {'-'*65}")

        contributions = []
        for gname, qtype in sorted(config.items()):
            loss = self.proxy_loss_single(gname, qtype)
            weight = self._group_weight.get(gname, 1.0)
            contributions.append((loss, gname, qtype, weight))

        contributions.sort(reverse=True)
        for loss, gname, qtype, weight in contributions[:25]:
            lines.append(f"  {gname:<35} {qtype:<10} {loss:>8.5f} {weight:>8.3f}")

        if len(contributions) > 25:
            lines.append(f"  ... and {len(contributions) - 25} more groups")

        return "\n".join(lines)
