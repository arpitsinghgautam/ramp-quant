#!/usr/bin/env python3
"""
RAMP-Local: NSDS Data-Free Sensitivity Analysis

Computes per-tensor and per-group sensitivity scores using the NSDS framework
(arXiv:2603.17354) which requires NO calibration data:

1. Numerical Vulnerability (NV): excess kurtosis of weight distribution
   - Higher kurtosis = heavier tails = harder to quantize = more sensitive
2. Structural Expressiveness (SE): spectral magnitude * exp(spectral entropy)
   - Higher SE = more diverse information encoded = more sensitive to distortion
3. Combined NSDS score via MAD-sigmoid normalization + Soft-OR

Additionally computes RAMP-style features for optional SAC agent:
- mean, std, range, shape descriptors, normalized depth
"""

import numpy as np
from scipy import stats as scipy_stats
from scipy.sparse.linalg import svds
from dataclasses import dataclass
from typing import Optional
import json
import os
import sys
import time

# Import from sibling module
from gguf_analyzer import GGUFAnalyzer, DecisionGroup, BLOCK_SIZES, BLOCK_BYTES


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TensorSensitivity:
    """Sensitivity scores for a single tensor."""
    name: str
    # Numerical Vulnerability
    kurtosis: float       # excess kurtosis (Fisher)
    nv_score: float       # normalized NV score [0,1]
    # Structural Expressiveness
    svd_sum: float        # sum of singular values
    svd_entropy: float    # Shannon entropy of normalized singular values
    se_raw: float         # svd_sum * exp(svd_entropy)
    se_score: float       # normalized SE score [0,1]
    # Combined
    nsds: float           # Soft-OR(nv_score, se_score)
    # Additional stats (for RAMP-style embedding)
    mean: float
    std: float
    weight_range: float
    n_elements: int


@dataclass
class GroupSensitivity:
    """Aggregated sensitivity for a decision group."""
    name: str
    nsds: float           # aggregated NSDS score
    nv_score: float       # aggregated NV score
    se_score: float       # aggregated SE score
    n_tensors: int
    total_elements: int


# ---------------------------------------------------------------------------
# Dequantization helpers
# ---------------------------------------------------------------------------

def dequant_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    """Vectorized Q8_0 dequantization: 34 bytes/block (2 f16 scale + 32 int8)."""
    BLOCK_BYTES = 34
    n_blocks = (n_elements + 31) // 32
    arr = np.frombuffer(data[:n_blocks * BLOCK_BYTES], dtype=np.uint8)
    blocks = arr.reshape(n_blocks, BLOCK_BYTES)
    scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    vals = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (vals * scales).flatten()[:n_elements]


def dequant_q4_k(data: bytes, n_elements: int) -> np.ndarray:
    """Vectorized approximate Q4_K dequantization: 144 bytes/block, 256 elements."""
    BPB = 144
    n_blocks = (n_elements + 255) // 256
    arr = np.frombuffer(data[:n_blocks * BPB], dtype=np.uint8)
    blocks = arr.reshape(n_blocks, BPB)
    # Extract d scale (first 2 bytes of each block)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    # Extract quant nibbles (bytes 16..143 = 128 bytes = 256 nibbles)
    qbytes = blocks[:, 16:144]  # (n_blocks, 128)
    low = (qbytes & 0x0F).astype(np.float32) - 8.0   # (n_blocks, 128)
    high = (qbytes >> 4).astype(np.float32) - 8.0     # (n_blocks, 128)
    vals = np.empty((n_blocks, 256), dtype=np.float32)
    vals[:, 0::2] = low
    vals[:, 1::2] = high
    return (vals * d).flatten()[:n_elements]


def dequant_q5_k(data: bytes, n_elements: int) -> np.ndarray:
    """Vectorized approximate Q5_K dequantization: 176 bytes/block, 256 elements."""
    BPB = 176
    n_blocks = (n_elements + 255) // 256
    expected = n_blocks * BPB
    if len(data) < expected:
        return np.zeros(n_elements, dtype=np.float32)
    arr = np.frombuffer(data[:expected], dtype=np.uint8)
    blocks = arr.reshape(n_blocks, BPB)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    # Q5_K: bytes 32..175 = 144 bytes (128 q4 nibbles + 32 high-bit bytes)
    # Approximate: use q4 nibbles with d scale (sufficient for sensitivity)
    qbytes = blocks[:, 32:160]  # 128 bytes of q4 nibbles
    low = (qbytes & 0x0F).astype(np.float32) - 16.0
    high = (qbytes >> 4).astype(np.float32) - 16.0
    vals = np.empty((n_blocks, 256), dtype=np.float32)
    vals[:, 0::2] = low
    vals[:, 1::2] = high
    return (vals * d).flatten()[:n_elements]


def dequant_q6_k(data: bytes, n_elements: int) -> np.ndarray:
    """Vectorized approximate Q6_K dequantization: 210 bytes/block, 256 elements."""
    BPB = 210
    n_blocks = (n_elements + 255) // 256
    expected = n_blocks * BPB
    if len(data) < expected:
        return np.zeros(n_elements, dtype=np.float32)
    arr = np.frombuffer(data[:expected], dtype=np.uint8)
    blocks = arr.reshape(n_blocks, BPB)
    # d scale at byte 208 (f16)
    d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    # Low 4 bits from first 128 bytes
    qbytes = blocks[:, :128]
    low = (qbytes & 0x0F).astype(np.float32) - 32.0
    high = (qbytes >> 4).astype(np.float32) - 32.0
    vals = np.empty((n_blocks, 256), dtype=np.float32)
    vals[:, 0::2] = low
    vals[:, 1::2] = high
    return (vals * d).flatten()[:n_elements]


def extract_block_scales(data: bytes, n_elements: int, bpb: int,
                         scale_offset: int = 0) -> np.ndarray:
    """Extract f16 block scales from any GGUF block format.

    For sensitivity analysis, the distribution of block scales captures the
    weight heterogeneity without needing full dequantization. Works for
    IQ3_S, IQ2_*, IQ4_*, Q3_K, etc.
    """
    block_size = 256  # all K-quant and IQ formats use 256-element superblocks
    n_blocks = (n_elements + block_size - 1) // block_size
    expected = n_blocks * bpb
    usable = min(len(data), expected)
    if usable < bpb:
        return np.zeros(1, dtype=np.float32)
    actual_blocks = usable // bpb
    arr = np.frombuffer(data[:actual_blocks * bpb], dtype=np.uint8)
    blocks = arr.reshape(actual_blocks, bpb)
    d = blocks[:, scale_offset:scale_offset + 2].copy().view(np.float16).astype(np.float32).flatten()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return d


# Block format specs: (bytes_per_block, scale_offset)
BLOCK_SCALE_INFO = {
    "IQ3_S": (110, 0), "IQ3_XXS": (98, 0), "IQ3_M": (110, 0),
    "IQ2_XXS": (66, 0), "IQ2_XS": (74, 0), "IQ2_S": (82, 0),
    "IQ4_XS": (136, 0), "IQ4_NL": (18, 0),
    "Q3_K": (110, 0), "Q2_K": (84, 0),
}


# Maximum elements to process for sensitivity (1M is sufficient for stable statistics)
SENSITIVITY_SAMPLE_SIZE = 1_000_000


def dequant_generic(data: bytes, n_elements: int, qtype_name: str) -> np.ndarray:
    """Approximate dequantization for sensitivity analysis.

    For sensitivity ranking we need distribution SHAPE (kurtosis, SVD spectrum),
    not exact values. All routines are vectorized (no Python loops).
    """
    if qtype_name == "F32":
        return np.frombuffer(data[:n_elements * 4], dtype=np.float32).copy()
    elif qtype_name == "F16":
        return np.frombuffer(data[:n_elements * 2], dtype=np.float16).astype(np.float32)
    elif qtype_name == "BF16":
        raw = np.frombuffer(data[:n_elements * 2], dtype=np.uint16)
        f32_int = raw.astype(np.uint32) << 16
        return f32_int.view(np.float32).copy()
    elif qtype_name == "Q8_0":
        return dequant_q8_0(data, n_elements)
    elif qtype_name in ("Q4_K", "Q4_K_M", "Q4_K_S"):
        return dequant_q4_k(data, n_elements)
    elif qtype_name in ("Q5_K", "Q5_K_M", "Q5_K_S", "Q5_K_XL"):
        return dequant_q5_k(data, n_elements)
    elif qtype_name in ("Q6_K",):
        return dequant_q6_k(data, n_elements)
    elif qtype_name in BLOCK_SCALE_INFO:
        # For complex quant formats: return block scales directly
        # (used only for sensitivity ranking, not for exact weight reconstruction)
        bpb, soff = BLOCK_SCALE_INFO[qtype_name]
        return extract_block_scales(data, n_elements, bpb, soff)
    else:
        # Unknown format: extract scale from header
        if len(data) >= 4:
            rough_scale = abs(np.frombuffer(data[:2], dtype=np.float16)[0])
            if rough_scale == 0 or np.isnan(rough_scale):
                rough_scale = 0.01
        else:
            rough_scale = 0.01
        return np.array([rough_scale], dtype=np.float32)


# ---------------------------------------------------------------------------
# Core sensitivity computations
# ---------------------------------------------------------------------------

def compute_kurtosis(W: np.ndarray) -> float:
    """Compute excess kurtosis (Fisher definition, normal = 0)."""
    flat = W.flatten()
    if flat.size < 4:
        return 0.0
    # Use scipy for numerical stability
    k = scipy_stats.kurtosis(flat, fisher=True, bias=False)
    return float(np.clip(k, -3, 1000))  # clip extreme values


def compute_structural_expressiveness(W: np.ndarray, max_rank: int = 64) -> tuple:
    """Compute spectral-based Structural Expressiveness.

    Returns: (svd_sum, svd_entropy, SE_raw)

    For efficiency, uses randomized SVD with max_rank components.
    This captures the dominant spectral structure which is what matters
    for sensitivity analysis.
    """
    # Reshape to 2D for SVD
    if W.ndim == 1:
        side = int(np.sqrt(W.size))
        if side * side == W.size:
            W2d = W.reshape(side, side)
        else:
            W2d = W.reshape(-1, 1)
    elif W.ndim == 2:
        W2d = W
    elif W.ndim == 3:
        # Expert tensors: (n_experts, in, out) -> flatten experts
        W2d = W.reshape(-1, W.shape[-1])
    else:
        W2d = W.reshape(-1, W.shape[-1])

    m, n = W2d.shape
    k = min(max_rank, min(m, n) - 1)
    if k < 1:
        return 0.0, 0.0, 0.0

    try:
        if min(m, n) <= max_rank + 5:
            # Full SVD for small matrices
            s = np.linalg.svd(W2d, compute_uv=False)
        else:
            # Randomized SVD for large matrices
            _, s, _ = svds(W2d.astype(np.float64), k=k)
            s = np.sort(s)[::-1]  # descending
    except Exception:
        return 0.0, 0.0, 0.0

    s = np.abs(s)
    s_sum = float(s.sum())
    if s_sum < 1e-12:
        return 0.0, 0.0, 0.0

    # Normalized singular values for entropy
    s_norm = s / s_sum
    # Avoid log(0)
    s_norm = s_norm[s_norm > 1e-12]
    entropy = float(-np.sum(s_norm * np.log(s_norm)))

    se = s_sum * np.exp(entropy)
    return s_sum, entropy, float(se)


def mad_sigmoid_normalize(values: np.ndarray) -> np.ndarray:
    """MAD-sigmoid normalization from NSDS paper.

    Robust Z-score using Median Absolute Deviation, then sigmoid.
    Maps to [0, 1] range with robustness to outliers.
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-12:
        mad = np.std(values)
    if mad < 1e-12:
        return np.full_like(values, 0.5)

    z = (values - median) / (1.4826 * mad)  # 1.4826 = consistency constant
    return 1.0 / (1.0 + np.exp(-z))


def soft_or(scores: list) -> float:
    """Soft-OR operator: 1 - prod(1 - s_i).
    Emphasizes the most sensitive dimension."""
    product = 1.0
    for s in scores:
        product *= (1.0 - np.clip(s, 0, 0.999))
    return 1.0 - product


# ---------------------------------------------------------------------------
# Main sensitivity analyzer
# ---------------------------------------------------------------------------

class NSDSSensitivity:
    """Compute NSDS data-free sensitivity for all tensors in a GGUF."""

    def __init__(self, analyzer: GGUFAnalyzer, expert_sample_k: int = 16):
        """
        Args:
            analyzer: parsed GGUF structure
            expert_sample_k: for expert groups, sample K experts instead of all 256
        """
        self.analyzer = analyzer
        self.expert_sample_k = expert_sample_k
        self.tensor_scores: dict[str, TensorSensitivity] = {}
        self.group_scores: dict[str, GroupSensitivity] = {}

    def compute_tensor_sensitivity(self, tensor_name: str) -> TensorSensitivity:
        """Compute sensitivity for a single tensor."""
        ti = self.analyzer.tensors[tensor_name]
        data = self.analyzer.read_tensor_data(tensor_name)
        W = dequant_generic(data, ti.n_elements, ti.qtype_name)
        del data  # free raw bytes immediately

        # For very large tensors, subsample for statistics
        # 1M values gives stable kurtosis/SVD estimates (within ~2% of full)
        if W.size > SENSITIVITY_SAMPLE_SIZE:
            rng = np.random.RandomState(hash(tensor_name) % 2**31)
            idx = rng.choice(W.size, SENSITIVITY_SAMPLE_SIZE, replace=False)
            W_sample = W[idx]
        else:
            W_sample = W

        # Numerical Vulnerability (fast: O(n))
        kurt = compute_kurtosis(W_sample)

        # Structural Expressiveness (slow: SVD)
        # Skip SVD for very large tensors (>10M elements) — use kurtosis only
        if ti.n_elements > 10_000_000:
            svd_sum, svd_entropy, se_raw = 0.0, 0.0, 0.0
        else:
            svd_sum, svd_entropy, se_raw = compute_structural_expressiveness(W)

        # Basic statistics (on sample for speed)
        mean_val = float(np.mean(W_sample))
        std_val = float(np.std(W_sample))
        range_val = float(np.ptp(W_sample))
        del W, W_sample  # free memory

        return TensorSensitivity(
            name=tensor_name,
            kurtosis=kurt, nv_score=0.0,  # normalized later
            svd_sum=svd_sum, svd_entropy=svd_entropy,
            se_raw=se_raw, se_score=0.0,  # normalized later
            nsds=0.0,  # computed after normalization
            mean=mean_val, std=std_val,
            weight_range=range_val, n_elements=ti.n_elements,
        )

    def compute_all(self, verbose: bool = True) -> None:
        """Compute sensitivity for all decision groups."""
        t0 = time.time()

        # Step 1: compute raw scores per tensor
        # For expert groups, sample K experts to save time
        tensors_to_analyze = []
        for gname, group in self.analyzer.groups.items():
            if "expert" in group.role and len(group.tensor_names) > self.expert_sample_k:
                # Sample K representative tensors
                rng = np.random.RandomState(42)
                sampled = rng.choice(group.tensor_names,
                                    size=self.expert_sample_k, replace=False)
                tensors_to_analyze.extend(sampled)
            else:
                tensors_to_analyze.extend(group.tensor_names)

        if verbose:
            print(f"Analyzing {len(tensors_to_analyze)} tensors "
                  f"(from {len(self.analyzer.tensors)} total)...")

        for i, tname in enumerate(tensors_to_analyze):
            if verbose and (i % 50 == 0 or i == len(tensors_to_analyze) - 1):
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(tensors_to_analyze) - i - 1)
                print(f"  [{i+1}/{len(tensors_to_analyze)}] "
                      f"{tname[:60]}... ({elapsed:.0f}s, ETA {eta:.0f}s)")
            self.tensor_scores[tname] = self.compute_tensor_sensitivity(tname)

        # Step 2: MAD-sigmoid normalization across all tensors
        all_kurt = np.array([s.kurtosis for s in self.tensor_scores.values()])
        all_se = np.array([s.se_raw for s in self.tensor_scores.values()])

        nv_normalized = mad_sigmoid_normalize(all_kurt)
        se_normalized = mad_sigmoid_normalize(all_se)

        for (name, score), nv, se in zip(self.tensor_scores.items(),
                                          nv_normalized, se_normalized):
            score.nv_score = float(nv)
            score.se_score = float(se)
            score.nsds = soft_or([score.nv_score, score.se_score])

        # Step 3: aggregate to group level
        for gname, group in self.analyzer.groups.items():
            # Find analyzed tensors in this group
            analyzed = [self.tensor_scores[t] for t in group.tensor_names
                       if t in self.tensor_scores]
            if not analyzed:
                # Use group average from sampled tensors
                self.group_scores[gname] = GroupSensitivity(
                    name=gname, nsds=0.5, nv_score=0.5, se_score=0.5,
                    n_tensors=len(group.tensor_names),
                    total_elements=group.total_elements)
                continue

            # Weighted average by element count
            total_el = sum(s.n_elements for s in analyzed)
            if total_el == 0:
                total_el = 1

            avg_nsds = sum(s.nsds * s.n_elements for s in analyzed) / total_el
            avg_nv = sum(s.nv_score * s.n_elements for s in analyzed) / total_el
            avg_se = sum(s.se_score * s.n_elements for s in analyzed) / total_el

            self.group_scores[gname] = GroupSensitivity(
                name=gname,
                nsds=avg_nsds, nv_score=avg_nv, se_score=avg_se,
                n_tensors=len(group.tensor_names),
                total_elements=group.total_elements,
            )

        elapsed = time.time() - t0
        if verbose:
            print(f"Sensitivity analysis complete in {elapsed:.1f}s")

    def get_group_sensitivity(self, group_name: str) -> GroupSensitivity:
        """Get sensitivity score for a decision group."""
        return self.group_scores.get(group_name,
            GroupSensitivity(name=group_name, nsds=0.5, nv_score=0.5,
                           se_score=0.5, n_tensors=0, total_elements=0))

    def save(self, path: str) -> None:
        """Save sensitivity scores to JSON."""
        data = {
            "tensor_scores": {
                name: {
                    "kurtosis": s.kurtosis, "nv_score": s.nv_score,
                    "svd_sum": s.svd_sum, "svd_entropy": s.svd_entropy,
                    "se_raw": s.se_raw, "se_score": s.se_score,
                    "nsds": s.nsds, "mean": s.mean, "std": s.std,
                    "weight_range": s.weight_range,
                    "n_elements": s.n_elements,
                }
                for name, s in self.tensor_scores.items()
            },
            "group_scores": {
                name: {
                    "nsds": s.nsds, "nv_score": s.nv_score,
                    "se_score": s.se_score,
                    "n_tensors": s.n_tensors,
                    "total_elements": s.total_elements,
                }
                for name, s in self.group_scores.items()
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load cached sensitivity scores from JSON."""
        with open(path) as f:
            data = json.load(f)

        for name, d in data.get("tensor_scores", {}).items():
            self.tensor_scores[name] = TensorSensitivity(name=name, **d)

        for name, d in data.get("group_scores", {}).items():
            self.group_scores[name] = GroupSensitivity(name=name, **d)

    def report(self, top_k: int = 20) -> str:
        """Print sensitivity report."""
        lines = [
            "=" * 70,
            "NSDS Sensitivity Report",
            "=" * 70,
            "",
            f"{'Group':<40} {'NSDS':>6} {'NV':>6} {'SE':>6} {'Elements':>12}",
            "-" * 70,
        ]

        sorted_groups = sorted(self.group_scores.items(),
                              key=lambda x: x[1].nsds, reverse=True)

        for gname, gs in sorted_groups[:top_k]:
            el_str = f"{gs.total_elements/1e6:.1f}M"
            lines.append(f"{gname:<40} {gs.nsds:>6.3f} {gs.nv_score:>6.3f} "
                        f"{gs.se_score:>6.3f} {el_str:>12}")

        if len(sorted_groups) > top_k:
            lines.append(f"  ... and {len(sorted_groups) - top_k} more groups")

        lines.append("")
        lines.append("Top 5 most sensitive (need high precision):")
        for gname, gs in sorted_groups[:5]:
            lines.append(f"  {gname}: NSDS={gs.nsds:.3f}")

        lines.append("")
        lines.append("Top 5 least sensitive (can use low precision):")
        for gname, gs in sorted_groups[-5:]:
            lines.append(f"  {gname}: NSDS={gs.nsds:.3f}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NSDS data-free sensitivity analysis for GGUF models")
    parser.add_argument("gguf_path", help="Path to GGUF file")
    parser.add_argument("--output", "-o", default=None,
                       help="Output JSON path (default: <gguf>.sensitivity.json)")
    parser.add_argument("--expert-sample-k", type=int, default=16,
                       help="Number of experts to sample per layer (default: 16)")
    parser.add_argument("--top-k", type=int, default=30,
                       help="Number of groups to show in report (default: 30)")
    parser.add_argument("--load-cache", default=None,
                       help="Load cached sensitivity scores instead of computing")
    args = parser.parse_args()

    analyzer = GGUFAnalyzer(args.gguf_path)
    print(analyzer.summary())
    print()

    sens = NSDSSensitivity(analyzer, expert_sample_k=args.expert_sample_k)

    if args.load_cache:
        sens.load(args.load_cache)
        print(f"Loaded cached scores from {args.load_cache}")
    else:
        sens.compute_all(verbose=True)

    print()
    print(sens.report(top_k=args.top_k))

    output = args.output or args.gguf_path + ".sensitivity.json"
    sens.save(output)
    print(f"\nSaved to {output}")
