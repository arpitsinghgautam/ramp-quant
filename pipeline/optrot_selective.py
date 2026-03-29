#!/usr/bin/env python3
"""
Selective OptRot: Orthogonal rotation for hybrid attention/GDN models.

Implements the OptRot algorithm (arXiv:2512.24124, NeurIPS 2025) SELECTIVELY:
- Rotates attention layers (10/40) and MoE expert FFN weights
- SKIPS all SSM/GDN tensors (ssm_*, gdn_qkv, gdn_gate for GDN layers)
- SKIPS all norm tensors (quantization-invariant, already near-uniform)

The rotation R minimizes sum(|R*W|^4) (element-wise 4th power), making the
weight distribution more uniform and reducing outliers that cause quantization
error. R is optimized on the Stiefel manifold via Cayley SGD.

For Qwen3.5-35B-A3B:
  - full_attention layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39
  - linear_attention (GDN) layers: all others (0-2, 4-6, 8-10, ...)
  - Only full_attention Q/K/V/O projections and all MoE expert weights get rotated
  - SSM recurrent tensors are NEVER touched (MambaQuant ICLR 2025: 21% accuracy drop)

Memory: processes layer-by-layer, peak ~4 GB for largest tensor + rotation matrix.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import gc
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.linalg import hadamard
except ImportError:
    hadamard = None

# -------------------------------------------------------------------------
# Model architecture constants
# -------------------------------------------------------------------------

# Qwen3.5-35B-A3B: layers where layer_types[i] == "full_attention"
# Pattern: every 4th layer starting from 3 (3, 7, 11, ..., 39)
FULL_ATTENTION_LAYERS = set(range(3, 40, 4))  # {3, 7, 11, 15, 19, 23, 27, 31, 35, 39}
NUM_LAYERS = 40
HIDDEN_DIM = 2048
NUM_EXPERTS = 256
MOE_INTERMEDIATE = 512  # per expert

# HF tensor name patterns for Qwen3.5
# Full attention layers have: self_attn.{q,k,v,o}_proj
# GDN layers have: linear_attn.{in_proj_qkv, in_proj_z, out_proj} + ssm_*
# All layers have: mlp.experts.{gate_up_proj, down_proj}, mlp.shared_expert.*,
#                  mlp.gate.weight, input_layernorm, post_attention_layernorm

PREFIX = "model.language_model.layers"


def is_rotatable_tensor(name: str) -> bool:
    """Determine if a HF tensor should be rotated.

    Rotate:
      - Full attention Q/K/V/O projections
      - MoE expert gate_up_proj and down_proj (all layers)
      - Shared expert gate/up/down_proj (all layers)
      - Embeddings and lm_head

    Skip:
      - All GDN/linear_attn tensors (in_proj_qkv, in_proj_z, out_proj)
      - All SSM tensors (ssm_*)
      - All norm tensors (*layernorm*, *norm*)
      - MoE router (gate.weight) -- tiny but structurally critical
      - Vision model tensors
      - MTP tensors
    """
    # Skip vision, MTP, non-language
    if "visual" in name or "mtp." in name:
        return False

    # Skip norms
    if "layernorm" in name or "norm" in name:
        return False

    # Skip SSM
    if "ssm_" in name:
        return False

    # Skip GDN (linear_attn) tensors entirely
    if "linear_attn" in name:
        return False

    # Skip MoE router (tiny, critical for routing decisions)
    if "mlp.gate.weight" in name:
        return False

    # Skip shared_expert_gate (scalar)
    if "shared_expert_gate" in name:
        return False

    # Rotate: full attention projections
    if "self_attn" in name and "proj" in name:
        return True

    # Rotate: expert FFN weights
    if "mlp.experts" in name:
        return True

    # Rotate: shared expert FFN weights
    if "mlp.shared_expert" in name and "proj" in name:
        return True

    # Rotate: embeddings and lm_head
    if "embed_tokens" in name or "lm_head" in name:
        return True

    return False


def rotation_group(name: str) -> Optional[str]:
    """Assign tensor to a rotation group (tensors in same group share R).

    Rotation pairs:
    - For attention layer L: R_L applied to Q,K,V (left-multiply) and O (right-multiply)
    - For MoE layer L: R_ffn_L applied to gate/up (left) and down (right)
    - Embeddings: R_0 (right-multiply)
    - lm_head: R_last^T (left-multiply)

    Returns group key or None if not rotatable.
    """
    if not is_rotatable_tensor(name):
        return None

    if "embed_tokens" in name:
        return "global.embed"
    if "lm_head" in name:
        return "global.lm_head"

    # Extract layer index
    import re
    m = re.search(r'layers\.(\d+)\.', name)
    if not m:
        return None
    layer = int(m.group(1))

    if "self_attn" in name:
        return f"layer.{layer}.attn"
    if "mlp.experts" in name or "mlp.shared_expert" in name:
        return f"layer.{layer}.ffn"

    return None


# -------------------------------------------------------------------------
# Hadamard initialization
# -------------------------------------------------------------------------

def random_hadamard(dim: int, rng: np.random.RandomState) -> np.ndarray:
    """Initialize rotation as random Hadamard matrix.

    For power-of-2 dimensions, uses Walsh-Hadamard.
    Otherwise, uses random orthogonal (QR of random Gaussian).
    """
    if hadamard is not None and (dim & (dim - 1)) == 0 and dim > 0:
        # Power of 2: use Hadamard with random sign flips
        H = hadamard(dim).astype(np.float64) / np.sqrt(dim)
        signs = rng.choice([-1, 1], size=dim).astype(np.float64)
        return H * signs[np.newaxis, :]
    else:
        # General case: random orthogonal via QR decomposition
        G = rng.randn(dim, dim).astype(np.float64)
        Q, _ = np.linalg.qr(G)
        return Q


# -------------------------------------------------------------------------
# OptRot: Cayley SGD on Stiefel manifold
# -------------------------------------------------------------------------

def optrot_objective(R: np.ndarray, W: np.ndarray) -> float:
    """Compute sum(|R @ W|^4) -- the L4 norm objective to minimize."""
    RW = R @ W
    return float(np.sum(RW ** 4))


def optrot_gradient(R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Gradient of sum(|R @ W|^4) w.r.t. R.

    d/dR sum((R*W)^4) = 4 * (R*W)^3 * W^T
    where (R*W)^3 is element-wise cube.
    """
    RW = R @ W
    return 4.0 * (RW ** 3) @ W.T


def cayley_retract(R: np.ndarray, grad: np.ndarray, tau: float) -> np.ndarray:
    """Cayley retraction on the Stiefel manifold.

    A = grad @ R^T - R @ grad^T   (skew-symmetric)
    R_new = (I + tau/2 * A)^(-1) @ (I - tau/2 * A) @ R
    """
    n = R.shape[0]
    A = grad @ R.T - R @ grad.T
    I = np.eye(n, dtype=R.dtype)
    half_tau_A = (tau / 2.0) * A
    # Solve (I + tau/2*A) @ R_new = (I - tau/2*A) @ R
    lhs = I + half_tau_A
    rhs = (I - half_tau_A) @ R
    R_new = np.linalg.solve(lhs, rhs)
    return R_new


def optimize_rotation(W_concat: np.ndarray,
                      lr: float = 1.5,
                      n_iter: int = 100,
                      seed: int = 42,
                      verbose: bool = False) -> np.ndarray:
    """Find optimal orthogonal rotation R minimizing sum(|R*W|^4).

    Args:
        W_concat: concatenated weight matrix [hidden_dim, total_cols]
                  (all tensors in the rotation group stacked column-wise)
        lr: initial learning rate (Cayley step size)
        n_iter: number of optimization iterations
        seed: random seed for initialization
        verbose: print progress

    Returns:
        R: optimal orthogonal rotation matrix [hidden_dim, hidden_dim]
    """
    dim = W_concat.shape[0]
    rng = np.random.RandomState(seed)

    # Initialize with random Hadamard
    R = random_hadamard(dim, rng)

    # Initial objective
    obj_init = optrot_objective(R, W_concat)
    if verbose:
        print(f"  OptRot init: L4={obj_init:.6e}, dim={dim}x{W_concat.shape[1]}")

    best_R = R.copy()
    best_obj = obj_init

    for it in range(n_iter):
        # Linear learning rate decay to 0
        tau = lr * (1.0 - it / n_iter)
        if tau < 1e-8:
            break

        # Compute gradient
        grad = optrot_gradient(R, W_concat)

        # Cayley retraction
        R = cayley_retract(R, grad, tau)

        # Track best
        obj = optrot_objective(R, W_concat)
        if obj < best_obj:
            best_obj = obj
            best_R = R.copy()

        if verbose and (it % 25 == 0 or it == n_iter - 1):
            reduction = (1.0 - obj / obj_init) * 100 if obj_init > 0 else 0
            print(f"    iter {it:3d}: L4={obj:.6e} ({reduction:+.1f}%)")

    if verbose:
        total_reduction = (1.0 - best_obj / obj_init) * 100 if obj_init > 0 else 0
        print(f"  OptRot done: L4 reduced {total_reduction:.1f}%")

    return best_R


# -------------------------------------------------------------------------
# Main rotation pipeline
# -------------------------------------------------------------------------

def load_tensor_from_shards(tensor_name: str, model_dir: Path,
                            index: dict) -> np.ndarray:
    """Load a single tensor from the correct safetensors shard."""
    import safetensors
    from safetensors import safe_open

    shard_file = index["weight_map"].get(tensor_name)
    if shard_file is None:
        raise KeyError(f"Tensor {tensor_name} not found in index")

    shard_path = model_dir / shard_file
    with safe_open(str(shard_path), framework="numpy") as f:
        return f.get_tensor(tensor_name).astype(np.float32)


def collect_rotation_groups(index: dict) -> dict:
    """Scan the weight map and organize tensors by rotation group.

    Returns: {group_key: [tensor_name, ...]}
    """
    groups = {}
    for name in index["weight_map"]:
        group = rotation_group(name)
        if group is not None:
            groups.setdefault(group, []).append(name)
    return groups


def apply_rotation_to_group(group_key: str, tensor_names: list,
                            model_dir: Path, index: dict,
                            output_dir: Path,
                            lr: float = 1.5, n_iter: int = 100,
                            seed: int = 42,
                            verbose: bool = True) -> np.ndarray:
    """Compute and apply OptRot to a group of tensors sharing one R.

    For attention groups (layer.L.attn):
      - Q, K, V: left-multiply by R  -> W_new = R @ W
      - O: right-multiply by R^T     -> W_new = W @ R^T

    For FFN groups (layer.L.ffn):
      - gate, up projections: left-multiply by R  -> W_new = R @ W
      - down projection: right-multiply by R^T    -> W_new = W @ R^T

    For embeddings:
      - embed_tokens: right-multiply by R  -> W_new = W @ R

    For lm_head:
      - lm_head: left-multiply by R^T  -> W_new = R^T @ W
      (or equivalently: right of embed R gets absorbed here)

    Returns the rotation matrix R for this group.
    """
    if verbose:
        print(f"\nGroup: {group_key} ({len(tensor_names)} tensors)")

    # Classify tensors by their rotation role
    left_tensors = []   # R @ W
    right_tensors = []  # W @ R^T

    for name in tensor_names:
        if "embed_tokens" in name:
            right_tensors.append(name)  # embed @ R
        elif "lm_head" in name:
            left_tensors.append(name)   # R^T @ W (but we treat as left with R^T)
        elif "self_attn" in name:
            if "o_proj" in name:
                right_tensors.append(name)  # O @ R^T
            else:
                left_tensors.append(name)   # R @ Q/K/V
        elif "down_proj" in name:
            right_tensors.append(name)  # down @ R^T
        else:
            left_tensors.append(name)   # R @ gate/up

    # Load all left-multiply tensors and concatenate for optimization
    # R is optimized over the concatenated matrix [W1 | W2 | ... | Wn]
    W_parts = []
    tensor_cache = {}
    for name in left_tensors:
        W = load_tensor_from_shards(name, model_dir, index)
        tensor_cache[name] = W
        # W shape is typically [out_features, in_features]
        # We want R to rotate the input dimension (columns of W^T = rows of W)
        W_parts.append(W.astype(np.float64))
        if verbose:
            print(f"  Left: {name} shape={W.shape}")

    # Also load right-multiply tensors (they constrain R from the other side)
    for name in right_tensors:
        W = load_tensor_from_shards(name, model_dir, index)
        tensor_cache[name] = W
        # For right-multiply: W @ R^T, so R operates on columns of W
        # Transpose to make it a left-multiply problem for optimization
        W_parts.append(W.T.astype(np.float64))
        if verbose:
            print(f"  Right: {name} shape={W.shape}")

    if not W_parts:
        if verbose:
            print("  No tensors to rotate, skipping")
        return np.eye(HIDDEN_DIM, dtype=np.float64)

    # Concatenate all weight matrices column-wise
    W_concat = np.concatenate(W_parts, axis=1)
    del W_parts
    gc.collect()

    # Determine rotation dimension from the first tensor
    dim = W_concat.shape[0]
    if verbose:
        print(f"  Optimizing R: {dim}x{dim} over {W_concat.shape[1]} columns")

    # Skip optimization for very small groups
    if dim < 4 or W_concat.shape[1] < dim:
        if verbose:
            print("  Too small for rotation, using identity")
        R = np.eye(dim, dtype=np.float64)
    else:
        R = optimize_rotation(W_concat, lr=lr, n_iter=n_iter,
                              seed=seed, verbose=verbose)
    del W_concat
    gc.collect()

    # Apply rotation and save
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in left_tensors:
        W = tensor_cache[name].astype(np.float64)
        # For lm_head: use R^T instead of R
        if "lm_head" in name:
            W_rot = (R.T @ W).astype(np.float32)
        else:
            W_rot = (R @ W).astype(np.float32)
        _save_tensor(output_dir, name, W_rot)
        del W, W_rot
        if verbose:
            print(f"  Saved rotated: {name}")

    for name in right_tensors:
        W = tensor_cache[name].astype(np.float64)
        if "embed_tokens" in name:
            W_rot = (W @ R).astype(np.float32)  # embed gets R (not R^T)
        else:
            W_rot = (W @ R.T).astype(np.float32)
        _save_tensor(output_dir, name, W_rot)
        del W, W_rot
        if verbose:
            print(f"  Saved rotated: {name}")

    del tensor_cache
    gc.collect()
    return R


def _save_tensor(output_dir: Path, tensor_name: str, data: np.ndarray):
    """Save a tensor as a safetensors shard (one tensor per file).

    Uses safetensors format for efficient loading by convert_hf_to_gguf.py.
    File name encodes the tensor name (dots replaced with --).
    """
    from safetensors.numpy import save_file

    safe_name = tensor_name.replace(".", "--")
    out_path = output_dir / f"{safe_name}.safetensors"
    save_file({tensor_name: data}, str(out_path))


def copy_unrotated_tensors(model_dir: Path, index: dict,
                           output_dir: Path, verbose: bool = True):
    """Copy tensors that are NOT rotated into the output directory unchanged.

    This ensures the output directory has all tensors needed for GGUF conversion.
    """
    from safetensors import safe_open
    from safetensors.numpy import save_file

    already_saved = set()
    for name in index["weight_map"]:
        if rotation_group(name) is not None:
            already_saved.add(name)

    # Group remaining tensors by shard for efficient loading
    shard_tensors = {}
    for name in index["weight_map"]:
        if name in already_saved:
            continue
        shard = index["weight_map"][name]
        shard_tensors.setdefault(shard, []).append(name)

    for shard_file, names in sorted(shard_tensors.items()):
        shard_path = model_dir / shard_file
        if verbose:
            print(f"  Copying {len(names)} tensors from {shard_file}")
        with safe_open(str(shard_path), framework="numpy") as f:
            for name in names:
                data = f.get_tensor(name)
                _save_tensor(output_dir, name, data)
                del data
        gc.collect()


def run_optrot(model_dir: str, output_dir: str,
               lr: float = 1.5, n_iter: int = 100,
               seed: int = 42,
               skip_copy: bool = False,
               verbose: bool = True) -> dict:
    """Run selective OptRot on a HuggingFace model directory.

    Args:
        model_dir: path to HF safetensors directory
        output_dir: path for rotated safetensors output
        lr: OptRot learning rate (Cayley step size)
        n_iter: optimization iterations per group
        seed: random seed
        skip_copy: if True, only save rotated tensors (not unrotated copies)
        verbose: print progress

    Returns:
        dict with rotation statistics
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load safetensors index
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No safetensors index at {index_path}")
    with open(index_path) as f:
        index = json.load(f)

    # Discover rotation groups
    groups = collect_rotation_groups(index)
    total_tensors = sum(len(v) for v in groups.values())

    if verbose:
        print(f"OptRot Selective Rotation")
        print(f"  Model: {model_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Groups: {len(groups)} rotation groups, {total_tensors} tensors")
        print(f"  Params: lr={lr}, n_iter={n_iter}, seed={seed}")
        print()

        # Show which layers are full attention vs GDN
        attn_groups = [k for k in groups if ".attn" in k]
        ffn_groups = [k for k in groups if ".ffn" in k]
        print(f"  Attention groups (rotated): {len(attn_groups)}")
        print(f"  FFN groups (rotated): {len(ffn_groups)}")
        skipped_gdn = [i for i in range(NUM_LAYERS) if i not in FULL_ATTENTION_LAYERS]
        print(f"  GDN layers (SKIPPED from attn rotation): {skipped_gdn[:5]}...")
        print()

    stats = {
        "groups_processed": 0,
        "tensors_rotated": 0,
        "tensors_copied": 0,
        "rotation_groups": {},
    }

    t0 = time.monotonic()

    # Process each rotation group
    for group_key in sorted(groups.keys()):
        tensor_names = groups[group_key]
        R = apply_rotation_to_group(
            group_key, tensor_names, model_dir, index, output_dir,
            lr=lr, n_iter=n_iter, seed=seed, verbose=verbose)

        stats["groups_processed"] += 1
        stats["tensors_rotated"] += len(tensor_names)
        stats["rotation_groups"][group_key] = {
            "n_tensors": len(tensor_names),
            "dim": int(R.shape[0]),
        }

    # Copy unrotated tensors
    if not skip_copy:
        if verbose:
            print(f"\nCopying unrotated tensors...")
        copy_unrotated_tensors(model_dir, index, output_dir, verbose=verbose)
        n_unrotated = len(index["weight_map"]) - total_tensors
        stats["tensors_copied"] = n_unrotated

    elapsed = time.monotonic() - t0
    stats["elapsed_s"] = elapsed

    if verbose:
        print(f"\nOptRot complete in {elapsed:.0f}s")
        print(f"  Rotated: {stats['tensors_rotated']} tensors in {stats['groups_processed']} groups")
        if not skip_copy:
            print(f"  Copied: {stats['tensors_copied']} unrotated tensors")

    # Save stats
    stats_path = output_dir / "optrot_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Selective OptRot: rotation-based outlier suppression for hybrid models")
    parser.add_argument("model_dir", help="HF safetensors model directory")
    parser.add_argument("output_dir", help="Output directory for rotated safetensors")
    parser.add_argument("--lr", type=float, default=1.5,
                        help="OptRot learning rate (default: 1.5)")
    parser.add_argument("--n-iter", type=int, default=100,
                        help="Optimization iterations per group (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-copy", action="store_true",
                        help="Only save rotated tensors (skip unrotated copies)")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    stats = run_optrot(
        args.model_dir, args.output_dir,
        lr=args.lr, n_iter=args.n_iter, seed=args.seed,
        skip_copy=args.skip_copy,
        verbose=not args.quiet)

    print(json.dumps(stats, indent=2))
