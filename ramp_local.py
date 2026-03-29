#!/usr/bin/env python3
"""
RAMP-Local: Main Orchestrator

Full pipeline:
  1. Analyze GGUF structure
  2. Compute NSDS sensitivity (data-free)
  3. Build proxy model
  4. Run search (greedy + evolutionary)
  5. Generate llama-quantize command

Usage:
  python3 ramp_local.py <gguf_path> [options]

Example:
  python3 ramp_local.py ~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf \
    --budget-gb 14.71 --output configs/best.json
"""

import argparse
import json
import os
import sys
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gguf_analyzer import GGUFAnalyzer, SEARCH_QUANT_TYPES
from nsds_sensitivity import NSDSSensitivity
from proxy_model import ProxyModel, QuantErrorDB
from search_evo import EvoSearch, GreedySearch
from gguf_builder import GGUFBuilder


def main():
    parser = argparse.ArgumentParser(
        description="RAMP-Local: RL-guided mixed-precision GGUF quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with current model size as budget
  python3 ramp_local.py model.gguf

  # Specify budget (smaller = more aggressive compression)
  python3 ramp_local.py model.gguf --budget-gb 13.0

  # Use cached sensitivity (skip recomputation)
  python3 ramp_local.py model.gguf --sensitivity-cache cache/sensitivity.json

  # Generate build command for BF16 source
  python3 ramp_local.py model.gguf --build --input-bf16 model-bf16.gguf
""")

    parser.add_argument("gguf_path", help="Path to GGUF model file")
    parser.add_argument("--budget-gb", type=float, default=None,
                       help="Max output size in GB (default: match input size)")
    parser.add_argument("--method", choices=["evo", "greedy", "both"],
                       default="both", help="Search method")
    parser.add_argument("--pop-size", type=int, default=128,
                       help="Evolutionary population size")
    parser.add_argument("--generations", type=int, default=200,
                       help="Number of evolutionary generations")
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--expert-sample-k", type=int, default=16,
                       help="Experts to sample per layer for sensitivity")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sensitivity-cache", default=None,
                       help="Load/save sensitivity cache path")
    parser.add_argument("--error-cache", default=None,
                       help="Load measured quant errors from JSON")

    parser.add_argument("--output", "-o", default=None,
                       help="Output config JSON path")
    parser.add_argument("--build", action="store_true",
                       help="Also generate llama-quantize command")
    parser.add_argument("--input-bf16", default=None,
                       help="BF16 source GGUF for --build")
    parser.add_argument("--imatrix", default=None,
                       help="imatrix file for --build")

    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet

    t_start = time.time()

    # -----------------------------------------------------------------------
    # Phase 1: GGUF Analysis
    # -----------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("RAMP-Local: Phase 1 -- GGUF Analysis")
        print("=" * 70)

    analyzer = GGUFAnalyzer(args.gguf_path)
    if verbose:
        print(analyzer.summary())

    budget = int(args.budget_gb * 1024**3) if args.budget_gb else analyzer.file_size
    if verbose:
        print(f"\nBudget: {budget / (1024**3):.2f} GB")

    # -----------------------------------------------------------------------
    # Phase 2: Sensitivity Analysis
    # -----------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("RAMP-Local: Phase 2 -- NSDS Sensitivity (data-free)")
        print("=" * 70)

    sens = NSDSSensitivity(analyzer, expert_sample_k=args.expert_sample_k)

    cache_path = args.sensitivity_cache or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cache", "sensitivity.json")

    if os.path.exists(cache_path):
        sens.load(cache_path)
        if verbose:
            print(f"Loaded cached sensitivity from {cache_path}")
    else:
        sens.compute_all(verbose=verbose)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        sens.save(cache_path)
        if verbose:
            print(f"Saved sensitivity cache to {cache_path}")

    if verbose:
        print(sens.report(top_k=15))

    # -----------------------------------------------------------------------
    # Phase 3: Proxy Model
    # -----------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("RAMP-Local: Phase 3 -- Proxy Model")
        print("=" * 70)

    error_db = QuantErrorDB()
    if args.error_cache and os.path.exists(args.error_cache):
        error_db.populate_from_measurements(args.error_cache)
        if verbose:
            print(f"Loaded measured errors from {args.error_cache}")
    else:
        error_db.populate_approximate(analyzer, sens)
        if verbose:
            print("Using approximate quantization error estimates")

    proxy = ProxyModel(analyzer, sens, error_db)

    # Fixed groups: norms and gates always Q8_0 (tiny, critical)
    fixed = {}
    for gname in analyzer.list_decision_groups():
        group = analyzer.groups[gname]
        if "norm" in group.role:
            fixed[gname] = "Q8_0"
        elif group.role in ("moe_gate", "shared_expert_gate"):
            fixed[gname] = "Q8_0"

    # Baseline
    current = analyzer.current_config()
    current_score = proxy.proxy_loss(current)
    if verbose:
        print(f"\nBaseline: score={current_score:.6f} "
              f"size={analyzer.file_size / (1024**3):.2f}GB "
              f"bpw={proxy.config_bpw(current):.2f}")

    # -----------------------------------------------------------------------
    # Phase 4: Search
    # -----------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("RAMP-Local: Phase 4 -- Search")
        print("=" * 70)

    results = {}

    if args.method in ("greedy", "both"):
        if verbose:
            print("\n--- Greedy Search (ScaleBITS-style) ---")
        greedy = GreedySearch(proxy, budget, fixed_groups=fixed)
        g_config, g_score = greedy.search(verbose=verbose)
        results["greedy"] = {"config": g_config, "score": g_score}

    if args.method in ("evo", "both"):
        if verbose:
            print("\n--- Evolutionary Search ---")
        evo = EvoSearch(proxy, budget,
                       population_size=args.pop_size,
                       generations=args.generations,
                       mutation_rate=args.mutation_rate,
                       seed=args.seed,
                       fixed_groups=fixed)
        e_config, e_score, history = evo.search(verbose=verbose)
        results["evo"] = {"config": e_config, "score": e_score}

    # -----------------------------------------------------------------------
    # Phase 5: Results
    # -----------------------------------------------------------------------
    best_method = min(results, key=lambda m: results[m]["score"])
    best = results[best_method]

    improvement = (current_score - best["score"]) / current_score * 100 if current_score else 0.0

    if verbose:
        print("\n" + "=" * 70)
        print("RAMP-Local: Results")
        print("=" * 70)
        print(f"\nBest method: {best_method}")
        print(f"Proxy score: {best['score']:.6f} (baseline: {current_score:.6f})")
        print(f"Improvement: {improvement:.1f}%")
        print(proxy.report_config(best["config"], f"Best ({best_method})"))

    # Save config
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs", "best.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    builder = GGUFBuilder(analyzer, best["config"])

    save_data = {
        "method": best_method,
        "proxy_score": best["score"],
        "baseline_score": current_score,
        "improvement_pct": improvement,
        "budget_bytes": budget,
        "budget_gb": budget / (1024**3),
        "config": best["config"],
        "base_type": builder.find_base_type(),
        "gguf_source": args.gguf_path,
    }
    if "evo" in results:
        save_data["evo_score"] = results["evo"]["score"]
    if "greedy" in results:
        save_data["greedy_score"] = results["greedy"]["score"]

    with open(output, 'w') as f:
        json.dump(save_data, f, indent=2)
    if verbose:
        print(f"\nConfig saved to {output}")

    # Summary table
    if verbose:
        print("\n" + builder.generate_summary())

    # -----------------------------------------------------------------------
    # Phase 6: Build command (optional)
    # -----------------------------------------------------------------------
    if args.build:
        if not args.input_bf16:
            print("\nERROR: --build requires --input-bf16")
            sys.exit(1)

        output_gguf = args.gguf_path.replace(".gguf", "-ramp.gguf")
        cmd = builder.generate_command(
            args.input_bf16, output_gguf,
            imatrix_path=args.imatrix)

        if verbose:
            print("\n" + "=" * 70)
            print("llama-quantize command:")
            print("=" * 70)
            print(cmd)
            print(f"\nRun this command to build the optimized GGUF.")
            print(f"Then validate with: llama-perplexity -m {output_gguf} "
                  f"-f wiki.test.raw")

    elapsed = time.time() - t_start
    if verbose:
        print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
