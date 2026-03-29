#!/usr/bin/env python3
"""
RAMP-Local: Evolutionary Search for Optimal Mixed-Precision Configuration

Implements a population-based evolutionary search over GGUF quantization
configurations, guided by the proxy loss function.

Based on EvoPress (IST-DASLab) design but using our proxy instead of KL-div,
and operating on GGUF quant types directly.
"""

import random
import time
import json
import numpy as np
from typing import Optional
from copy import deepcopy

from gguf_analyzer import GGUFAnalyzer, SEARCH_QUANT_TYPES
from proxy_model import ProxyModel


class EvoSearch:
    """Evolutionary search for optimal quantization configuration."""

    def __init__(self, proxy: ProxyModel,
                 budget_bytes: int,
                 population_size: int = 128,
                 generations: int = 200,
                 elite_frac: float = 0.125,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.3,
                 quant_types: list = None,
                 seed: int = 42,
                 fixed_groups: dict = None):
        """
        Args:
            proxy: proxy model for fast evaluation
            budget_bytes: maximum total GGUF size in bytes
            population_size: number of configurations per generation
            generations: number of generations to evolve
            elite_frac: fraction of population to keep as elite
            mutation_rate: per-decision-point mutation probability
            crossover_rate: probability of crossover vs mutation
            quant_types: list of allowed quant types (default: SEARCH_QUANT_TYPES)
            seed: random seed for reproducibility
            fixed_groups: dict of {group_name: quant_type} that cannot be changed
                          (e.g., norms always Q8_0)
        """
        self.proxy = proxy
        self.budget = budget_bytes
        self.pop_size = population_size
        self.generations = generations
        self.elite_k = max(2, int(population_size * elite_frac))
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.quant_types = quant_types or list(SEARCH_QUANT_TYPES)
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Decision points: all groups except fixed ones
        self.fixed_groups = fixed_groups or {}
        all_groups = proxy.analyzer.list_decision_groups()
        self.decision_points = [g for g in all_groups
                               if g not in self.fixed_groups]

        # Sort quant types by BPW (ascending) for repair
        from gguf_analyzer import QTYPE_NAME_TO_ID, BLOCK_BYTES, BLOCK_SIZES
        self._bpw_order = sorted(
            self.quant_types,
            key=lambda qt: BLOCK_BYTES[QTYPE_NAME_TO_ID[qt]] * 8.0 /
                           BLOCK_SIZES[QTYPE_NAME_TO_ID[qt]]
        )

        # History for analysis
        self.history = []

    def _full_config(self, partial: dict) -> dict:
        """Merge partial config (decision points only) with fixed groups."""
        config = dict(self.fixed_groups)
        config.update(partial)
        return config

    def random_config(self) -> dict:
        """Generate random configuration for decision points."""
        config = {}
        for dp in self.decision_points:
            config[dp] = self.rng.choice(self.quant_types)
        return config

    def repair(self, config: dict) -> dict:
        """Repair configuration to meet budget constraint.

        Strategy: greedily downgrade the least-sensitive group with the
        highest current precision until within budget.
        """
        full = self._full_config(config)
        attempts = 0
        max_attempts = len(self.decision_points) * len(self.quant_types)

        while self.proxy.total_size(full) > self.budget and attempts < max_attempts:
            attempts += 1

            # Find group to downgrade: lowest sensitivity * highest current BPW
            best_candidate = None
            best_score = -float('inf')

            for dp in self.decision_points:
                qt = config[dp]
                idx = self._bpw_order.index(qt)
                if idx == 0:
                    continue  # already at minimum

                # Score: prefer downgrading low-sensitivity, high-BPW groups
                sens = self.proxy._sensitivity_scores.get(dp, 0.5)
                weight = self.proxy._group_weight.get(dp, 1.0)
                # Lower weight = less important = better candidate for downgrade
                score = (1.0 - weight / max(self.proxy._group_weight.values())) + \
                        idx / len(self._bpw_order)

                if score > best_score:
                    best_score = score
                    best_candidate = (dp, idx)

            if best_candidate is None:
                break  # all at minimum

            dp, idx = best_candidate
            config[dp] = self._bpw_order[idx - 1]
            full = self._full_config(config)

        return config

    def upgrade_remaining_budget(self, config: dict) -> dict:
        """If under budget, greedily upgrade most-sensitive groups.

        Fills remaining budget by upgrading groups with highest proxy impact.
        """
        full = self._full_config(config)
        remaining = self.budget - self.proxy.total_size(full)

        while remaining > 0:
            best_candidate = None
            best_improvement = -float('inf')
            best_size_cost = float('inf')

            for dp in self.decision_points:
                qt = config[dp]
                idx = self._bpw_order.index(qt)
                if idx >= len(self._bpw_order) - 1:
                    continue  # already at maximum

                new_qt = self._bpw_order[idx + 1]
                # Compute improvement and size cost
                old_loss = self.proxy.proxy_loss_single(dp, qt)
                new_loss = self.proxy.proxy_loss_single(dp, new_qt)
                improvement = old_loss - new_loss

                old_size = self.proxy.analyzer.group_byte_size(dp, qt)
                new_size = self.proxy.analyzer.group_byte_size(dp, new_qt)
                size_cost = new_size - old_size

                if size_cost <= remaining and size_cost > 0:
                    ratio = improvement / size_cost
                    if ratio > best_improvement:
                        best_improvement = ratio
                        best_candidate = (dp, new_qt, size_cost)

            if best_candidate is None:
                break

            dp, new_qt, cost = best_candidate
            config[dp] = new_qt
            remaining -= cost

        return config

    def mutate(self, config: dict) -> dict:
        """Mutate configuration with per-point probability."""
        new = dict(config)
        for dp in self.decision_points:
            if self.rng.random() < self.mutation_rate:
                new[dp] = self.rng.choice(self.quant_types)
        return self.repair(new)

    def crossover(self, a: dict, b: dict) -> dict:
        """Uniform crossover between two configurations."""
        child = {}
        for dp in self.decision_points:
            child[dp] = a[dp] if self.rng.random() < 0.5 else b[dp]
        return self.repair(child)

    def search(self, verbose: bool = True) -> tuple:
        """Run evolutionary search.

        Returns: (best_config, best_score, history)
            best_config includes both decision points and fixed groups
        """
        t0 = time.time()

        # Initialize population
        population = []
        for _ in range(self.pop_size):
            config = self.random_config()
            config = self.repair(config)
            population.append(config)

        best_config = None
        best_score = float('inf')
        stagnation = 0

        for gen in range(self.generations):
            # Evaluate
            scores = []
            for config in population:
                full = self._full_config(config)
                score = self.proxy.proxy_loss(full)
                scores.append((score, config))
            scores.sort(key=lambda x: x[0])

            # Track best
            gen_best_score, gen_best_config = scores[0]
            if gen_best_score < best_score:
                best_score = gen_best_score
                best_config = deepcopy(gen_best_config)
                stagnation = 0
            else:
                stagnation += 1

            # Record history
            gen_stats = {
                "generation": gen,
                "best_score": float(gen_best_score),
                "mean_score": float(np.mean([s for s, _ in scores])),
                "worst_score": float(scores[-1][0]),
                "global_best": float(best_score),
            }
            self.history.append(gen_stats)

            if verbose and (gen % 20 == 0 or gen == self.generations - 1):
                full_best = self._full_config(gen_best_config)
                size_gb = self.proxy.total_size(full_best) / (1024**3)
                bpw = self.proxy.config_bpw(full_best)
                elapsed = time.time() - t0
                print(f"Gen {gen:4d}: best={gen_best_score:.6f} "
                      f"global={best_score:.6f} "
                      f"size={size_gb:.2f}GB bpw={bpw:.2f} "
                      f"stag={stagnation} ({elapsed:.1f}s)")

            # Adaptive mutation: increase when stagnating
            effective_mutation = self.mutation_rate
            if stagnation > 20:
                effective_mutation = min(0.5, self.mutation_rate * 2)
            if stagnation > 50:
                effective_mutation = min(0.8, self.mutation_rate * 4)

            # Selection: elite + offspring
            elite = [config for _, config in scores[:self.elite_k]]
            offspring = []
            for _ in range(self.pop_size - self.elite_k):
                if self.rng.random() < self.crossover_rate:
                    p1, p2 = self.rng.sample(elite, 2)
                    child = self.crossover(p1, p2)
                else:
                    parent = self.rng.choice(elite)
                    child = self.mutate(parent)
                offspring.append(child)

            population = elite + offspring

        # Final: upgrade remaining budget in best config
        best_config = self.upgrade_remaining_budget(best_config)
        full_best = self._full_config(best_config)
        best_score = self.proxy.proxy_loss(full_best)

        elapsed = time.time() - t0
        if verbose:
            size_gb = self.proxy.total_size(full_best) / (1024**3)
            bpw = self.proxy.config_bpw(full_best)
            print(f"\nSearch complete in {elapsed:.1f}s")
            print(f"Best: score={best_score:.6f} size={size_gb:.2f}GB bpw={bpw:.2f}")
            print(f"Evaluations: {self.pop_size * self.generations}")

        return full_best, best_score, self.history


class GreedySearch:
    """ScaleBITS-style greedy search (fast, deterministic baseline)."""

    def __init__(self, proxy: ProxyModel, budget_bytes: int,
                 quant_types: list = None, fixed_groups: dict = None):
        self.proxy = proxy
        self.budget = budget_bytes
        self.quant_types = quant_types or list(SEARCH_QUANT_TYPES)
        self.fixed_groups = fixed_groups or {}

        all_groups = proxy.analyzer.list_decision_groups()
        self.decision_points = [g for g in all_groups
                               if g not in self.fixed_groups]

        from gguf_analyzer import QTYPE_NAME_TO_ID, BLOCK_BYTES, BLOCK_SIZES
        self._bpw_order = sorted(
            self.quant_types,
            key=lambda qt: BLOCK_BYTES[QTYPE_NAME_TO_ID[qt]] * 8.0 /
                           BLOCK_SIZES[QTYPE_NAME_TO_ID[qt]]
        )

    def search(self, verbose: bool = True) -> tuple:
        """Greedy upgrade from minimum quant until budget exhausted.

        Returns: (config, score)
        """
        t0 = time.time()

        # Start at lowest quant for all decision points
        config = {dp: self._bpw_order[0] for dp in self.decision_points}
        config.update(self.fixed_groups)

        step = 0
        while True:
            step += 1
            best_upgrade = None
            best_ratio = -float('inf')

            for dp in self.decision_points:
                idx = self._bpw_order.index(config[dp])
                if idx >= len(self._bpw_order) - 1:
                    continue

                new_qt = self._bpw_order[idx + 1]
                old_loss = self.proxy.proxy_loss_single(dp, config[dp])
                new_loss = self.proxy.proxy_loss_single(dp, new_qt)
                improvement = old_loss - new_loss

                old_size = self.proxy.analyzer.group_byte_size(dp, config[dp])
                new_size = self.proxy.analyzer.group_byte_size(dp, new_qt)
                cost = new_size - old_size
                if cost <= 0:
                    continue

                # Check budget
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
                print(f"Step {step}: score={score:.6f} size={size_gb:.2f}GB")

        score = self.proxy.proxy_loss(config)
        elapsed = time.time() - t0

        if verbose:
            size_gb = self.proxy.total_size(config) / (1024**3)
            bpw = self.proxy.config_bpw(config)
            print(f"Greedy search: {step} steps in {elapsed:.2f}s")
            print(f"Result: score={score:.6f} size={size_gb:.2f}GB bpw={bpw:.2f}")

        return config, score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAMP-Local: Search for optimal mixed-precision GGUF config")
    parser.add_argument("gguf_path", help="Path to GGUF file")
    parser.add_argument("--budget-gb", type=float, default=None,
                       help="Max size in GB (default: current file size)")
    parser.add_argument("--method", choices=["evo", "greedy", "both"],
                       default="both", help="Search method (default: both)")
    parser.add_argument("--pop-size", type=int, default=128)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--sensitivity-cache", default=None,
                       help="Path to cached sensitivity JSON")
    parser.add_argument("--output", "-o", default=None,
                       help="Output config JSON path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Phase 1: analyze GGUF
    print("=" * 70)
    print("Phase 1: GGUF Analysis")
    print("=" * 70)
    analyzer = GGUFAnalyzer(args.gguf_path)
    print(analyzer.summary())

    budget = int(args.budget_gb * 1024**3) if args.budget_gb else analyzer.file_size

    # Phase 2: compute sensitivity
    print("\n" + "=" * 70)
    print("Phase 2: NSDS Sensitivity Analysis")
    print("=" * 70)
    from nsds_sensitivity import NSDSSensitivity
    sens = NSDSSensitivity(analyzer)
    if args.sensitivity_cache:
        sens.load(args.sensitivity_cache)
        print(f"Loaded cached sensitivity from {args.sensitivity_cache}")
    else:
        sens.compute_all(verbose=True)
    print(sens.report(top_k=15))

    # Phase 3: build proxy
    print("\n" + "=" * 70)
    print("Phase 3: Building Proxy Model")
    print("=" * 70)
    from proxy_model import QuantErrorDB
    error_db = QuantErrorDB()
    error_db.populate_approximate(analyzer, sens)
    proxy = ProxyModel(analyzer, sens, error_db)

    # Fixed groups: norms always Q8_0
    fixed = {}
    for gname in analyzer.list_decision_groups():
        if "norm" in gname or "gate" in gname.split(".")[-1]:
            fixed[gname] = "Q8_0"

    # Current config baseline
    current = analyzer.current_config()
    current_score = proxy.proxy_loss(current)
    print(f"\nBaseline (current config): score={current_score:.6f} "
          f"size={analyzer.file_size / (1024**3):.2f}GB")

    # Phase 4: search
    print("\n" + "=" * 70)
    print("Phase 4: Search")
    print("=" * 70)

    results = {}

    if args.method in ("greedy", "both"):
        print("\n--- Greedy Search ---")
        greedy = GreedySearch(proxy, budget, fixed_groups=fixed)
        greedy_config, greedy_score = greedy.search(verbose=True)
        results["greedy"] = {"config": greedy_config, "score": greedy_score}
        print(proxy.report_config(greedy_config, "Greedy"))

    if args.method in ("evo", "both"):
        print("\n--- Evolutionary Search ---")
        evo = EvoSearch(proxy, budget,
                       population_size=args.pop_size,
                       generations=args.generations,
                       mutation_rate=args.mutation_rate,
                       seed=args.seed,
                       fixed_groups=fixed)
        evo_config, evo_score, history = evo.search(verbose=True)
        results["evo"] = {"config": evo_config, "score": evo_score}
        print(proxy.report_config(evo_config, "Evolutionary"))

    # Save best result
    if results:
        best_method = min(results, key=lambda m: results[m]["score"])
        best = results[best_method]
        print(f"\n{'='*70}")
        print(f"Best method: {best_method} (score={best['score']:.6f})")

        output = args.output or args.gguf_path + ".ramp-config.json"
        with open(output, 'w') as f:
            json.dump({
                "method": best_method,
                "score": best["score"],
                "budget_bytes": budget,
                "config": best["config"],
                "baseline_score": current_score,
                "improvement": current_score - best["score"],
            }, f, indent=2)
        print(f"Saved to {output}")
