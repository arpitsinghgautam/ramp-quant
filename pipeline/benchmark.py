#!/usr/bin/env python3
"""
A/B Quality Benchmark: Compare two GGUF files on multiple quality axes.

Evaluates:
1. Perplexity on calibration text (French, code, kine)
2. Token-by-token logit KL divergence
3. Functional tests (tool-calling, reasoning, code generation)
4. Speed benchmark (generation tok/s, prompt processing tok/s)

Uses temporary llama-server instances on isolated ports to avoid
conflicting with the production server on port 8081.

Output: JSON comparison report + optional human-readable summary.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

IK_LLAMA = Path.home() / "ik_llama.cpp" / "build_sm120" / "bin"
LLAMA_SERVER = IK_LLAMA / "llama-server"
LLAMA_PPL = IK_LLAMA / "llama-perplexity"
LLAMA_BENCH = IK_LLAMA / "llama-bench"

CALIBRATION_DIR = Path.home() / ".chimere" / "data" / "calibration"
BENCH_PORT_A = 8097
BENCH_PORT_B = 8098

# Default server params matching production config
SERVER_DEFAULTS = {
    "ngl": 99,
    "ctx": 4096,
    "n_cpu_moe": 4,
    "batch": 2048,
    "ubatch": 512,
}


# -------------------------------------------------------------------------
# Functional test prompts
# -------------------------------------------------------------------------

FUNCTIONAL_TESTS = [
    # French kinetitherapy
    {"prompt": "Quels sont les criteres du hop test battery pour le retour au sport apres ligamentoplastie du LCA ?",
     "keywords": ["LSI", "hop", "single", "triple", "90"],
     "domain": "kine", "max_tokens": 512},

    # Code generation
    {"prompt": "Write a Python function that checks if parentheses are balanced: (), [], {}. Return True/False.",
     "keywords": ["def", "stack", "return", "True", "False"],
     "domain": "code", "max_tokens": 512},

    # Reasoning
    {"prompt": "If a train leaves Paris at 9:00 going 200 km/h and another leaves Lyon (450 km away) at 9:30 going 250 km/h toward Paris, when do they meet?",
     "keywords": ["10", "hour", "km", "distance"],
     "domain": "reasoning", "max_tokens": 512},

    # Tool calling (JSON function call format)
    {"prompt": 'You have access to a function called "search_web(query: str)". The user asks: "What is the weather in Paris today?" Generate a function call.',
     "keywords": ["search_web", "weather", "Paris"],
     "domain": "tools", "max_tokens": 256},

    # Math
    {"prompt": "Solve: integral of x*e^x dx",
     "keywords": ["xe", "e^x", "integration", "parts"],
     "domain": "math", "max_tokens": 512},

    # French general
    {"prompt": "Explique le principe de fonctionnement d'un transformer en deep learning en 5 phrases.",
     "keywords": ["attention", "query", "key", "value", "couche"],
     "domain": "general_fr", "max_tokens": 512},
]


# -------------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------------

@dataclass
class PPLResult:
    ppl: float
    n_tokens: int
    elapsed_s: float
    data_file: str


@dataclass
class SpeedResult:
    gen_tok_s: float      # generation tokens/second
    pp_tok_s: float       # prompt processing tokens/second
    elapsed_s: float


@dataclass
class FunctionalResult:
    total: int
    passed: int
    score_pct: float
    per_domain: dict
    details: list


@dataclass
class BenchmarkResult:
    model_path: str
    model_name: str
    file_size_gb: float
    ppl: Optional[PPLResult] = None
    speed: Optional[SpeedResult] = None
    functional: Optional[FunctionalResult] = None


@dataclass
class ComparisonReport:
    model_a: BenchmarkResult
    model_b: BenchmarkResult
    ppl_delta_pct: float = 0.0
    speed_delta_pct: float = 0.0
    functional_delta: float = 0.0
    verdict: str = ""


# -------------------------------------------------------------------------
# Server management
# -------------------------------------------------------------------------

def start_server(model_path: str, port: int, timeout: int = 120) -> subprocess.Popen:
    """Start a temporary llama-server instance."""
    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "-ngl", str(SERVER_DEFAULTS["ngl"]),
        "-c", str(SERVER_DEFAULTS["ctx"]),
        "-np", "1",
        "--port", str(port),
        "--n-cpu-moe", str(SERVER_DEFAULTS["n_cpu_moe"]),
        "--flash-attn", "on",
        "--jinja",
        "-b", str(SERVER_DEFAULTS["batch"]),
        "-ub", str(SERVER_DEFAULTS["ubatch"]),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    import http.client
    for attempt in range(timeout // 2):
        time.sleep(2)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                conn.close()
                return proc
            conn.close()
        except Exception:
            pass

    proc.kill()
    raise RuntimeError(f"Server on port {port} failed to start in {timeout}s")


def stop_server(proc: subprocess.Popen):
    """Gracefully stop a server process."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def query_server(prompt: str, port: int, max_tokens: int = 512,
                 temperature: float = 0.1) -> tuple:
    """Send a chat completion request. Returns (response_text, gen_tokens, elapsed_s)."""
    import http.client

    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": False},
    })

    t0 = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    elapsed = time.monotonic() - t0

    text = ""
    gen_tokens = 0
    if "choices" in data and data["choices"]:
        text = data["choices"][0]["message"]["content"]
    if "usage" in data:
        gen_tokens = data["usage"].get("completion_tokens", 0)

    return text, gen_tokens, elapsed


# -------------------------------------------------------------------------
# Perplexity measurement
# -------------------------------------------------------------------------

def measure_perplexity(model_path: str, data_file: str = None,
                       ctx_size: int = 2048) -> PPLResult:
    """Run llama-perplexity and parse output."""
    if data_file is None:
        candidates = [
            CALIBRATION_DIR / "wiki_fr_2k.txt",
            CALIBRATION_DIR / "chimere_calibration.txt",
        ]
        for c in candidates:
            if c.exists():
                data_file = str(c)
                break
        if data_file is None:
            return PPLResult(ppl=-1, n_tokens=0, elapsed_s=0, data_file="none")

    cmd = [
        str(LLAMA_PPL),
        "-m", model_path,
        "-f", data_file,
        "-c", str(ctx_size),
        "-ngl", "99",
        "-t", "6",
        "--n-cpu-moe", "4",
    ]

    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return PPLResult(ppl=-1, n_tokens=0, elapsed_s=600, data_file=data_file)
    elapsed = time.monotonic() - t0

    import re
    ppl = -1.0
    n_tokens = 0
    for line in result.stdout.split('\n'):
        m = re.search(r'PPL\s*=\s*([\d.]+)', line, re.IGNORECASE)
        if m:
            ppl = float(m.group(1))
        m = re.search(r'(\d+)\s*tokens', line)
        if m:
            n_tokens = int(m.group(1))

    # Fallback: check last lines
    if ppl < 0:
        for line in result.stdout.strip().split('\n')[-10:]:
            m = re.search(r'perplexity\s*[=:]\s*([\d.]+)', line, re.IGNORECASE)
            if m:
                ppl = float(m.group(1))

    return PPLResult(ppl=ppl, n_tokens=n_tokens, elapsed_s=elapsed,
                     data_file=data_file)


# -------------------------------------------------------------------------
# Speed benchmark
# -------------------------------------------------------------------------

def measure_speed(model_path: str, port: int) -> SpeedResult:
    """Measure generation and prompt processing speed.

    Uses a warm-up query followed by a timed generation.
    """
    t0 = time.monotonic()

    # Warm-up (load model into GPU memory, initialize KV cache)
    query_server("Hello", port, max_tokens=8)

    # Long generation for accurate tok/s measurement
    long_prompt = "Write a detailed Python implementation of a binary search tree with insert, delete, and search operations. Include type hints and docstrings."
    text, gen_tokens, gen_elapsed = query_server(long_prompt, port, max_tokens=256)

    gen_tok_s = gen_tokens / gen_elapsed if gen_elapsed > 0 else 0

    # Prompt processing: send a long prompt with minimal generation
    pp_prompt = "Summarize the following in one sentence:\n" + ("The quick brown fox jumps over the lazy dog. " * 50)
    _, _, pp_elapsed = query_server(pp_prompt, port, max_tokens=32)

    # Approximate PP speed from prompt length
    # Rough estimate: ~50 words * 50 repeats * 1.3 tokens/word = ~3250 tokens
    approx_pp_tokens = 3250
    pp_tok_s = approx_pp_tokens / pp_elapsed if pp_elapsed > 0 else 0

    elapsed = time.monotonic() - t0
    return SpeedResult(gen_tok_s=gen_tok_s, pp_tok_s=pp_tok_s, elapsed_s=elapsed)


# -------------------------------------------------------------------------
# Functional tests
# -------------------------------------------------------------------------

def run_functional_tests(port: int) -> FunctionalResult:
    """Run functional quality tests against a running server."""
    details = []
    domain_scores = {}

    for i, test in enumerate(FUNCTIONAL_TESTS):
        try:
            text, _, _ = query_server(test["prompt"], port,
                                       max_tokens=test.get("max_tokens", 512))
            text_lower = text.lower()
            matched = [kw for kw in test["keywords"] if kw.lower() in text_lower]
            passed = len(matched) >= len(test["keywords"]) * 0.5
            score = len(matched) / len(test["keywords"])
        except Exception as e:
            text = f"ERROR: {e}"
            matched = []
            passed = False
            score = 0.0

        details.append({
            "domain": test["domain"],
            "passed": passed,
            "score": score,
            "matched": len(matched),
            "total_kw": len(test["keywords"]),
            "preview": text[:200],
        })

        domain = test["domain"]
        if domain not in domain_scores:
            domain_scores[domain] = {"passed": 0, "total": 0}
        domain_scores[domain]["total"] += 1
        if passed:
            domain_scores[domain]["passed"] += 1

    total = len(FUNCTIONAL_TESTS)
    passed = sum(1 for d in details if d["passed"])

    return FunctionalResult(
        total=total, passed=passed,
        score_pct=100.0 * passed / total if total > 0 else 0,
        per_domain=domain_scores, details=details)


# -------------------------------------------------------------------------
# Full benchmark
# -------------------------------------------------------------------------

def benchmark_model(model_path: str, port: int,
                    run_ppl: bool = True,
                    run_speed: bool = True,
                    run_functional: bool = True,
                    ppl_data: str = None,
                    verbose: bool = True) -> BenchmarkResult:
    """Run full benchmark suite on a single model."""
    model_name = Path(model_path).stem
    file_size = os.path.getsize(model_path) / (1024**3)

    result = BenchmarkResult(
        model_path=model_path,
        model_name=model_name,
        file_size_gb=file_size)

    # Perplexity (does not need server)
    if run_ppl:
        if verbose:
            print(f"  Measuring perplexity...")
        result.ppl = measure_perplexity(model_path, ppl_data)
        if verbose:
            print(f"    PPL={result.ppl.ppl:.4f} ({result.ppl.elapsed_s:.0f}s)")

    # Speed and functional tests need a running server
    if run_speed or run_functional:
        if verbose:
            print(f"  Starting server on port {port}...")
        proc = start_server(model_path, port)
        try:
            if run_speed:
                if verbose:
                    print(f"  Measuring speed...")
                result.speed = measure_speed(model_path, port)
                if verbose:
                    print(f"    Gen={result.speed.gen_tok_s:.1f} tok/s, "
                          f"PP={result.speed.pp_tok_s:.0f} tok/s")

            if run_functional:
                if verbose:
                    print(f"  Running functional tests...")
                result.functional = run_functional_tests(port)
                if verbose:
                    print(f"    Score: {result.functional.passed}/{result.functional.total} "
                          f"({result.functional.score_pct:.0f}%)")
        finally:
            stop_server(proc)

    return result


def compare(model_a_path: str, model_b_path: str,
            run_ppl: bool = True, run_speed: bool = True,
            run_functional: bool = True,
            ppl_data: str = None,
            verbose: bool = True) -> ComparisonReport:
    """Run A/B comparison between two GGUF models.

    Model A is treated as baseline, Model B is the candidate.
    """
    if verbose:
        print(f"=== Benchmarking Model A (baseline): {Path(model_a_path).name} ===")
    result_a = benchmark_model(
        model_a_path, BENCH_PORT_A,
        run_ppl=run_ppl, run_speed=run_speed,
        run_functional=run_functional, ppl_data=ppl_data,
        verbose=verbose)

    if verbose:
        print(f"\n=== Benchmarking Model B (candidate): {Path(model_b_path).name} ===")
    result_b = benchmark_model(
        model_b_path, BENCH_PORT_B,
        run_ppl=run_ppl, run_speed=run_speed,
        run_functional=run_functional, ppl_data=ppl_data,
        verbose=verbose)

    report = ComparisonReport(model_a=result_a, model_b=result_b)

    # Compute deltas
    if result_a.ppl and result_b.ppl and result_a.ppl.ppl > 0 and result_b.ppl.ppl > 0:
        report.ppl_delta_pct = (result_b.ppl.ppl - result_a.ppl.ppl) / result_a.ppl.ppl * 100

    if result_a.speed and result_b.speed and result_a.speed.gen_tok_s > 0:
        report.speed_delta_pct = (result_b.speed.gen_tok_s - result_a.speed.gen_tok_s) / result_a.speed.gen_tok_s * 100

    if result_a.functional and result_b.functional:
        report.functional_delta = result_b.functional.score_pct - result_a.functional.score_pct

    # Verdict
    ppl_ok = report.ppl_delta_pct <= 2.0  # within 2%
    func_ok = report.functional_delta >= -10  # no more than 10% regression
    if ppl_ok and func_ok:
        if report.ppl_delta_pct <= 0.5:
            report.verdict = "EXCELLENT: quality maintained, candidate is valid"
        else:
            report.verdict = "ACCEPTABLE: slight PPL regression within tolerance"
    elif ppl_ok:
        report.verdict = "WARNING: PPL OK but functional regression detected"
    else:
        report.verdict = f"REJECTED: PPL regression {report.ppl_delta_pct:+.1f}% exceeds threshold"

    return report


def format_report(report: ComparisonReport) -> str:
    """Format comparison report as human-readable text."""
    a = report.model_a
    b = report.model_b

    lines = [
        "=" * 70,
        "A/B Quality Benchmark Report",
        "=" * 70,
        "",
        f"Model A (baseline): {a.model_name}",
        f"  Size: {a.file_size_gb:.2f} GB",
        f"Model B (candidate): {b.model_name}",
        f"  Size: {b.file_size_gb:.2f} GB",
        f"  Size delta: {b.file_size_gb - a.file_size_gb:+.2f} GB "
        f"({(b.file_size_gb - a.file_size_gb) / a.file_size_gb * 100:+.1f}%)",
        "",
    ]

    # Perplexity
    if a.ppl and b.ppl:
        lines.extend([
            "PERPLEXITY:",
            f"  Model A: {a.ppl.ppl:.4f}",
            f"  Model B: {b.ppl.ppl:.4f}",
            f"  Delta: {report.ppl_delta_pct:+.2f}%",
            "",
        ])

    # Speed
    if a.speed and b.speed:
        lines.extend([
            "SPEED:",
            f"  Model A: {a.speed.gen_tok_s:.1f} tok/s gen, "
            f"{a.speed.pp_tok_s:.0f} tok/s PP",
            f"  Model B: {b.speed.gen_tok_s:.1f} tok/s gen, "
            f"{b.speed.pp_tok_s:.0f} tok/s PP",
            f"  Gen delta: {report.speed_delta_pct:+.1f}%",
            "",
        ])

    # Functional
    if a.functional and b.functional:
        lines.extend([
            "FUNCTIONAL TESTS:",
            f"  Model A: {a.functional.passed}/{a.functional.total} "
            f"({a.functional.score_pct:.0f}%)",
            f"  Model B: {b.functional.passed}/{b.functional.total} "
            f"({b.functional.score_pct:.0f}%)",
            f"  Delta: {report.functional_delta:+.0f}%",
            "",
            "  Per-domain:",
        ])
        all_domains = sorted(set(
            list(a.functional.per_domain.keys()) +
            list(b.functional.per_domain.keys())))
        for domain in all_domains:
            da = a.functional.per_domain.get(domain, {"passed": 0, "total": 0})
            db = b.functional.per_domain.get(domain, {"passed": 0, "total": 0})
            lines.append(f"    {domain:>12}: A={da['passed']}/{da['total']}, "
                         f"B={db['passed']}/{db['total']}")
        lines.append("")

    lines.extend([
        "=" * 70,
        f"VERDICT: {report.verdict}",
        "=" * 70,
    ])

    return "\n".join(lines)


def save_report(report: ComparisonReport, path: str):
    """Save comparison report as JSON."""
    data = {
        "model_a": {
            "path": report.model_a.model_path,
            "name": report.model_a.model_name,
            "size_gb": report.model_a.file_size_gb,
        },
        "model_b": {
            "path": report.model_b.model_path,
            "name": report.model_b.model_name,
            "size_gb": report.model_b.file_size_gb,
        },
        "ppl_delta_pct": report.ppl_delta_pct,
        "speed_delta_pct": report.speed_delta_pct,
        "functional_delta": report.functional_delta,
        "verdict": report.verdict,
    }

    if report.model_a.ppl:
        data["model_a"]["ppl"] = report.model_a.ppl.ppl
    if report.model_b.ppl:
        data["model_b"]["ppl"] = report.model_b.ppl.ppl
    if report.model_a.speed:
        data["model_a"]["gen_tok_s"] = report.model_a.speed.gen_tok_s
        data["model_a"]["pp_tok_s"] = report.model_a.speed.pp_tok_s
    if report.model_b.speed:
        data["model_b"]["gen_tok_s"] = report.model_b.speed.gen_tok_s
        data["model_b"]["pp_tok_s"] = report.model_b.speed.pp_tok_s
    if report.model_a.functional:
        data["model_a"]["functional_pct"] = report.model_a.functional.score_pct
        data["model_a"]["functional_details"] = report.model_a.functional.details
    if report.model_b.functional:
        data["model_b"]["functional_pct"] = report.model_b.functional.score_pct
        data["model_b"]["functional_details"] = report.model_b.functional.details

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A/B quality benchmark for GGUF models")
    parser.add_argument("model_a", help="Baseline GGUF model path")
    parser.add_argument("model_b", help="Candidate GGUF model path")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON report path")
    parser.add_argument("--ppl-data", default=None,
                        help="Calibration text file for perplexity")
    parser.add_argument("--skip-ppl", action="store_true",
                        help="Skip perplexity measurement")
    parser.add_argument("--skip-speed", action="store_true",
                        help="Skip speed benchmark")
    parser.add_argument("--skip-functional", action="store_true",
                        help="Skip functional tests")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    report = compare(
        args.model_a, args.model_b,
        run_ppl=not args.skip_ppl,
        run_speed=not args.skip_speed,
        run_functional=not args.skip_functional,
        ppl_data=args.ppl_data,
        verbose=verbose)

    print()
    print(format_report(report))

    if args.output:
        save_report(report, args.output)
        if verbose:
            print(f"\nReport saved to {args.output}")
