#!/usr/bin/env python3
"""
RAMP-Local: GGUF Validation — Perplexity + Functional Benchmarks

Validates a RAMP-optimized GGUF against the baseline by running:
1. Perplexity on calibration text (llama-perplexity)
2. Functional tests (keyword-matching on domain questions via llama-server)
3. A/B comparison report

Usage:
    # Compare RAMP-optimized vs baseline
    python3 validate.py --compare baseline.gguf ramp.gguf

    # Perplexity only
    python3 validate.py --ppl model.gguf

    # Functional tests only (starts temporary llama-server)
    python3 validate.py --functional model.gguf

    # Full validation report
    python3 validate.py --full model.gguf --baseline baseline.gguf
"""

from __future__ import annotations
import argparse, json, subprocess, sys, time, signal, os
from pathlib import Path
from dataclasses import dataclass

IK_LLAMA = Path.home() / "ik_llama.cpp" / "build_sm120" / "bin"
LLAMA_SERVER = IK_LLAMA / "llama-server"
LLAMA_PPL = IK_LLAMA / "llama-perplexity"

CALIBRATION_DIR = Path.home() / ".chimere" / "data" / "calibration"
WIKI_DATA = CALIBRATION_DIR / "wiki_fr_2k.txt"
CHIMERE_DATA = CALIBRATION_DIR / "chimere_calibration.txt"

VALIDATION_PORT = 8099  # Temporary port, avoid conflict with prod (8081)

# ---------------------------------------------------------------------------
# Functional tests — reused from benchmark_gguf.py
# ---------------------------------------------------------------------------

TESTS = [
    # KINE
    {"q": "Quels sont les 5 criteres du hop test battery pour le retour au sport apres LCA ?",
     "kw": ["LSI", "90", "hop", "single", "triple", "crossover"], "domain": "kine"},
    {"q": "Protocole Alfredson tendinopathie Achille dosage precis",
     "kw": ["excentrique", "3x15", "genou", "12 semaines"], "domain": "kine"},
    {"q": "Red flags lombalgie aigue bilan initial",
     "kw": ["cauda equina", "cancer", "fracture", "infection"], "domain": "kine"},
    {"q": "Score VISA-A combien de questions et score maximum",
     "kw": ["8", "100"], "domain": "kine"},
    # CODE
    {"q": "Ecris une fonction Python verifiant des parentheses valides () [] {}",
     "kw": ["stack", "def", "return", "True"], "domain": "code"},
    {"q": "Implemente merge sort en Python",
     "kw": ["def", "merge", "sort", "left", "right"], "domain": "code"},
    # MATH
    {"q": "Derivee de f(x) = x^3 + 2x^2 - 5x + 1",
     "kw": ["3x", "4x", "5"], "domain": "math"},
    # GENERAL
    {"q": "Explique le theoreme de Bayes en 3 phrases",
     "kw": ["prior", "posterior", "likelihood", "condition"], "domain": "general"},
    {"q": "Quels sont les avantages de Rust par rapport au C++ ?",
     "kw": ["memory", "safe", "ownership", "borrow"], "domain": "general"},
    {"q": "Comment fonctionne un transformer en deep learning ?",
     "kw": ["attention", "query", "key", "value"], "domain": "general"},
]


@dataclass
class PPLResult:
    ppl: float
    nll: float
    n_tokens: int
    model_path: str
    data_file: str
    elapsed_s: float


@dataclass
class FunctionalResult:
    total: int
    passed: int
    score_pct: float
    per_domain: dict
    details: list
    model_path: str
    elapsed_s: float


# ---------------------------------------------------------------------------
# Perplexity measurement
# ---------------------------------------------------------------------------

def measure_ppl(model_path: str, data_file: str = None,
                ctx_size: int = 2048, n_gpu_layers: int = 99,
                threads: int = 6) -> PPLResult:
    """Run llama-perplexity and parse output."""
    if data_file is None:
        data_file = str(WIKI_DATA) if WIKI_DATA.exists() else str(CHIMERE_DATA)

    if not Path(data_file).exists():
        print(f"WARNING: calibration file {data_file} not found", flush=True)
        return PPLResult(ppl=-1, nll=-1, n_tokens=0,
                        model_path=model_path, data_file=data_file, elapsed_s=0)

    cmd = [
        str(LLAMA_PPL),
        "-m", model_path,
        "-f", data_file,
        "-c", str(ctx_size),
        "-ngl", str(n_gpu_layers),
        "-t", str(threads),
        "--n-cpu-moe", "4",
    ]

    print(f"Running perplexity: {Path(model_path).name}", flush=True)
    print(f"  Data: {data_file}", flush=True)
    t0 = time.monotonic()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("ERROR: perplexity timed out (10 min)", flush=True)
        return PPLResult(ppl=-1, nll=-1, n_tokens=0,
                        model_path=model_path, data_file=data_file, elapsed_s=600)

    elapsed = time.monotonic() - t0

    # Parse output: look for "Final estimate: PPL = X.XXXX"
    ppl = -1.0
    nll = -1.0
    n_tokens = 0
    for line in result.stdout.split('\n'):
        if 'Final estimate' in line and 'PPL' in line:
            import re
            m = re.search(r'PPL\s*=\s*([\d.]+)', line)
            if m:
                ppl = float(m.group(1))
            m = re.search(r'NLL\s*=\s*([\d.]+)', line)
            if m:
                nll = float(m.group(1))
        elif 'tokens' in line.lower():
            import re
            m = re.search(r'(\d+)\s*tokens', line)
            if m:
                n_tokens = int(m.group(1))

    # Also check last lines for perplexity output
    if ppl < 0:
        for line in result.stdout.split('\n')[-10:]:
            import re
            m = re.search(r'perplexity\s*[=:]\s*([\d.]+)', line, re.IGNORECASE)
            if m:
                ppl = float(m.group(1))

    if ppl < 0:
        print(f"WARNING: could not parse perplexity from output", flush=True)
        print(f"  stdout (last 5 lines):", flush=True)
        for line in result.stdout.strip().split('\n')[-5:]:
            print(f"    {line}", flush=True)
        if result.stderr.strip():
            print(f"  stderr (last 3 lines):", flush=True)
            for line in result.stderr.strip().split('\n')[-3:]:
                print(f"    {line}", flush=True)

    return PPLResult(ppl=ppl, nll=nll, n_tokens=n_tokens,
                    model_path=model_path, data_file=data_file,
                    elapsed_s=elapsed)


# ---------------------------------------------------------------------------
# Functional tests via temporary llama-server
# ---------------------------------------------------------------------------

def start_server(model_path: str, port: int = VALIDATION_PORT,
                 n_gpu_layers: int = 99, ctx: int = 4096) -> subprocess.Popen:
    """Start a temporary llama-server for functional testing."""
    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx),
        "-np", "1",
        "--port", str(port),
        "--n-cpu-moe", "4",
        "--flash-attn", "on",
        "--jinja",
        "-b", "2048",
        "-ub", "512",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to be ready
    import http.client
    for attempt in range(60):
        time.sleep(2)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                print(f"  Server ready on port {port} (attempt {attempt+1})", flush=True)
                conn.close()
                return proc
            conn.close()
        except Exception:
            pass

    # Server didn't start
    proc.kill()
    raise RuntimeError(f"llama-server failed to start in 120s")


def query_server(prompt: str, port: int = VALIDATION_PORT,
                 max_tokens: int = 512) -> str:
    """Send a completion request to llama-server."""
    import http.client
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request("POST", "/v1/chat/completions", body,
                {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()

    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"]
    return ""


def run_functional_tests(model_path: str,
                         port: int = VALIDATION_PORT) -> FunctionalResult:
    """Run functional tests against a model via temporary server."""
    t0 = time.monotonic()

    print(f"Starting server for functional tests: {Path(model_path).name}", flush=True)
    proc = start_server(model_path, port)

    try:
        details = []
        domain_scores = {}

        for i, test in enumerate(TESTS):
            print(f"  Test {i+1}/{len(TESTS)} [{test['domain']}]...", end=" ", flush=True)
            try:
                answer = query_server(test["q"], port)
                answer_lower = answer.lower()
                matched = [kw for kw in test["kw"]
                          if kw.lower() in answer_lower]
                passed = len(matched) >= len(test["kw"]) * 0.5
                score = len(matched) / len(test["kw"])
            except Exception as e:
                answer = f"ERROR: {e}"
                matched = []
                passed = False
                score = 0.0

            details.append({
                "question": test["q"],
                "domain": test["domain"],
                "passed": passed,
                "score": score,
                "matched_kw": matched,
                "total_kw": len(test["kw"]),
                "answer_preview": answer[:200],
            })
            print(f"{'PASS' if passed else 'FAIL'} ({len(matched)}/{len(test['kw'])} kw)",
                  flush=True)

            # Accumulate domain scores
            domain = test["domain"]
            if domain not in domain_scores:
                domain_scores[domain] = {"passed": 0, "total": 0}
            domain_scores[domain]["total"] += 1
            if passed:
                domain_scores[domain]["passed"] += 1

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    elapsed = time.monotonic() - t0
    total = len(TESTS)
    passed = sum(1 for d in details if d["passed"])

    return FunctionalResult(
        total=total, passed=passed,
        score_pct=100.0 * passed / total if total > 0 else 0.0,
        per_domain=domain_scores, details=details,
        model_path=model_path, elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def compare_report(baseline_ppl: PPLResult, ramp_ppl: PPLResult,
                   baseline_func: FunctionalResult = None,
                   ramp_func: FunctionalResult = None) -> str:
    """Generate A/B comparison report."""
    lines = [
        "=" * 70,
        "RAMP-Local Validation Report",
        "=" * 70,
        "",
        "PERPLEXITY COMPARISON:",
        f"  {'Model':<45} {'PPL':>10} {'NLL':>10}",
        f"  {'-'*65}",
        f"  Baseline: {Path(baseline_ppl.model_path).name:<33} "
        f"{baseline_ppl.ppl:>10.4f} {baseline_ppl.nll:>10.4f}",
        f"  RAMP:     {Path(ramp_ppl.model_path).name:<33} "
        f"{ramp_ppl.ppl:>10.4f} {ramp_ppl.nll:>10.4f}",
    ]

    if baseline_ppl.ppl > 0 and ramp_ppl.ppl > 0:
        delta = ramp_ppl.ppl - baseline_ppl.ppl
        delta_pct = 100.0 * delta / baseline_ppl.ppl
        direction = "WORSE" if delta > 0 else "BETTER"
        lines.append(f"  Delta:    {delta:+.4f} ({delta_pct:+.2f}%) — {direction}")
    lines.append("")

    if baseline_func and ramp_func:
        lines.extend([
            "FUNCTIONAL TESTS:",
            f"  Baseline: {baseline_func.passed}/{baseline_func.total} "
            f"({baseline_func.score_pct:.0f}%)",
            f"  RAMP:     {ramp_func.passed}/{ramp_func.total} "
            f"({ramp_func.score_pct:.0f}%)",
            "",
            "  Per-domain:",
        ])
        all_domains = set(list(baseline_func.per_domain.keys()) +
                         list(ramp_func.per_domain.keys()))
        for domain in sorted(all_domains):
            bl = baseline_func.per_domain.get(domain, {"passed": 0, "total": 0})
            rm = ramp_func.per_domain.get(domain, {"passed": 0, "total": 0})
            lines.append(f"    {domain:>10}: baseline {bl['passed']}/{bl['total']}, "
                        f"RAMP {rm['passed']}/{rm['total']}")

    # File sizes
    bl_size = os.path.getsize(baseline_ppl.model_path)
    rm_size = os.path.getsize(ramp_ppl.model_path)
    bl_gb = bl_size / (1024**3)
    rm_gb = rm_size / (1024**3)
    delta_gb = rm_gb - bl_gb
    delta_pct_size = 100.0 * delta_gb / bl_gb if bl_gb > 0 else 0

    lines.extend([
        "",
        "FILE SIZE:",
        f"  Baseline: {bl_gb:.2f} GB",
        f"  RAMP:     {rm_gb:.2f} GB",
        f"  Delta:    {delta_gb:+.2f} GB ({delta_pct_size:+.1f}%)",
    ])

    # Verdict
    lines.extend(["", "=" * 70])
    if baseline_ppl.ppl > 0 and ramp_ppl.ppl > 0:
        if ramp_ppl.ppl <= baseline_ppl.ppl * 1.005:  # within 0.5%
            if ramp_func and ramp_func.score_pct >= baseline_func.score_pct - 5:
                lines.append("VERDICT: RAMP config is VALID — quality maintained")
            else:
                lines.append("VERDICT: RAMP PPL OK but functional regression detected")
        elif ramp_ppl.ppl <= baseline_ppl.ppl * 1.02:  # within 2%
            lines.append("VERDICT: RAMP config is MARGINAL — slight PPL regression")
        else:
            lines.append("VERDICT: RAMP config REGRESSED — PPL too high")
    else:
        lines.append("VERDICT: Perplexity measurement failed — manual review needed")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAMP-Local: Validate GGUF quality (perplexity + functional)")
    sub = parser.add_subparsers(dest="mode")

    p_ppl = sub.add_parser("ppl", help="Measure perplexity only")
    p_ppl.add_argument("model", help="GGUF model path")
    p_ppl.add_argument("--data", default=None, help="Calibration text file")
    p_ppl.add_argument("--ctx", type=int, default=2048)

    p_func = sub.add_parser("functional", help="Run functional tests only")
    p_func.add_argument("model", help="GGUF model path")

    p_compare = sub.add_parser("compare", help="A/B comparison")
    p_compare.add_argument("baseline", help="Baseline GGUF")
    p_compare.add_argument("ramp", help="RAMP-optimized GGUF")
    p_compare.add_argument("--ppl-only", action="store_true",
                          help="Skip functional tests (faster)")
    p_compare.add_argument("--data", default=None, help="Calibration text file")

    p_full = sub.add_parser("full", help="Full validation (ppl + functional)")
    p_full.add_argument("model", help="GGUF model path")
    p_full.add_argument("--baseline", default=None, help="Baseline GGUF for comparison")
    p_full.add_argument("--data", default=None, help="Calibration text file")
    p_full.add_argument("--output", "-o", default=None, help="Output JSON report")

    args = parser.parse_args()

    if args.mode == "ppl":
        result = measure_ppl(args.model, args.data, args.ctx)
        print(f"\nPerplexity: {result.ppl:.4f} (NLL={result.nll:.4f}, "
              f"{result.n_tokens} tokens, {result.elapsed_s:.0f}s)")

    elif args.mode == "functional":
        result = run_functional_tests(args.model)
        print(f"\nFunctional: {result.passed}/{result.total} "
              f"({result.score_pct:.0f}%) in {result.elapsed_s:.0f}s")

    elif args.mode == "compare":
        # Run perplexity on both
        print("=== Baseline PPL ===")
        bl_ppl = measure_ppl(args.baseline, args.data)
        print(f"  PPL={bl_ppl.ppl:.4f}")

        print("\n=== RAMP PPL ===")
        rm_ppl = measure_ppl(args.ramp, args.data)
        print(f"  PPL={rm_ppl.ppl:.4f}")

        bl_func = None
        rm_func = None
        if not args.ppl_only:
            print("\n=== Baseline Functional ===")
            bl_func = run_functional_tests(args.baseline, VALIDATION_PORT)
            print("\n=== RAMP Functional ===")
            rm_func = run_functional_tests(args.ramp, VALIDATION_PORT + 1)

        report = compare_report(bl_ppl, rm_ppl, bl_func, rm_func)
        print(f"\n{report}")

    elif args.mode == "full":
        print("=== Perplexity ===")
        ppl = measure_ppl(args.model, args.data)
        print(f"  PPL={ppl.ppl:.4f}")

        print("\n=== Functional Tests ===")
        func = run_functional_tests(args.model)
        print(f"  Score: {func.passed}/{func.total} ({func.score_pct:.0f}%)")

        if args.baseline:
            print("\n=== Baseline PPL ===")
            bl_ppl = measure_ppl(args.baseline, args.data)
            print(f"  PPL={bl_ppl.ppl:.4f}")

            print("\n=== Baseline Functional ===")
            bl_func = run_functional_tests(args.baseline, VALIDATION_PORT + 1)

            report = compare_report(bl_ppl, ppl, bl_func, func)
            print(f"\n{report}")

        if args.output:
            output_data = {
                "model": args.model,
                "ppl": ppl.ppl,
                "nll": ppl.nll,
                "functional_score": func.score_pct,
                "functional_passed": func.passed,
                "functional_total": func.total,
                "per_domain": func.per_domain,
                "details": func.details,
            }
            if args.baseline:
                output_data["baseline_ppl"] = bl_ppl.ppl
                output_data["baseline_functional"] = bl_func.score_pct
            Path(args.output).write_text(json.dumps(output_data, indent=2))
            print(f"\nReport saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
