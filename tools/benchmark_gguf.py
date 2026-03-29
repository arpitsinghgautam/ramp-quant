#!/usr/bin/env python3
"""benchmark_gguf.py — Compare GGUF quality: perplexity + functional tests

Usage:
    # Perplexity on calibration data (requires llama-perplexity)
    python3 benchmark_gguf.py --ppl model.gguf

    # Functional test (keyword matching on domain questions)
    python3 benchmark_gguf.py --functional model.gguf

    # Full A/B comparison
    python3 benchmark_gguf.py --compare model_a.gguf model_b.gguf

    # Quick smoke test (5 questions)
    python3 benchmark_gguf.py --smoke model.gguf
"""

from __future__ import annotations
import argparse, http.client, json, subprocess, sys, time
from pathlib import Path

IK_LLAMA = Path.home() / "ik_llama.cpp" / "build_sm120" / "bin"
WIKI_DATA = Path.home() / ".chimere" / "data" / "calibration" / "wiki_fr_2k.txt"
KINE_DATA = Path.home() / ".chimere" / "data" / "calibration" / "kine_calibration.txt"

# Domain-specific functional tests
TESTS = [
    # KINE (domain knowledge)
    {"q": "Quels sont les 5 criteres du hop test battery pour le retour au sport apres LCA ?",
     "kw": ["LSI", "90", "hop", "single", "triple", "crossover"], "domain": "kine"},
    {"q": "Protocole Alfredson tendinopathie Achille dosage precis",
     "kw": ["excentrique", "3x15", "genou", "12 semaines"], "domain": "kine"},
    {"q": "Red flags lombalgie aigue bilan initial",
     "kw": ["cauda equina", "cancer", "fracture", "infection"], "domain": "kine"},
    {"q": "Difference entre tendinopathie et tendinite pour un patient",
     "kw": ["degenerat", "inflammat", "chronique"], "domain": "kine"},
    {"q": "Score VISA-A combien de questions et score maximum",
     "kw": ["8", "100"], "domain": "kine"},
    # CODE
    {"q": "Ecris une fonction Python verifiant des parentheses valides () [] {}",
     "kw": ["stack", "def", "return", "True"], "domain": "code"},
    {"q": "Implemente merge sort en Python",
     "kw": ["def", "merge", "sort", "left", "right"], "domain": "code"},
    {"q": "Implemente un rate limiter token bucket en Python",
     "kw": ["class", "def", "time", "token", "bucket"], "domain": "code"},
    # MATH
    {"q": "Aire d un triangle cotes 7 8 9",
     "kw": ["heron", "12", "aire"], "domain": "math"},
    {"q": "Derivee de f(x) = x^3 + 2x^2 - 5x + 1",
     "kw": ["3x", "4x", "5"], "domain": "math"},
    # REASONING
    {"q": "Si tous les A sont B et certains B sont C, peut-on conclure que certains A sont C ?",
     "kw": ["non", "pas", "necessairement", "syllogisme"], "domain": "reasoning"},
    {"q": "Un train part a 14h de Paris a 300km/h. Un autre part a 14h30 de Lyon (450km) a 350km/h. Quand se croisent-ils ?",
     "kw": ["heure", "km", "crois"], "domain": "reasoning"},
]


def run_perplexity(gguf_path: str, data_path: str = None) -> dict:
    """Run llama-perplexity on a GGUF file."""
    perp_bin = IK_LLAMA / "llama-perplexity"
    if not perp_bin.exists():
        perp_bin = IK_LLAMA / "perplexity"
    if not perp_bin.exists():
        return {"error": "llama-perplexity not found"}

    # Use wiki data or custom
    if data_path and Path(data_path).exists():
        data = data_path
    elif WIKI_DATA.exists():
        data = str(WIKI_DATA)
    else:
        return {"error": f"No calibration data at {WIKI_DATA}"}

    cmd = [
        str(perp_bin),
        "-m", gguf_path,
        "-f", data,
        "--chunks", "20",
        "-ngl", "99",
        "--n-cpu-moe", "4",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # Parse perplexity from output
        for line in result.stdout.split("\n"):
            if "perplexity" in line.lower() and "=" in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    try:
                        ppl = float(parts[-1].strip().split()[0])
                        return {"perplexity": ppl, "chunks": 20}
                    except ValueError:
                        pass
        return {"error": "Could not parse perplexity", "stdout": result.stdout[-500:]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout (600s)"}
    except Exception as e:
        return {"error": str(e)}


def run_functional(port: int = 8081, tests: list = None) -> dict:
    """Run functional keyword tests against a running server."""
    if tests is None:
        tests = TESTS
    total_kw = total_found = 0
    results = []
    by_domain = {}

    for t in tests:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
            body = json.dumps({
                "model": "test", "messages": [{"role": "user", "content": t["q"]}],
                "max_tokens": 1024, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            t0 = time.monotonic()
            conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
            data = json.loads(conn.getresponse().read())
            conn.close()
            ms = (time.monotonic() - t0) * 1000

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            found = sum(1 for kw in t["kw"] if kw.lower() in content.lower())
            total_kw += len(t["kw"])
            total_found += found
            passed = found >= len(t["kw"]) * 0.6
            results.append({"domain": t["domain"], "found": found, "total": len(t["kw"]), "pass": passed, "ms": ms})
            by_domain.setdefault(t["domain"], []).append(passed)
        except Exception as e:
            total_kw += len(t["kw"])
            results.append({"domain": t["domain"], "found": 0, "total": len(t["kw"]), "pass": False, "error": str(e)})
            by_domain.setdefault(t["domain"], []).append(False)

    domain_scores = {}
    for d, passes in by_domain.items():
        domain_scores[d] = {"pass": sum(passes), "total": len(passes), "pct": sum(passes) / len(passes) * 100}

    return {
        "total_keywords": f"{total_found}/{total_kw} ({total_found/total_kw*100:.0f}%)",
        "total_pass": f"{sum(r['pass'] for r in results)}/{len(results)}",
        "domains": domain_scores,
        "details": results,
    }


def compare_models(gguf_a: str, gguf_b: str):
    """Full A/B comparison (requires stopping/starting server)."""
    print(f"=== COMPARING ===")
    print(f"  A: {Path(gguf_a).name}")
    print(f"  B: {Path(gguf_b).name}")
    print()
    print("NOTE: This requires manually swapping the GGUF in chimere-server")
    print("and running --functional on each. Use --functional for quick tests.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GGUF quality")
    parser.add_argument("gguf", nargs="?", help="GGUF file path")
    parser.add_argument("--ppl", action="store_true", help="Run perplexity benchmark")
    parser.add_argument("--functional", action="store_true", help="Run functional tests")
    parser.add_argument("--smoke", action="store_true", help="Quick 5-question test")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two GGUFs")
    parser.add_argument("--port", type=int, default=8081, help="Server port")
    parser.add_argument("--data", type=str, help="Custom calibration data path")
    args = parser.parse_args()

    if args.compare:
        compare_models(*args.compare)
        return

    if args.ppl and args.gguf:
        print(f"=== PERPLEXITY: {Path(args.gguf).name} ===")
        result = run_perplexity(args.gguf, args.data)
        print(json.dumps(result, indent=2))

    if args.functional or args.smoke:
        tests = TESTS[:5] if args.smoke else TESTS
        print(f"\n=== FUNCTIONAL ({len(tests)} tests, port {args.port}) ===")
        result = run_functional(port=args.port, tests=tests)
        print(f"Keywords: {result['total_keywords']}")
        print(f"Pass: {result['total_pass']}")
        for d, s in result["domains"].items():
            print(f"  {d:12s} {s['pass']}/{s['total']} ({s['pct']:.0f}%)")

    if not any([args.ppl, args.functional, args.smoke, args.compare]):
        parser.print_help()


if __name__ == "__main__":
    main()
