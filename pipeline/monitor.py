#!/usr/bin/env python3
"""
Quality Monitor: Real-time quantization error tracking and anomaly detection.

Monitors the quantization process by computing per-tensor error metrics
(Frobenius norm, cosine similarity) and detecting anomalies where error
exceeds 2 standard deviations from the mean for that layer type.

Produces an HTML report with:
- Heatmap of per-layer quantization quality
- Error distribution by component type (attention, GDN, SSM, experts)
- Anomaly flags with recommended actions
- Summary statistics

This module can be used standalone or integrated into run_pipeline.py.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import numpy as np

# -------------------------------------------------------------------------
# Architecture constants
# -------------------------------------------------------------------------

FULL_ATTENTION_LAYERS = set(range(3, 40, 4))

LAYER_TYPE_MAP = {
    "attn": "attention",
    "gdn": "gdn",
    "ssm": "ssm",
    "shared": "shared_expert",
    "experts": "experts",
    "norms": "norms",
    "gates": "gates",
}


# -------------------------------------------------------------------------
# Error tracking
# -------------------------------------------------------------------------

@dataclass
class TensorError:
    """Per-tensor quantization error metrics."""
    name: str
    group: str
    layer_idx: int
    component: str  # attention, gdn, ssm, experts, etc.
    qtype: str
    frobenius_rel: float   # ||W - W_q||_F / ||W||_F
    cosine_sim: float      # cos(W, W_q)
    max_abs_error: float   # max |W_i - W_q_i|
    n_elements: int
    is_anomaly: bool = False
    anomaly_zscore: float = 0.0


@dataclass
class LayerStats:
    """Aggregated statistics for a model layer."""
    layer_idx: int
    layer_type: str  # "full_attention" or "gdn"
    components: dict = field(default_factory=dict)
    mean_error: float = 0.0
    max_error: float = 0.0
    n_tensors: int = 0
    n_anomalies: int = 0


class QuantizationMonitor:
    """Tracks quantization errors across all tensors and detects anomalies."""

    def __init__(self):
        self.tensor_errors: list[TensorError] = []
        self.layer_stats: dict[int, LayerStats] = {}
        self.component_stats: dict[str, list[float]] = defaultdict(list)
        self._finalized = False

    def record_error(self, name: str, group: str, layer_idx: int,
                     component: str, qtype: str,
                     W_orig: np.ndarray, W_quant: np.ndarray):
        """Record quantization error for a single tensor.

        Args:
            name: tensor name
            group: decision group name
            layer_idx: layer index (-1 for global)
            component: component type (attention, gdn, ssm, experts, etc.)
            qtype: quantization type applied
            W_orig: original weight values (flat float32)
            W_quant: quantized weight values (flat float32)
        """
        diff = W_orig - W_quant
        norm_orig = float(np.linalg.norm(W_orig))
        norm_diff = float(np.linalg.norm(diff))

        frob_rel = norm_diff / norm_orig if norm_orig > 1e-12 else 0.0
        max_abs = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0

        dot = np.dot(W_orig.flatten(), W_quant.flatten())
        norm_q = float(np.linalg.norm(W_quant))
        cos_sim = float(dot / (norm_orig * norm_q)) if (norm_orig > 1e-12 and norm_q > 1e-12) else 1.0

        te = TensorError(
            name=name, group=group, layer_idx=layer_idx,
            component=component, qtype=qtype,
            frobenius_rel=frob_rel, cosine_sim=cos_sim,
            max_abs_error=max_abs, n_elements=W_orig.size)

        self.tensor_errors.append(te)
        self.component_stats[component].append(frob_rel)
        self._finalized = False

    def record_error_metrics(self, name: str, group: str, layer_idx: int,
                             component: str, qtype: str,
                             frobenius_rel: float, cosine_sim: float = 1.0,
                             max_abs_error: float = 0.0, n_elements: int = 0):
        """Record pre-computed error metrics (for integration with ramp-quantize)."""
        te = TensorError(
            name=name, group=group, layer_idx=layer_idx,
            component=component, qtype=qtype,
            frobenius_rel=frobenius_rel, cosine_sim=cosine_sim,
            max_abs_error=max_abs_error, n_elements=n_elements)

        self.tensor_errors.append(te)
        self.component_stats[component].append(frobenius_rel)
        self._finalized = False

    def finalize(self):
        """Compute anomaly detection and aggregate layer statistics."""
        if self._finalized:
            return

        # Compute per-component mean and std
        component_mean = {}
        component_std = {}
        for comp, errors in self.component_stats.items():
            arr = np.array(errors)
            component_mean[comp] = float(np.mean(arr))
            component_std[comp] = float(np.std(arr))

        # Flag anomalies (> 2 sigma from component mean)
        for te in self.tensor_errors:
            mean = component_mean.get(te.component, 0)
            std = component_std.get(te.component, 1)
            if std > 1e-8:
                te.anomaly_zscore = (te.frobenius_rel - mean) / std
                te.is_anomaly = te.anomaly_zscore > 2.0
            else:
                te.anomaly_zscore = 0.0
                te.is_anomaly = False

        # Aggregate per-layer stats
        self.layer_stats = {}
        for te in self.tensor_errors:
            idx = te.layer_idx
            if idx not in self.layer_stats:
                ltype = "full_attention" if idx in FULL_ATTENTION_LAYERS else "gdn"
                if idx < 0:
                    ltype = "global"
                self.layer_stats[idx] = LayerStats(
                    layer_idx=idx, layer_type=ltype)

            ls = self.layer_stats[idx]
            ls.n_tensors += 1
            ls.mean_error = (ls.mean_error * (ls.n_tensors - 1) + te.frobenius_rel) / ls.n_tensors
            ls.max_error = max(ls.max_error, te.frobenius_rel)
            if te.is_anomaly:
                ls.n_anomalies += 1

            comp = te.component
            if comp not in ls.components:
                ls.components[comp] = {"mean_error": 0.0, "count": 0}
            ls.components[comp]["count"] += 1
            old_mean = ls.components[comp]["mean_error"]
            n = ls.components[comp]["count"]
            ls.components[comp]["mean_error"] = old_mean + (te.frobenius_rel - old_mean) / n

        self._finalized = True

    def get_anomalies(self) -> list[TensorError]:
        """Return list of anomalous tensors (error > 2 sigma)."""
        self.finalize()
        return [te for te in self.tensor_errors if te.is_anomaly]

    def summary(self) -> str:
        """Text summary of quantization quality."""
        self.finalize()

        lines = [
            "=" * 70,
            "Quantization Quality Summary",
            "=" * 70,
            "",
        ]

        # Overall stats
        all_errors = [te.frobenius_rel for te in self.tensor_errors]
        if all_errors:
            arr = np.array(all_errors)
            lines.append(f"Total tensors: {len(all_errors)}")
            lines.append(f"Mean relative error: {np.mean(arr):.6f}")
            lines.append(f"Median relative error: {np.median(arr):.6f}")
            lines.append(f"Max relative error: {np.max(arr):.6f}")
            lines.append(f"Std relative error: {np.std(arr):.6f}")
        else:
            lines.append("No tensors recorded yet.")
            return "\n".join(lines)

        # Per-component breakdown
        lines.extend(["", "Error by component type:",
                       f"  {'Component':<20} {'Mean':>10} {'Std':>10} "
                       f"{'Max':>10} {'Count':>6}"])
        lines.append(f"  {'-'*60}")

        for comp in sorted(self.component_stats.keys()):
            errors = np.array(self.component_stats[comp])
            lines.append(
                f"  {comp:<20} {np.mean(errors):>10.6f} "
                f"{np.std(errors):>10.6f} {np.max(errors):>10.6f} "
                f"{len(errors):>6}")

        # Anomalies
        anomalies = self.get_anomalies()
        if anomalies:
            lines.extend(["", f"ANOMALIES ({len(anomalies)} tensors with error > 2 sigma):"])
            for te in sorted(anomalies, key=lambda t: t.frobenius_rel, reverse=True)[:10]:
                lines.append(
                    f"  {te.name:<50} err={te.frobenius_rel:.6f} "
                    f"z={te.anomaly_zscore:.1f} [{te.qtype}]")
            if len(anomalies) > 10:
                lines.append(f"  ... and {len(anomalies) - 10} more")
        else:
            lines.extend(["", "No anomalies detected (all errors within 2 sigma)."])

        return "\n".join(lines)

    def generate_html_report(self, output_path: str, title: str = "Quantization Quality"):
        """Generate an HTML report with heatmap and statistics.

        The heatmap shows per-layer, per-component error intensity.
        """
        self.finalize()

        # Prepare heatmap data: layers x components
        all_components = sorted(set(te.component for te in self.tensor_errors))
        layer_indices = sorted(set(te.layer_idx for te in self.tensor_errors))

        # Build error matrix
        error_matrix = {}
        for te in self.tensor_errors:
            key = (te.layer_idx, te.component)
            if key not in error_matrix:
                error_matrix[key] = []
            error_matrix[key].append(te.frobenius_rel)

        # HTML generation
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            f"<title>{title}</title>",
            "<style>",
            "body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; "
            "background: #1a1a2e; color: #e0e0e0; }",
            "h1 { color: #00d4ff; }",
            "h2 { color: #7fdbff; margin-top: 30px; }",
            "table { border-collapse: collapse; margin: 10px 0; }",
            "th, td { padding: 4px 8px; border: 1px solid #333; text-align: center; "
            "font-size: 12px; }",
            "th { background: #16213e; color: #00d4ff; }",
            ".anomaly { background: #ff4444 !important; color: white; font-weight: bold; }",
            ".good { background: #0a3d0a; }",
            ".ok { background: #3d3d0a; }",
            ".bad { background: #3d0a0a; }",
            ".stats-table td { text-align: left; padding: 4px 16px; }",
            ".stats-table th { text-align: left; }",
            "pre { background: #16213e; padding: 10px; border-radius: 4px; "
            "overflow-x: auto; }",
            "</style>",
            "</head><body>",
            f"<h1>{title}</h1>",
            f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        # Summary statistics
        all_errors = [te.frobenius_rel for te in self.tensor_errors]
        arr = np.array(all_errors) if all_errors else np.array([0.0])

        html_parts.extend([
            "<h2>Summary</h2>",
            '<table class="stats-table">',
            f"<tr><td>Total tensors</td><td>{len(all_errors)}</td></tr>",
            f"<tr><td>Mean error</td><td>{np.mean(arr):.6f}</td></tr>",
            f"<tr><td>Median error</td><td>{np.median(arr):.6f}</td></tr>",
            f"<tr><td>Max error</td><td>{np.max(arr):.6f}</td></tr>",
            f"<tr><td>Anomalies</td><td>{len(self.get_anomalies())}</td></tr>",
            "</table>",
        ])

        # Heatmap
        html_parts.extend([
            "<h2>Error Heatmap (Layer x Component)</h2>",
            "<p>Color: green=low error, yellow=medium, red=high, "
            "bright red=anomaly (&gt;2 sigma)</p>",
            "<table>",
            "<tr><th>Layer</th>",
        ])
        for comp in all_components:
            html_parts.append(f"<th>{comp[:12]}</th>")
        html_parts.append("</tr>")

        # Compute global percentiles for color mapping
        all_means = []
        for key, vals in error_matrix.items():
            all_means.append(np.mean(vals))
        if all_means:
            p50 = np.percentile(all_means, 50)
            p90 = np.percentile(all_means, 90)
        else:
            p50, p90 = 0.05, 0.1

        for layer_idx in layer_indices:
            ltype = "A" if layer_idx in FULL_ATTENTION_LAYERS else "G"
            if layer_idx < 0:
                ltype = "*"
            html_parts.append(f"<tr><th>{layer_idx} ({ltype})</th>")

            for comp in all_components:
                key = (layer_idx, comp)
                if key in error_matrix:
                    mean_err = np.mean(error_matrix[key])
                    # Check for anomaly
                    has_anomaly = any(
                        te.is_anomaly for te in self.tensor_errors
                        if te.layer_idx == layer_idx and te.component == comp)

                    if has_anomaly:
                        css_class = "anomaly"
                    elif mean_err <= p50:
                        css_class = "good"
                    elif mean_err <= p90:
                        css_class = "ok"
                    else:
                        css_class = "bad"

                    html_parts.append(
                        f'<td class="{css_class}">{mean_err:.4f}</td>')
                else:
                    html_parts.append('<td style="background:#111">-</td>')

            html_parts.append("</tr>")
        html_parts.append("</table>")

        # Per-component breakdown
        html_parts.extend([
            "<h2>Error by Component Type</h2>",
            "<table>",
            "<tr><th>Component</th><th>Mean</th><th>Std</th>"
            "<th>Max</th><th>Count</th></tr>",
        ])
        for comp in sorted(self.component_stats.keys()):
            errors = np.array(self.component_stats[comp])
            html_parts.append(
                f"<tr><td>{comp}</td>"
                f"<td>{np.mean(errors):.6f}</td>"
                f"<td>{np.std(errors):.6f}</td>"
                f"<td>{np.max(errors):.6f}</td>"
                f"<td>{len(errors)}</td></tr>")
        html_parts.append("</table>")

        # Anomaly list
        anomalies = self.get_anomalies()
        if anomalies:
            html_parts.extend([
                f"<h2>Anomalies ({len(anomalies)} tensors)</h2>",
                "<table>",
                "<tr><th>Tensor</th><th>Error</th><th>Z-score</th>"
                "<th>QType</th><th>Component</th></tr>",
            ])
            for te in sorted(anomalies, key=lambda t: t.frobenius_rel, reverse=True):
                html_parts.append(
                    f"<tr><td>{te.name}</td>"
                    f"<td>{te.frobenius_rel:.6f}</td>"
                    f"<td>{te.anomaly_zscore:.1f}</td>"
                    f"<td>{te.qtype}</td>"
                    f"<td>{te.component}</td></tr>")
            html_parts.append("</table>")

        html_parts.extend(["</body></html>"])

        Path(output_path).write_text("\n".join(html_parts))

    def save_json(self, path: str):
        """Save all error data as JSON for later analysis."""
        self.finalize()
        data = {
            "tensor_errors": [
                {
                    "name": te.name, "group": te.group,
                    "layer_idx": te.layer_idx, "component": te.component,
                    "qtype": te.qtype,
                    "frobenius_rel": te.frobenius_rel,
                    "cosine_sim": te.cosine_sim,
                    "max_abs_error": te.max_abs_error,
                    "n_elements": te.n_elements,
                    "is_anomaly": te.is_anomaly,
                    "anomaly_zscore": te.anomaly_zscore,
                }
                for te in self.tensor_errors
            ],
            "component_stats": {
                comp: {
                    "mean": float(np.mean(errors)),
                    "std": float(np.std(errors)),
                    "max": float(np.max(errors)),
                    "count": len(errors),
                }
                for comp, errors in self.component_stats.items()
            },
            "n_anomalies": len(self.get_anomalies()),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: str):
        """Load error data from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.tensor_errors = []
        self.component_stats = defaultdict(list)
        for td in data["tensor_errors"]:
            te = TensorError(**td)
            self.tensor_errors.append(te)
            self.component_stats[te.component].append(te.frobenius_rel)
        self._finalized = False


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantization quality monitor and anomaly detector")
    parser.add_argument("errors_json", help="Path to monitor JSON data")
    parser.add_argument("--html", default=None,
                        help="Output HTML report path")
    parser.add_argument("--summary", action="store_true",
                        help="Print text summary")
    args = parser.parse_args()

    monitor = QuantizationMonitor()
    monitor.load_json(args.errors_json)

    if args.summary:
        print(monitor.summary())

    if args.html:
        monitor.generate_html_report(args.html)
        print(f"HTML report saved to {args.html}")
    elif not args.summary:
        print(monitor.summary())
