#!/usr/bin/env python3
"""
Classical Shadows vs MPS — Accuracy vs Resources
=================================================

Both methods estimate ⟨Z_i Z_j⟩ on a central bond of a 2D TFIM.
The demo sweeps resources (M snapshots for shadows, χ for MPS) and
shows that MPS converges to the correct answer with far fewer resources.

Usage:
    python mipt_classical_shadows.py --small        # 4×4 = 16 qubits (~5-10 min)
    python mipt_classical_shadows.py --small --gpu  # GPU acceleration
    python mipt_classical_shadows.py                # 6×6 = 36 qubits (GPU workstation)
    python mipt_classical_shadows.py --gpu          # full 6×6 with GPU
"""

import sys
import os
import time
import json
import hashlib

import numpy as np

from helpers import (
    Config, site_index, get_nn_bonds, build_pauli_observable,
    PAULI_Z,
    compute_reference, sweep_shadow_accuracy, sweep_mps_accuracy,
    plot_accuracy_vs_resources, SV_LIMIT
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────

def header(title):
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")


def banner(config):
    gpu_str = 'ENABLED' if config.use_gpu else 'DISABLED (use --gpu)'
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MAESTRO SHOWCASE: Classical Shadows vs MPS                    ║")
    print("║  Accuracy vs Resources — Who Wins?                             ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Lattice: {config.lx}×{config.ly} = {config.n_qubits} qubits"
          f"{'':>{50 - len(str(config.lx)) - len(str(config.ly)) - len(str(config.n_qubits))}}║")
    print(f"║  GPU mode: {gpu_str:{52}s}║")
    print("╚══════════════════════════════════════════════════════════════════╝")


# ─────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────

def build_config():
    """Parse CLI args and return Config."""
    use_gpu = '--gpu' in sys.argv
    small = '--small' in sys.argv
    chain = '--chain' in sys.argv

    if small:
        if chain:
            return Config(
                lx=16, ly=1,
                n_trotter_steps=6,
                chi_low=16, chi_high=32,
                use_gpu=use_gpu,
            )
        else:
            return Config(
                lx=4, ly=4,
                n_trotter_steps=6,
                chi_low=16, chi_high=32,
                use_gpu=use_gpu,
            )
    else:
        if chain:
            return Config(
                lx=36, ly=1,
                n_trotter_steps=8,
                chi_low=16, chi_high=64,
                use_gpu=use_gpu,
            )
        else:
            return Config(
                lx=6, ly=6,
                n_trotter_steps=8,
                chi_low=16, chi_high=64,
                use_gpu=use_gpu,
            )

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = build_config()
    banner(config)
    total_start = time.time()

    small = '--small' in sys.argv
    chain = '--chain' in sys.argv
    n = config.n_qubits
    bonds = get_nn_bonds(config.lx, config.ly)

    # ── Observable: ZZ on the central bond ──
    cx, cy = config.lx // 2, config.ly // 2
    q_i = site_index(cx, cy, config.ly)
    q_j = site_index(cx, cy + 1, config.ly)
    obs_dict = {q_i: PAULI_Z, q_j: PAULI_Z}
    obs_str = build_pauli_observable(n, {q_i: 'Z', q_j: 'Z'})
    fixed_depth = config.n_trotter_steps

    print(f"\n  Observable: ⟨Z_{q_i} Z_{q_j}⟩  "
          f"(central bond at ({cx},{cy})↔({cx},{cy+1}))")
    print(f"  Trotter depth: {fixed_depth}  |  dt = {config.dt:.4f}")

    # ── Act 1: Reference value (cached) ──
    header("ACT 1: REFERENCE VALUE")
    if chain:
        ref_label = """exact value known: \n
                        ⟨Z_i Z_{i+1}⟩ = -Γ_{2i+1, 2i+2}          (nearest-neighbour, simplest case) \n                                                                                                                                                     │)
                        ⟨Z_i Z_j⟩ = Pfaffian(Γ_{2i+1..2j, 2i+1..2j})   (general case, Wick's theorem)  """
    else:
        ref_label = "exact statevector " if n <= SV_LIMIT else f"MPS chi={config.chi_high * 2}"

    cache_key = hashlib.md5(
        f"{config.lx},{config.ly},{config.j_coupling},{config.h_field},"
        f"{config.t_total},{config.n_trotter_steps},{obs_str}".encode()
    ).hexdigest()
    cache_path = os.path.join(SCRIPT_DIR, '.reference_cache.json')

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    if cache_key in cache:
        ref_value = cache[cache_key]
        print(f"  Loaded from cache ({ref_label})")
        ref_time = 0.0
    else:
        print(f"  Computing via {ref_label} ...")
        t0 = time.time()
        ref_value = compute_reference(config, fixed_depth, bonds, obs_str)
        ref_time = time.time() - t0
        cache[cache_key] = ref_value
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
        print(f"  Cached for future runs.")

    print(f"\n  ⟨Z_{q_i} Z_{q_j}⟩ = {ref_value:+.6f}"
          + (f"  ({ref_time:.2f}s)" if ref_time else "  (cached)"))

    # ── Act 2: MPS accuracy sweep ──
    header("ACT 2: MPS — Accuracy vs Bond Dimension χ")
    if small:
        chi_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    else:
        chi_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    print(f"  χ values: {chi_values}")
    reference_parameter = "exact" if chain else "statevector" if n <= SV_LIMIT else f"MPS chi={config.chi_high * 2}"
    print(f"  Reference {reference_parameter}\n")

    t0 = time.time()
    mps_results = sweep_mps_accuracy(
        config, fixed_depth, bonds, obs_str, ref_value, chi_values
    )
    mps_time = time.time() - t0

    print(f"  Done in {mps_time:.1f}s")
    print(f"\n  {'χ':>6}  {'⟨ZZ⟩':>12}  {'MAE':>10}  {'Time(s)':>10}")
    print(f"  {'──':>6}  {'────':>12}  {'──────────':>10}  {'──────────':>10}")
    for r in mps_results:
        print(f"  {r['chi']:>6d}  {r['value']:>+12.6f}  {r['mae']:>10.6f}  {r['elapsed']:>10.3f}")

    shadows_limit_time = 2 * mps_time

    # ── Act 3: Shadow accuracy sweep ──
    header("ACT 3: CLASSICAL SHADOWS — Accuracy vs Snapshots M")
    if small:
        m_values = [10, 25, 50, 100, 200, 400, 800]
    else:
        m_values = [10, 25, 50, 100, 200]

    print(f"  M values: {m_values}")
    backend_label = "statevector" if n <= SV_LIMIT else f"MPS chi={config.chi_high}"
    print(f"  Each snapshot: TFIM circuit + random Clifford layer + 1-shot {backend_label}\n")

    t0 = time.time()
    shadow_results = sweep_shadow_accuracy(
        config, fixed_depth, bonds, obs_dict, obs_str, ref_value, m_values, shadows_limit_time
    )
    shadow_time = time.time() - t0

    print(f"\n  Done in {shadow_time:.1f}s")
    print(f"\n  {'M':>6}  {'Mean ⟨ZZ⟩':>12}  {'Std err':>10}  {'MAE':>10}")
    print(f"  {'──────':>6}  {'────────────':>12}  {'──────────':>10}  {'──────────':>10}")
    for r in shadow_results:
        print(f"  {r['m']:>6d}  {r['mean']:>+12.6f}  {r['std_err']:>10.6f}  {r['mae']:>10.6f}")

    # ── Act 4: Plot + summary ──
    header("ACT 4: VISUALIZING THE COMPARISON")
    obs_label = f'⟨Z_{q_i} Z_{q_j}⟩'
    plot_path = plot_accuracy_vs_resources(
        shadow_results, mps_results,
        obs_label, ref_value, config,
        os.path.join(SCRIPT_DIR, 'accuracy_vs_resources.png'),
    )
    print(f"  Saved: {plot_path}")

    # Summary
    # eps = 0.05
    # m_thresh = next((r['m'] for r in shadow_results if r['mae'] < eps), None)
    # chi_thresh = next((r['chi'] for r in mps_results if r['mae'] < eps), None)

    total_elapsed = time.time() - total_start

    header("SUMMARY")
    print(f"  System: {config.lx}×{config.ly} = {n} qubits | depth = {fixed_depth}")
    print(f"  GPU: {'Yes' if config.use_gpu else 'No'}")
    print(f"  Reference ⟨Z_{q_i} Z_{q_j}⟩ = {ref_value:+.6f}  ({ref_label})")
    print(f"  Total runtime: {total_elapsed:.1f}s\n")

    # if m_thresh:
    #     print(f"  Shadows reach MAE < {eps} at M = {m_thresh}")

    best_shadow = min(shadow_results, key=lambda r: r['mae'])
    print(f"  Shadows best MAE = {best_shadow['mae']:.4f} at M = {best_shadow['m']}")
    best_mps = min(mps_results, key=lambda r: r['mae'])
    print(f"  MPS     best MAE = {best_mps['mae']:.4f} at chi = {best_mps['chi']}")

    # if chi_thresh:
    #     print(f"  MPS     reaches MAE < {eps} at χ = {chi_thresh}")
    #     best_mps = min(mps_results, key=lambda r: r['mae'])
    #     print(f"  MPS best MAE = {best_mps['mae']:.4f} at χ = {best_mps['chi']}")

    if best_mps['mae'] < best_shadow['mae']:
        print(f"\n  → MPS wins!")
    elif best_mps['mae'] > best_shadow['mae']:
        print(f"\n  → Classical shadows wins!")
    else:
        print(f"\n  → Tie!")

    print(f"\n  Plot: {plot_path}")
