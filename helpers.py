"""
Classical Shadows — Helper Functions
=====================================

Reusable building blocks for the Maestro classical shadows showcase.
Each function does one thing, takes explicit arguments, returns results.
No globals, no side effects beyond circuit construction.
"""

import time
import numpy as np
import maestro
from maestro.circuits import QuantumCircuit
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SV_LIMIT = 25


# ─────────────────────────────────────────────────────────────────────
# Pauli matrices
# ─────────────────────────────────────────────────────────────────────

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Single source of truth for all simulation parameters."""
    # Lattice
    lx: int = 6
    ly: int = 6

    # Hamiltonian
    j_coupling: float = 1.0       # ZZ coupling
    h_field: float = 1.0          # Transverse field

    # Time evolution
    t_total: float = 2.0
    n_trotter_steps: int = 8

    # MPS bond dimensions
    chi_low: int = 16
    chi_high: int = 64
    entanglement_threshold: float = 0.5

    # Hardware
    use_gpu: bool = False

    # Derived (computed in __post_init__)
    n_qubits: int = field(init=False)
    dt: float = field(init=False)

    def __post_init__(self):
        self.n_qubits = self.lx * self.ly
        self.dt = self.t_total / self.n_trotter_steps

    @property
    def simulator_type(self):
        """Backend selector: GPU or CPU."""
        return (maestro.SimulatorType.Gpu if self.use_gpu
                else maestro.SimulatorType.QCSim)


# ─────────────────────────────────────────────────────────────────────
# Lattice geometry
# ─────────────────────────────────────────────────────────────────────

def site_index(x, y, ly):
    """Map 2D lattice coordinate (x, y) to linear qubit index."""
    return x * ly + y


def site_coords(idx, ly):
    """Map linear qubit index to 2D lattice coordinate (x, y)."""
    return idx // ly, idx % ly


def get_nn_bonds(lx: int, ly: int) -> List[Tuple[int, int]]:
    """Nearest-neighbor bonds on an LX×LY 2D rectangular lattice."""
    bonds = []
    for x in range(lx):
        for y in range(ly):
            idx = site_index(x, y, ly)
            if x + 1 < lx:
                bonds.append((idx, site_index(x + 1, y, ly)))
            if y + 1 < ly:
                bonds.append((idx, site_index(x, y + 1, ly)))
    return bonds


# ─────────────────────────────────────────────────────────────────────
# Circuit construction
# ─────────────────────────────────────────────────────────────────────

CLIFFORD_GATES = ['I', 'H', 'HS']#, 'SH', 'HSdg', 'SHSdg']


def apply_clifford_gate(qc: QuantumCircuit, qubit: int, label: str):
    """Apply a named single-qubit Clifford gate to a circuit."""
    if label == 'I':
        pass
    elif label == 'H':
        qc.h(qubit)
    elif label == 'HS':
        qc.s(qubit)
        qc.h(qubit)
    elif label == 'SH':
        qc.h(qubit)
        qc.s(qubit)
    elif label == 'HSdg':
        qc.sdg(qubit)
        qc.h(qubit)
    elif label == 'SHSdg':
        qc.sdg(qubit)
        qc.h(qubit)
        qc.s(qubit)


def build_tfim_trotter_circuit(
    n_qubits: int,
    bonds: List[Tuple[int, int]],
    j: float, h: float, dt: float,
    n_steps: int,
) -> QuantumCircuit:
    """
    Build a TFIM Trotterized time-evolution circuit.

    Prepares |+⟩^n, then applies `n_steps` first-order Trotter layers:
      exp(-i J dt ZZ) on each bond, exp(-i h dt X) on each qubit.
    """
    qc = QuantumCircuit()
    for q in range(n_qubits):
        qc.h(q)
    for _ in range(n_steps):
        for q1, q2 in bonds:
            qc.cx(q1, q2)
            qc.rz(q2, 2.0 * j * dt)
            qc.cx(q1, q2)
        for q in range(n_qubits):
            qc.h(q)
            qc.rz(q, 2.0 * h * dt)
            qc.h(q)
    return qc


def append_random_clifford_layer(
    qc: QuantumCircuit, n_qubits: int, rng: np.random.Generator
) -> List[str]:
    """
    Append a random single-qubit Clifford to each qubit.
    Returns the list of Clifford labels (needed for shadow reconstruction).
    """
    labels = []
    for q in range(n_qubits):
        label = rng.choice(CLIFFORD_GATES)
        labels.append(label)
        apply_clifford_gate(qc, q, label)
    return labels


# ─────────────────────────────────────────────────────────────────────
# Clifford unitary matrices
# ─────────────────────────────────────────────────────────────────────

def clifford_unitary_matrix(label: str) -> np.ndarray:
    """Return the 2×2 unitary matrix for a single-qubit Clifford gate."""
    I = np.eye(2, dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
    gates = {
        'I': I, 'H': H, 'HS': H @ S, 'SH': S @ H,
        'HSdg': H @ Sdg, 'SHSdg': S @ H @ Sdg,
    }
    return gates[label]


# ─────────────────────────────────────────────────────────────────────
# Pauli observable builder
# ─────────────────────────────────────────────────────────────────────

def build_pauli_observable(n_qubits: int, ops: dict) -> str:
    """Build a Pauli string: ops is a dict {qubit_idx: 'X'|'Y'|'Z'}."""
    pauli_chars = []
    for q in range(n_qubits):
        pauli_chars.append(ops.get(q, 'I'))
    return ''.join(pauli_chars)


# ─────────────────────────────────────────────────────────────────────
# Shadow estimation: single snapshot
# ─────────────────────────────────────────────────────────────────────

def estimate_pauli_from_snapshot(
    bits: List[int],
    clifford_labels: List[str],
    observable_dict: Dict[int, np.ndarray],
) -> float:
    """
    Estimate Tr(O ρ) from a single classical shadow snapshot.

    Uses the factorized product formula for local observables:
      Tr(O ρ̂) = ∏_{j ∈ supp(O)} Tr(O_j ρ̂_j)
    where ρ̂_j = 3 U_j† |b_j⟩⟨b_j| U_j − I  (single-qubit shadow).

    observable_dict: {qubit_index: 2×2 Pauli matrix}
    """
    result = 1.0
    for q, pauli_mat in observable_dict.items():
        b = bits[q]
        ket = np.array([[1 - b], [b]], dtype=complex)
        proj = ket @ ket.conj().T
        U = clifford_unitary_matrix(clifford_labels[q])
        shadow_q = 3.0 * (U.conj().T @ proj @ U) - np.eye(2, dtype=complex)
        result *= np.real(np.trace(pauli_mat @ shadow_q))
    return float(result)


# ─────────────────────────────────────────────────────────────────────
# Reference value computation
# ─────────────────────────────────────────────────────────────────────

def compute_reference(
    config: Config,
    n_steps: int,
    bonds: List[Tuple[int, int]],
    obs_str: str,
) -> float:
    """
    Compute reference ⟨O⟩ at fixed Trotter depth.

    For n ≤ 20: exact statevector simulation (numpy). obs_str must
    be an I/Z-only Pauli string for this path.
    For n > 20: MPS with chi = chi_high * 2.
    """
    n = config.n_qubits

    if config.ly == 1:
        z_indices = [i for i, char in enumerate(obs_str) if char == 'Z']
        if len(z_indices) != 2:
            raise ValueError("Free fermion fast-path requires exactly two Z operators.")

        q_i, q_j = z_indices[0], z_indices[1]
        if q_j != q_i + 1:
            raise ValueError("Free fermion fast-path currently only supports nearest-neighbor bonds.")

        # Precompute rotation angles
        theta_zz = 2.0 * config.j_coupling * config.dt
        cos_zz, sin_zz = np.cos(theta_zz), np.sin(theta_zz)

        theta_x = 2.0 * config.h_field * config.dt
        cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)

        # Initialize the two observable vectors for Z_i Z_{i+1}
        a, b = 2 * q_i + 1, 2 * q_j
        v_a = np.zeros(2 * n, dtype=float)
        v_b = np.zeros(2 * n, dtype=float)
        v_a[a] = 1.0
        v_b[b] = 1.0

        # Heisenberg picture: evolve observables backwards
        for _ in range(n_steps):
            # Reverse of X gates
            for k in reversed(range(n)):
                i, j = 2 * k, 2 * k + 1
                va_i, va_j = v_a[i], v_a[j]
                v_a[i] = va_i * cos_x - va_j * sin_x
                v_a[j] = va_i * sin_x + va_j * cos_x

                vb_i, vb_j = v_b[i], v_b[j]
                v_b[i] = vb_i * cos_x - vb_j * sin_x
                v_b[j] = vb_i * sin_x + vb_j * cos_x

            # Reverse of ZZ gates
            for k in reversed(range(n - 1)):
                i, j = 2 * k + 1, 2 * k + 2
                va_i, va_j = v_a[i], v_a[j]
                v_a[i] = va_i * cos_zz - va_j * sin_zz
                v_a[j] = va_i * sin_zz + va_j * cos_zz

                vb_i, vb_j = v_b[i], v_b[j]
                v_b[i] = vb_i * cos_zz - vb_j * sin_zz
                v_b[j] = vb_i * sin_zz + vb_j * cos_zz

        # ⟨Z_i Z_j⟩ = - (v_a * Gamma(0) * v_b^T)
        expectation = -np.sum(v_a[1::2] * v_b[0::2] - v_a[0::2] * v_b[1::2])
        return float(expectation)

    if n <= SV_LIMIT:
        qc = build_tfim_trotter_circuit(
            n, bonds, config.j_coupling, config.h_field, config.dt, n_steps
        )
        result = qc.estimate(
            simulator_type=config.simulator_type,
            simulation_type=maestro.SimulationType.Statevector,
            observables=[obs_str],
        )
        return float(result['expectation_values'][0])

    else:
        chi_ref = config.chi_high * 2
        qc = build_tfim_trotter_circuit(
            n, bonds, config.j_coupling, config.h_field, config.dt, n_steps
        )
        result = qc.estimate(
            simulator_type=config.simulator_type,
            simulation_type=maestro.SimulationType.MatrixProductState,
            observables=[obs_str],
            max_bond_dimension=chi_ref,
        )
        return float(result['expectation_values'][0])


# ─────────────────────────────────────────────────────────────────────
# Shadow accuracy sweep
# ─────────────────────────────────────────────────────────────────────

def sweep_shadow_accuracy(
        config: Config,
        n_steps: int,
        bonds: List[Tuple[int, int]],
        obs_dict: Dict[int, np.ndarray],
        obs_str: str,
        ref_value: float,
        m_values: List[int],
        shadow_limit_time: float,
) -> List[dict]:
    """
    Sweep shadow snapshot counts M and measure MAE vs reference.
    """
    n = config.n_qubits

    max_m = max(m_values)
    total_snapshots = max_m
    estimates = []

    use_sv = n <= SV_LIMIT
    backend_label = "statevector" if use_sv else f"MPS chi={config.chi_high}"
    print(f"  Collecting {total_snapshots} shadow snapshots ({backend_label})...")

    t_start = time.time()
    snapshot_elapsed = []

    # Generate the pool of 800 snapshots
    for s_idx in range(total_snapshots):
        if (s_idx + 1) % 100 == 0:
            print(f"    {s_idx + 1}/{total_snapshots}")
        rng = np.random.default_rng(seed=s_idx)

        qc = build_tfim_trotter_circuit(
            n, bonds, config.j_coupling, config.h_field, config.dt, n_steps
        )
        labels = append_random_clifford_layer(qc, n, rng)
        qc.measure_all()

        if use_sv:
            result = qc.execute(
                simulator_type=config.simulator_type,
                simulation_type=maestro.SimulationType.Statevector,
                shots=1,
            )
        else:
            result = qc.execute(
                simulator_type=config.simulator_type,
                simulation_type=maestro.SimulationType.MatrixProductState,
                shots=1,
                max_bond_dimension=config.chi_high,
            )

        bitstring = list(result['counts'].keys())[0]
        bits = [int(b) for b in bitstring[:n]]
        est = estimate_pauli_from_snapshot(bits, labels, obs_dict)
        estimates.append(est)
        snapshot_elapsed.append(time.time() - t_start)

    results = []
    estimates_arr = np.array(estimates)
    n_bootstraps = 50  # Number of resampling blocks

    for m in m_values:
        block_maes = []
        for seed in range(n_bootstraps):
            # Randomly sample 'm' estimates from our pool of 800 with replacement
            rng_boot = np.random.default_rng(seed=(m * n_bootstraps + seed))
            sample_indices = rng_boot.choice(total_snapshots, size=m, replace=True)
            block_mean = float(np.mean(estimates_arr[sample_indices]))
            block_maes.append(abs(block_mean - ref_value))

        mae = float(np.mean(block_maes))
        std_err = float(np.std(block_maes) / np.sqrt(n_bootstraps))

        # Original simple mean for the raw value output
        mean = float(np.mean(estimates[:m]))

        results.append({'m': m, 'mean': mean, 'std_err': std_err, 'mae': mae,
                        'elapsed': snapshot_elapsed[m - 1], 'n_blocks': n_bootstraps})

        # if snapshot_elapsed[m - 1] > shadow_limit_time:
        #     break

    return results

# ─────────────────────────────────────────────────────────────────────
# MPS accuracy sweep
# ─────────────────────────────────────────────────────────────────────

def sweep_mps_accuracy(
    config: Config,
    n_steps: int,
    bonds: List[Tuple[int, int]],
    obs_str: str,
    ref_value: float,
    chi_values: List[int],
) -> List[dict]:
    """
    Sweep MPS bond dimension chi and measure MAE vs reference.
    Returns list of {chi, value, mae, elapsed}.
    """
    n = config.n_qubits
    results = []

    for chi in chi_values:
        qc = build_tfim_trotter_circuit(
            n, bonds, config.j_coupling, config.h_field, config.dt, n_steps
        )
        t0 = time.time()
        res = qc.estimate(
            simulator_type=config.simulator_type,
            simulation_type=maestro.SimulationType.MatrixProductState,
            observables=[obs_str],
            max_bond_dimension=chi,
        )
        elapsed = time.time() - t0
        value = float(res['expectation_values'][0])
        mae = abs(value - ref_value)
        results.append({'chi': chi, 'value': value, 'mae': mae, 'elapsed': elapsed})

    return results


# ─────────────────────────────────────────────────────────────────────
# Accuracy vs resources plot
# ─────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_resources(
    shadow_results: List[dict],
    mps_results: List[dict],
    obs_label: str,
    ref_value: float,
    config: Config,
    save_path: str,
) -> str:
    """
    Two-panel figure comparing classical shadows and MPS accuracy:
      Left:  MAE vs M (shadow snapshots), log-log. Slope ≈ −0.5.
      Right: MAE vs χ (bond dimension), log-log. Rapid convergence.
    Both panels share the same y-axis range so the gap is obvious.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    m_vals = [r['m'] for r in shadow_results]
    shadow_maes = [max(r['mae'], 1e-8) for r in shadow_results]
    shadow_times = [r['elapsed'] for r in shadow_results]
    chi_vals = [r['chi'] for r in mps_results]
    mps_maes = [max(r['mae'], 1e-8) for r in mps_results]
    mps_times = [r['elapsed'] for r in mps_results]

    all_maes = shadow_maes + mps_maes
    ymin = min(all_maes) * 0.3
    ymax = max(all_maes) * 4.0

    TIME_COLOR = '#E65100'   # orange for time lines
    t_max = max(shadow_times[-1], max(mps_times)) * 1.15

    def _add_time_axis(base_ax, x_vals, t_vals, time_label):
        """Attach a right-side time axis with shared [0, t_max] scale."""
        axr = base_ax.twinx()
        axr.plot(x_vals, t_vals, 's--', color=TIME_COLOR, linewidth=1.5,
                 markersize=6, alpha=0.8, label=time_label)
        axr.set_ylabel('Time (s)', fontsize=10, color=TIME_COLOR)
        axr.tick_params(axis='y', labelcolor=TIME_COLOR)
        axr.set_ylim(0, t_max)

        return axr

    # ── Left: MAE vs M (log-log) ──
    ax1.loglog(m_vals, shadow_maes, 'o-', color='#7B1FA2', linewidth=2.5,
               markersize=8, label='MAE')
    M_arr = np.array(m_vals, dtype=float)
    ref_curve = shadow_maes[0] * np.sqrt(m_vals[0]) / np.sqrt(M_arr)
    ax1.loglog(M_arr, ref_curve, 'k--', alpha=0.4, linewidth=1.5, label='1/√M reference')
    ax1.set_xlabel('Shadow snapshots M', fontsize=11)
    ax1.set_ylabel('MAE vs reference', fontsize=11)
    ax1.set_title(
        f'Classical Shadows: MAE vs M\n'
        f'(depth={config.n_trotter_steps}, {config.lx}×{config.ly} TFIM)',
        fontsize=11,
    )
    ax1.set_xticks(M_arr)
    ax1.set_xticklabels([str(int(M)) for M in M_arr])
    ax1.set_ylim(ymin, ymax)
    ax1.grid(True, which='both', alpha=0.3)
    ax1_twin = _add_time_axis(ax1, m_vals, shadow_times, 'Time for M snapshots (s)')

    # ── Right: MAE vs χ (log-log) ──
    ax2.loglog(chi_vals, mps_maes, 's-', color='#1565C0', linewidth=2.5,
               markersize=8, label='MAE')
    ax2.set_xlabel('Bond dimension χ', fontsize=11)
    ax2.set_ylabel('MAE vs reference', fontsize=11)
    ax2.set_title(
        f'MPS: MAE vs χ\n'
        f'({config.n_qubits} qubits, depth={config.n_trotter_steps})',
        fontsize=11,
    )
    ax2.set_xticks(chi_vals)
    ax2.set_xticklabels([str(c) for c in chi_vals])
    ax2.set_ylim(ymin, ymax)
    ax2.grid(True, which='both', alpha=0.3)
    ax2_twin = _add_time_axis(ax2, chi_vals, mps_times, 'Time per estimate (s)')

    for ax_main, ax_twin in [(ax1, ax1_twin), (ax2, ax2_twin)]:
        lh, ll = ax_main.get_legend_handles_labels()
        lhr, llr = ax_twin.get_legend_handles_labels()
        ax_main.legend(lh + lhr, ll + llr, loc='best', fontsize=9)

    # ── ε = 0.05 threshold line ──
    # eps = 0.05
    # if ymin < eps < ymax:
    #     for ax in (ax1, ax2):
    #         ax.axhline(y=eps, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    #
    #     ax1.text(m_vals[0] * 1.3, eps * 1.5, f'ε={eps}', color='red', fontsize=9)
    #     ax2.text(chi_vals[0] * 1.3, eps * 1.5, f'ε={eps}', color='red', fontsize=9)

    # ── Footer annotation ──
    # m_thresh = next((r['m'] for r in shadow_results if r['mae'] < eps), None)
    # chi_thresh = next((r['chi'] for r in mps_results if r['mae'] < eps), None)
    # parts = []
    # if m_thresh:
    #     parts.append(f'Shadows reach ε={eps} at M={m_thresh}')
    # if chi_thresh:
    #     parts.append(f'MPS reaches ε={eps} at χ={chi_thresh}')
    # if parts:
    #     fig.text(0.5, 0.01, '  |  '.join(parts),
    #              ha='center', fontsize=10, style='italic', color='#333333')

    fig.suptitle(
        f'Classical Shadows vs MPS — {obs_label} (central bond correlation) (ref = {ref_value:.4f})',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path
