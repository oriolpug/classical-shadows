"""
Classical Shadows — Helper Functions
=====================================

Reusable building blocks for the Maestro classical shadows showcase.
Each function does one thing, takes explicit arguments, returns results.
No globals, no side effects beyond circuit construction.
"""

import numpy as np
import maestro
from maestro.circuits import QuantumCircuit
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    h_field: float = 1.0          # Transverse field (ordered phase)

    # Time evolution
    t_total: float = 2.0
    n_trotter_steps: int = 10

    # MPS bond dimensions
    chi_low: int = 16             # CPU low-bond stage
    chi_high: int = 64            # High-bond stage (GPU when available)
    entanglement_threshold: float = 0.5

    # Classical shadows
    n_shadows: int = 200          # Snapshots per depth point
    n_shots: int = 1000           # Bitstrings for Act 5 demo
    subsystem_size: int = 2       # Qubits in subsystem A

    # Sweep
    trotter_depths: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 10]
    )

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

    @property
    def d_A(self):
        """Hilbert space dimension of subsystem A."""
        return 2 ** self.subsystem_size


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

CLIFFORD_GATES = ['I', 'H', 'HS', 'SH', 'HSdg', 'SHSdg']


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


def build_clifford_tfim_circuit(
    n_qubits: int,
    bonds: List[Tuple[int, int]],
    n_layers: int,
) -> QuantumCircuit:
    """
    Build a Clifford approximation of the TFIM Trotter circuit.

    Mirrors the real Trotter structure but replaces Rz(θ) → S (= Rz(π/2)),
    keeping everything Clifford for PP compatibility.
    Structure per layer: CNOT-S-CNOT on each bond (ZZ term) + H-S-H on each qubit (X term).
    """
    qc = QuantumCircuit()
    for q in range(n_qubits):
        qc.h(q)
    for _ in range(n_layers):
        # ZZ interaction (Clifford approximation: Rz(θ) → S)
        for q1, q2 in bonds:
            qc.cx(q1, q2)
            qc.s(q2)
            qc.cx(q1, q2)
        # Transverse field (Clifford approximation: Rz(θ) → S)
        for q in range(n_qubits):
            qc.h(q)
            qc.s(q)
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
# PP Scout: identify the most entangled subsystem
# ─────────────────────────────────────────────────────────────────────

def scout_entanglement(
    config: Config,
) -> dict:
    """
    Scout phase: use Pauli Propagator on a single TFIM Trotter step to
    identify the most and least entangled subsystems.

    Runs the *real* TFIM circuit (with non-Clifford Rz gates) for just
    one Trotter step. At depth=1 the branching is small enough that PP
    handles it instantly. Computes ⟨Z_i Z_j⟩ for all nearest-neighbor bonds.

    Selects subsystems by coordination-weighted entanglement score:
      score(i,j) = |⟨Z_i Z_j⟩| × (nn_i + nn_j) / 8
    Bulk bonds (high coordination) that are already correlated → hottest.
    Corner bonds (low coordination) → coldest.

    Returns dict with 'hot_qubits', 'cold_qubits', and full correlation data.
    """
    n = config.n_qubits
    bonds = get_nn_bonds(config.lx, config.ly)

    # Build single-step real TFIM circuit
    qc = build_tfim_trotter_circuit(
        n, bonds, config.j_coupling, config.h_field, config.dt, n_steps=1
    )

    # Compute ⟨Z_i Z_j⟩ for all bonds via PP (instant at depth=1)
    observables = [
        build_pauli_observable(n, {q1: 'Z', q2: 'Z'})
        for q1, q2 in bonds
    ]
    result = qc.estimate(
        simulation_type=maestro.SimulationType.PauliPropagator,
        observables=observables,
    )
    exp_vals = result['expectation_values']

    # Coordination-weighted score: bulk bonds rank higher
    def coordination(q):
        x, y = site_coords(q, config.ly)
        nn = 0
        if x > 0: nn += 1
        if x < config.lx - 1: nn += 1
        if y > 0: nn += 1
        if y < config.ly - 1: nn += 1
        return nn

    scored = []
    for (q1, q2), zz in zip(bonds, exp_vals):
        nn_sum = coordination(q1) + coordination(q2)
        score = abs(zz) * nn_sum / 8.0  # normalized by max coordination sum
        scored.append(((q1, q2), zz, score, nn_sum))

    scored.sort(key=lambda x: x[2], reverse=True)

    hot_bond = scored[0]
    cold_bond = scored[-1]

    return {
        'hot_qubits': list(hot_bond[0]),
        'hot_corr': hot_bond[1],
        'hot_score': hot_bond[2],
        'hot_coord': hot_bond[3],
        'cold_qubits': list(cold_bond[0]),
        'cold_corr': cold_bond[1],
        'cold_score': cold_bond[2],
        'cold_coord': cold_bond[3],
        'scored_bonds': scored,
    }


# ─────────────────────────────────────────────────────────────────────
# Classical shadows: reconstruction and estimation
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


def build_shadow_snapshot(
    bits: List[int],
    clifford_labels: List[str],
    subsystem_qubits: List[int],
) -> np.ndarray:
    """
    Build the reduced shadow density matrix ρ̂_A for an arbitrary subsystem.

    ρ̂_A = ⊗_{q∈A} (3 U_q† |b_q⟩⟨b_q| U_q − I)   [Huang et al. 2020]

    subsystem_qubits: list of qubit indices making up subsystem A.
    """
    single_qubit_shadows = []
    for q in subsystem_qubits:
        b = bits[q] if q < len(bits) else 0
        ket = np.array([[1 - b], [b]], dtype=complex)
        proj = ket @ ket.conj().T
        U = clifford_unitary_matrix(clifford_labels[q])
        shadow_q = 3.0 * (U.conj().T @ proj @ U) - np.eye(2, dtype=complex)
        single_qubit_shadows.append(shadow_q)

    rho = single_qubit_shadows[0]
    for i in range(1, len(single_qubit_shadows)):
        rho = np.kron(rho, single_qubit_shadows[i])
    return rho


def estimate_purity_from_shadows(shadows: List[np.ndarray]) -> float:
    """
    Unbiased U-statistics estimator for Tr(ρ_A²).
    Uses only cross-terms: Tr(ρ²) ≈ (2/M(M-1)) Σ_{i<j} Tr(ρ̂_i ρ̂_j)
    """
    d_A = shadows[0].shape[0]
    running_sum = np.zeros((d_A, d_A), dtype=complex)
    cross_total = 0.0
    for i, rho in enumerate(shadows):
        if i > 0:
            cross_total += np.real(np.trace(rho @ running_sum))
        running_sum += rho
    M = len(shadows)
    return float((2.0 * cross_total) / (M * (M - 1)))


def renyi_s2(purity: float, d_A: int) -> Tuple[float, float]:
    """Compute S₂ = -log₂(purity), with clamping. Returns (S₂, clamped_purity)."""
    clamped = float(np.clip(purity, 1.0 / d_A, 1.0))
    return -np.log2(clamped), clamped


# ─────────────────────────────────────────────────────────────────────
# Shadow snapshot collection
# ─────────────────────────────────────────────────────────────────────

def collect_shadow_snapshots(
    config: Config,
    n_trotter_steps: int,
    bonds: List[Tuple[int, int]],
    subsystem_qubits: List[int],
    verbose: bool = True,
) -> List[np.ndarray]:
    """
    Collect M shadow snapshots for a given Trotter depth and subsystem.

    Each snapshot: build circuit → random Clifford layer → measure → reconstruct ρ̂_A.
    """
    n = config.n_qubits
    shadows = []

    for s_idx in range(config.n_shadows):
        rng = np.random.default_rng(seed=s_idx)

        qc = build_tfim_trotter_circuit(
            n, bonds, config.j_coupling, config.h_field, config.dt, n_trotter_steps
        )
        labels = append_random_clifford_layer(qc, n, rng)
        qc.measure_all()

        result = qc.execute(
            simulator_type=config.simulator_type,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=1,
            max_bond_dimension=config.chi_high if config.use_gpu else config.chi_low,
        )
        bitstring = list(result['counts'].keys())[0]
        bits = [int(b) for b in bitstring[:n]]

        rho = build_shadow_snapshot(bits, labels, subsystem_qubits)
        shadows.append(rho)

        if verbose and (s_idx + 1) % 100 == 0:
            print(f"    Collected {s_idx + 1}/{config.n_shadows} snapshots...")

    return shadows


# ─────────────────────────────────────────────────────────────────────
# Exact reference (statevector ED)
# ─────────────────────────────────────────────────────────────────────

def compute_exact_s2(
    config: Config,
    subsystem_qubits: Optional[List[int]] = None,
) -> Optional[dict]:
    """
    Compute exact S₂(t) for an arbitrary subsystem via full statevector simulation.

    subsystem_qubits: list of qubit indices. Defaults to [0, 1, ..., subsystem_size-1].
    Returns None if system is too large (n > 20).
    """
    n = config.n_qubits
    if n > 20:
        return None

    if subsystem_qubits is None:
        subsystem_qubits = list(range(config.subsystem_size))

    bonds = get_nn_bonds(config.lx, config.ly)
    A_size = len(subsystem_qubits)
    d_A = 2 ** A_size

    # Gate matrices
    H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    def rz_mat(theta):
        return np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]], dtype=complex)

    def apply_single(state, gate, q):
        state = np.moveaxis(state, q, 0)
        shape = state.shape
        state = state.reshape(2, -1)
        state = gate @ state
        state = state.reshape(shape)
        return np.moveaxis(state, 0, q)

    def apply_cx(state, ctrl, targ):
        n_q = len(state.shape)
        idx_1 = [slice(None)] * n_q
        idx_1[ctrl] = slice(1, 2)
        block1 = state[tuple(idx_1)].copy()
        state_t = np.moveaxis(block1, targ, 0)
        shape = state_t.shape
        state_t = state_t.reshape(2, -1)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        state_t = X @ state_t
        state_t = state_t.reshape(shape)
        block1_new = np.moveaxis(state_t, 0, targ)
        result = state.copy()
        result[tuple(idx_1)] = block1_new
        return result

    def trotter_step(state):
        for q1, q2 in bonds:
            state = apply_cx(state, q1, q2)
            state = apply_single(state, rz_mat(2.0 * config.j_coupling * config.dt), q2)
            state = apply_cx(state, q1, q2)
        for q in range(n):
            state = apply_single(state, H_gate, q)
            state = apply_single(state, rz_mat(2.0 * config.h_field * config.dt), q)
            state = apply_single(state, H_gate, q)
        return state

    def compute_s2_from_state(state):
        # Partial trace for arbitrary subsystem qubits:
        # move subsystem qubits to front, reshape to (d_A, d_B), then ρ_A = ψ·ψ†
        env_qubits = [q for q in range(n) if q not in subsystem_qubits]
        perm = list(subsystem_qubits) + env_qubits
        state_perm = np.transpose(state, perm)
        psi = state_perm.reshape(d_A, -1)
        rho_A = psi @ psi.conj().T
        tr_rho_sq = np.real(np.trace(rho_A @ rho_A))
        S2 = -np.log2(max(tr_rho_sq, 1.0 / d_A))
        return S2, tr_rho_sq

    # Initial state |+⟩^n
    state = np.ones((2,) * n, dtype=complex) / np.sqrt(2 ** n)

    max_depth = max(config.trotter_depths)
    s2_by_depth = {}
    for step in range(1, max_depth + 1):
        state = trotter_step(state)
        if step in config.trotter_depths:
            s2, purity = compute_s2_from_state(state)
            s2_by_depth[step] = (s2, purity)

    results = {'depths': [], 'times': [], 's2': [], 'purity': []}
    for d in config.trotter_depths:
        s2, purity = s2_by_depth[d]
        results['depths'].append(d)
        results['times'].append(d * config.dt)
        results['s2'].append(s2)
        results['purity'].append(purity)
    return results


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_energy_evolution(times, energies, backends, save_path):
    """Plot E(t) during time evolution, annotating backend handoff."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2196F3' if 'low' in b else '#E91E63' for b in backends]
    for i in range(len(times) - 1):
        ax.plot(times[i:i+2], energies[i:i+2],
                color=colors[i], linewidth=2, marker='o', markersize=5)
    switch_idx = next(
        (i for i in range(1, len(backends)) if backends[i] != backends[i-1]), None
    )
    if switch_idx is not None:
        ax.axvline(x=times[switch_idx], color='red', linestyle='--',
                   alpha=0.7, label=f'Backend handoff (t={times[switch_idx]:.2f})')
        ax.legend()
    ax.set_xlabel('Simulation Time t')
    ax.set_ylabel('Energy E(t)')
    ax.set_title(f'TFIM Energy Evolution — {len(energies)-1} Trotter Steps')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_scout_comparison(
    hot_results: dict,
    cold_results: dict,
    hot_exact: Optional[dict],
    cold_exact: Optional[dict],
    hot_qubits: List[int],
    cold_qubits: List[int],
    config: Config,
    save_path: str,
):
    """
    Plot S₂ vs time comparing the PP-scouted 'hot' subsystem against
    the 'cold' subsystem. Shows that the scout correctly identified
    where entanglement lives.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    max_s2 = config.subsystem_size

    # ── Left panel: Hot subsystem (PP-selected) ──
    if hot_exact is not None:
        ax1.fill_between(hot_exact['times'], 0, hot_exact['s2'],
                         alpha=0.12, color='green')
        ax1.plot(hot_exact['times'], hot_exact['s2'],
                 'g-', linewidth=2.5, label='Exact (statevector ED)')

    ax1.plot(hot_results['times'], hot_results['s2'], 'o--',
             color='#7B1FA2', linewidth=2, markersize=7,
             label='Classical shadows (Maestro MPS)')
    ax1.axhline(y=max_s2, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Simulation Time t')
    ax1.set_ylabel('2nd-order Rényi Entropy S₂')
    hx0, hy0 = site_coords(hot_qubits[0], config.ly)
    hx1, hy1 = site_coords(hot_qubits[1], config.ly)
    ax1.set_title(f'PP-Scouted "Hot" Subsystem\n'
                  f'Qubits {hot_qubits} — site ({hx0},{hy0}),({hx1},{hy1})')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=-0.05, top=max_s2 + 0.15)

    full_tomo = 4 ** config.n_qubits
    ax1.text(0.02, 0.97,
             f'{config.n_shadows} snapshots/depth\n'
             f'Full tomo: 4^{config.n_qubits} ≈ {full_tomo:.1e}\n'
             f'Speedup: ×{full_tomo // max(config.n_shadows, 1):.1e}',
             transform=ax1.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                       edgecolor='#2E7D32', alpha=0.85))

    # ── Right panel: Cold subsystem (contrast) ──
    if cold_exact is not None:
        ax2.fill_between(cold_exact['times'], 0, cold_exact['s2'],
                         alpha=0.12, color='green')
        ax2.plot(cold_exact['times'], cold_exact['s2'],
                 'g-', linewidth=2.5, label='Exact (statevector ED)')

    ax2.plot(cold_results['times'], cold_results['s2'], 's--',
             color='#E65100', linewidth=2, markersize=7,
             label='Classical shadows (Maestro MPS)')
    ax2.axhline(y=max_s2, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Simulation Time t')
    ax2.set_ylabel('2nd-order Rényi Entropy S₂')
    cx0, cy0 = site_coords(cold_qubits[0], config.ly)
    cx1, cy1 = site_coords(cold_qubits[1], config.ly)
    ax2.set_title(f'"Cold" Subsystem (Contrast)\n'
                  f'Qubits {cold_qubits} — site ({cx0},{cy0}),({cx1},{cy1})')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=-0.05, top=max_s2 + 0.15)

    fig.suptitle(f'Entanglement Growth — {config.lx}×{config.ly} TFIM  |  '
                 f'PP Scout → MPS Sniper',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


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
# Benchmark: efficiency metrics (shots saved, accuracy gain)
# ─────────────────────────────────────────────────────────────────────

def run_efficiency_benchmark(
    config: Config,
    hot_qubits: List[int],
    cold_qubits: List[int],
    hot_exact: Optional[dict],
    cold_exact: Optional[dict],
    shot_counts: List[int],
    benchmark_depth: int = 4,
    n_seeds: int = 3,
) -> dict:
    """
    Sweep shadow counts [M1, M2, ...] for both hot and cold subsystems
    at a fixed Trotter depth. Averages over n_seeds random seeds per M
    for smooth, reproducible curves.

    Returns dict with 'shot_counts', 'hot_mae', 'cold_mae', headline numbers.
    """
    bonds = get_nn_bonds(config.lx, config.ly)

    # Exact reference at benchmark_depth
    exact_hot_s2 = None
    exact_cold_s2 = None
    if hot_exact is not None:
        idx = hot_exact['depths'].index(benchmark_depth) if benchmark_depth in hot_exact['depths'] else -1
        if idx >= 0:
            exact_hot_s2 = hot_exact['s2'][idx]
    if cold_exact is not None:
        idx = cold_exact['depths'].index(benchmark_depth) if benchmark_depth in cold_exact['depths'] else -1
        if idx >= 0:
            exact_cold_s2 = cold_exact['s2'][idx]

    hot_mae_list = []
    cold_mae_list = []

    print(f"\n  Benchmark depth: {benchmark_depth} (t={benchmark_depth * config.dt:.2f})")
    print(f"  Exact S₂: HOT={exact_hot_s2:.4f}  COLD={exact_cold_s2:.4f}" if
          exact_hot_s2 and exact_cold_s2 else "  (No exact reference)")
    print(f"  Averaging over {n_seeds} random seeds per shot count\n")
    print(f"  {'Shots':>8}  {'HOT MAE':>10}  {'COLD MAE':>10}  {'Accuracy gain':>14}")
    print(f"  {'──────':>8}  {'──────────':>10}  {'──────────':>10}  {'──────────────':>14}")

    for M in shot_counts:
        hot_maes_seeds = []
        cold_maes_seeds = []

        for seed_offset in range(n_seeds):
            cfg_m = Config(
                lx=config.lx, ly=config.ly,
                j_coupling=config.j_coupling, h_field=config.h_field,
                t_total=config.t_total, n_trotter_steps=config.n_trotter_steps,
                chi_low=config.chi_low, chi_high=config.chi_high,
                n_shadows=M, use_gpu=config.use_gpu,
                trotter_depths=config.trotter_depths,
            )
            # Use different seed offsets per trial by seeding inside collect
            rng_offset = seed_offset * 1000  # shift seeds for this trial
            # Temporarily monkey-patch seeds by building fresh circuit each time
            hot_shadows = _collect_shadows_seeded(
                cfg_m, benchmark_depth, bonds, hot_qubits, seed_base=rng_offset
            )
            cold_shadows = _collect_shadows_seeded(
                cfg_m, benchmark_depth, bonds, cold_qubits, seed_base=rng_offset
            )

            hot_purity = estimate_purity_from_shadows(hot_shadows)
            hot_s2, _ = renyi_s2(hot_purity, 2 ** len(hot_qubits))
            cold_purity = estimate_purity_from_shadows(cold_shadows)
            cold_s2, _ = renyi_s2(cold_purity, 2 ** len(cold_qubits))

            if exact_hot_s2 is not None:
                hot_maes_seeds.append(abs(hot_s2 - exact_hot_s2))
            if exact_cold_s2 is not None:
                cold_maes_seeds.append(abs(cold_s2 - exact_cold_s2))

        hot_mae = float(np.mean(hot_maes_seeds)) if hot_maes_seeds else float('nan')
        cold_mae = float(np.mean(cold_maes_seeds)) if cold_maes_seeds else float('nan')
        acc_gain = (cold_mae - hot_mae) / cold_mae * 100 if cold_mae > 0 else 0.0

        hot_mae_list.append(hot_mae)
        cold_mae_list.append(cold_mae)
        print(f"  {M:>8d}  {hot_mae:>10.4f}  {cold_mae:>10.4f}  {acc_gain:>+13.1f}%")

    # Shot reduction: interpolate cold MAE to find M where cold_mae = hot_mae[-1]
    best_hot_mae = hot_mae_list[-1]  # use the final (largest M) hot MAE as target
    shot_reduction_pct = 0.0
    # Linearly interpolate cold MAE curve to find shots needed
    cold_arr = np.array(cold_mae_list)
    shot_arr = np.array(shot_counts, dtype=float)
    above = cold_arr > best_hot_mae  # indices where cold still needs more shots
    if above.any() and not above.all():
        # Find crossing point
        cross_idx = np.where(~above)[0][0]
        if cross_idx > 0:
            m1, m2 = shot_arr[cross_idx - 1], shot_arr[cross_idx]
            e1, e2 = cold_arr[cross_idx - 1], cold_arr[cross_idx]
            m_needed = float(m1 + (best_hot_mae - e1) / (e2 - e1) * (m2 - m1))
            shot_reduction_pct = max(0.0, (m_needed - shot_counts[-1]) / m_needed * 100)
    elif above.all():
        # Cold never reaches hot's accuracy — extrapolate with 1/sqrt(M)
        if len(shot_counts) >= 2 and cold_mae_list[-1] > 0:
            m_needed = shot_counts[-1] * (cold_mae_list[-1] / best_hot_mae) ** 2
            shot_reduction_pct = max(0.0, (m_needed - shot_counts[-1]) / m_needed * 100)

    # Accuracy gain at maximum shot count
    final_acc_gain = ((cold_mae_list[-1] - hot_mae_list[-1]) / cold_mae_list[-1] * 100
                      if cold_mae_list[-1] > 0 else 0.0)

    metrics = {
        'shot_counts': shot_counts,
        'hot_mae': hot_mae_list,
        'cold_mae': cold_mae_list,
        'exact_hot_s2': exact_hot_s2,
        'exact_cold_s2': exact_cold_s2,
        'best_hot_mae': best_hot_mae,
        'shot_reduction_pct': shot_reduction_pct,
        'final_accuracy_gain_pct': final_acc_gain,
        'benchmark_depth': benchmark_depth,
    }

    print(f"\n  === HEADLINE METRICS ===")
    print(f"  Accuracy gain (same M={shot_counts[-1]}):  +{final_acc_gain:.1f}% more accurate")
    print(f"  Shot reduction to match hot accuracy:  ~{shot_reduction_pct:.0f}% fewer shots")

    return metrics


def _collect_shadows_seeded(
    config: Config,
    n_trotter_steps: int,
    bonds: List[Tuple[int, int]],
    subsystem_qubits: List[int],
    seed_base: int = 0,
) -> List[np.ndarray]:
    """
    Like collect_shadow_snapshots but uses seed_base + snapshot_index as seed.
    This lets us run independent trials with different random Clifford choices.
    """
    n = config.n_qubits
    shadows = []
    for s_idx in range(config.n_shadows):
        rng = np.random.default_rng(seed=seed_base + s_idx)
        qc = build_tfim_trotter_circuit(
            n, bonds, config.j_coupling, config.h_field, config.dt, n_trotter_steps
        )
        labels = append_random_clifford_layer(qc, n, rng)
        qc.measure_all()
        result = qc.execute(
            simulator_type=config.simulator_type,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=1,
            max_bond_dimension=config.chi_high if config.use_gpu else config.chi_low,
        )
        bitstring = list(result['counts'].keys())[0]
        bits = [int(b) for b in bitstring[:n]]
        rho = build_shadow_snapshot(bits, labels, subsystem_qubits)
        shadows.append(rho)
    return shadows


def plot_lattice_heatmap(
    config: Config,
    scout: dict,
    save_path: str,
):
    """
    Plot the 2D TFIM lattice as a heatmap colored by PP scout score.
    Hot bonds are highlighted in gold, cold in blue.
    """
    lx, ly = config.lx, config.ly
    scored = scout['scored_bonds']
    hot_q = scout['hot_qubits']
    cold_q = scout['cold_qubits']

    # Build score matrix on bonds: for display, colour each site by max bond score
    site_score = np.zeros((lx, ly))
    for (q1, q2), zz, score, nn_sum in scored:
        x1, y1 = site_coords(q1, ly)
        x2, y2 = site_coords(q2, ly)
        site_score[x1, y1] = max(site_score[x1, y1], score)
        site_score[x2, y2] = max(site_score[x2, y2], score)

    fig, ax = plt.subplots(figsize=(max(5, ly * 1.2), max(5, lx * 1.2)))

    im = ax.imshow(site_score, cmap='YlOrRd', aspect='equal',
                   vmin=0, vmax=site_score.max() * 1.1)
    plt.colorbar(im, ax=ax, label='PP Scout Score (coord-weighted |⟨ZZ⟩|)')

    # Draw bonds
    for (q1, q2), zz, score, nn_sum in scored:
        x1, y1 = site_coords(q1, ly)
        x2, y2 = site_coords(q2, ly)
        alpha = 0.3 + 0.7 * (score / (site_score.max() + 1e-8))
        ax.plot([y1, y2], [x1, x2], 'k-', alpha=alpha, linewidth=1.5 * alpha)

    # Annotate sites
    for q in range(config.n_qubits):
        x, y = site_coords(q, ly)
        ax.text(y, x, str(q), ha='center', va='center', fontsize=8,
                fontweight='bold', color='white' if site_score[x, y] > site_score.max() * 0.5 else 'black')

    # Highlight hot and cold pairs
    for label, qubits, color, lw in [('HOT', hot_q, 'gold', 4), ('COLD', cold_q, 'deepskyblue', 3)]:
        x1, y1 = site_coords(qubits[0], ly)
        x2, y2 = site_coords(qubits[1], ly)
        ax.plot([y1, y2], [x1, x2], '-', color=color, linewidth=lw, zorder=5,
                label=f'{label}: qubits {qubits}')
        for x, y in [(x1, y1), (x2, y2)]:
            ax.add_patch(plt.Circle((y, x), 0.35, color=color, zorder=6, linewidth=2,
                                    fill=False))

    ax.set_xticks(range(ly))
    ax.set_yticks(range(lx))
    ax.set_xticklabels([f'y={i}' for i in range(ly)])
    ax.set_yticklabels([f'x={i}' for i in range(lx)])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'PP Scout Heatmap — {lx}×{ly} TFIM Lattice\n'
                 f'Gold = HOT (bulk, high entanglement)  |  Blue = COLD (edge)')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_efficiency_curves(
    metrics: dict,
    hot_qubits: List[int],
    cold_qubits: List[int],
    config: Config,
    save_path: str,
):
    """
    Log-log plot of MAE vs snapshot count for hot and cold subsystems.
    Shows the theoretical 1/sqrt(M) scaling and the efficiency crossover.
    """
    shot_counts = metrics['shot_counts']
    hot_mae = metrics['hot_mae']
    cold_mae = metrics['cold_mae']

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.loglog(shot_counts, hot_mae, 'o-', color='#7B1FA2', linewidth=2.5,
              markersize=8, label=f'HOT subsystem {hot_qubits} (PP-scouted bulk)')
    ax.loglog(shot_counts, cold_mae, 's-', color='#E65100', linewidth=2.5,
              markersize=8, label=f'COLD subsystem {cold_qubits} (un-scouted edge)')

    # 1/sqrt(M) reference line
    M = np.array(shot_counts, dtype=float)
    ref = cold_mae[0] * np.sqrt(float(shot_counts[0])) / np.sqrt(M)
    ax.loglog(M, ref, 'k--', alpha=0.4, linewidth=1.5, label='1/√M reference')

    # Annotate accuracy gain at max shots
    if len(shot_counts) >= 1:
        ax.annotate(
            f'+{metrics["final_accuracy_gain_pct"]:.0f}% accuracy\n(same shot budget)',
            xy=(shot_counts[-1], hot_mae[-1]),
            xytext=(shot_counts[-1] * 0.45, hot_mae[-1] * 2.5),
            fontsize=9, color='#7B1FA2', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5),
        )

    ax.set_xlabel('Number of Shadow Snapshots (M)', fontsize=11)
    ax.set_ylabel('Mean Absolute Error vs Exact ED', fontsize=11)
    ax.set_title(
        f'PP Scout Targeting Efficiency\n'
        f'{config.lx}×{config.ly} TFIM — S₂ estimation at Trotter depth {metrics["benchmark_depth"]}',
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_savings_summary(
    metrics: dict,
    scout_time_s: float,
    total_time_s: float,
    config: Config,
    save_path: str,
):
    """
    Clean infographic summarising the three headline efficiency metrics:
    accuracy gain, shot reduction, and scout overhead.
    """
    acc_gain = metrics['final_accuracy_gain_pct']
    shot_red = metrics['shot_reduction_pct']
    scout_overhead = scout_time_s / total_time_s * 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#0D1117')

    cards = [
        {
            'ax': axes[0],
            'value': f'+{acc_gain:.0f}%',
            'label': 'Accuracy Gain',
            'sub': f'HOT subsystem is {acc_gain:.0f}% more\naccurate than naive targeting\n(same snapshot budget)',
            'color': '#7B1FA2',
            'bg': '#1A0A2E',
        },
        {
            'ax': axes[1],
            'value': f'~{shot_red:.0f}%',
            'label': 'Fewer Shots',
            'sub': f'Reach the same accuracy\nwith ~{shot_red:.0f}% fewer shadows\nthanks to PP-guided targeting',
            'color': '#00897B',
            'bg': '#0A1E1B',
        },
        {
            'ax': axes[2],
            'value': f'{scout_overhead:.3f}%',
            'label': 'Scout Overhead',
            'sub': f'PP scout completed in {scout_time_s*1000:.1f}ms\nvs {total_time_s:.0f}s total pipeline\n(Heisenberg picture = near-zero cost)',
            'color': '#F57C00',
            'bg': '#1E0F00',
        },
    ]

    for card in cards:
        ax = card['ax']
        ax.set_facecolor(card['bg'])
        for spine in ax.spines.values():
            spine.set_edgecolor(card['color'])
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0.5, 0.65, card['value'], transform=ax.transAxes,
                fontsize=42, fontweight='bold', color=card['color'],
                ha='center', va='center')
        ax.text(0.5, 0.38, card['label'], transform=ax.transAxes,
                fontsize=14, color='white', ha='center', va='center',
                fontweight='bold')
        ax.text(0.5, 0.15, card['sub'], transform=ax.transAxes,
                fontsize=9, color='#AAAAAA', ha='center', va='center',
                linespacing=1.5)

    fig.suptitle(
        f'PP Scout → MPS Sniper  |  Classical Shadows Efficiency  '
        f'|  {config.lx}×{config.ly} TFIM ({config.n_qubits} qubits)',
        fontsize=12, color='white', y=1.01,
    )
    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────
# Metric 1: search cost reduction
# ─────────────────────────────────────────────────────────────────────

def compute_search_cost_table(
    base_config: Config,
    shadow_budget_per_pair: int = 300,
    lattice_sizes: Optional[List[Tuple[int, int]]] = None,
) -> List[dict]:
    """
    For each lattice size, measure actual PP scan time and compute the
    theoretical shadow cost to scan all bonds exhaustively.

    Theoretical shadow scan = n_bonds × shadow_budget_per_pair circuits.
    We estimate one circuit execution time from the base_config timing.

    Returns list of dicts, one per lattice size.
    """
    import time as _time
    if lattice_sizes is None:
        lattice_sizes = [(4, 4), (6, 6), (8, 8), (10, 10), (12, 12)]

    # Estimate single-circuit execution time from a quick benchmark
    # (just run one shadow snapshot and time it)
    bonds_base = get_nn_bonds(base_config.lx, base_config.ly)
    qc_bench = build_tfim_trotter_circuit(
        base_config.n_qubits, bonds_base,
        base_config.j_coupling, base_config.h_field, base_config.dt, n_steps=4,
    )
    rng_bench = np.random.default_rng(0)
    append_random_clifford_layer(qc_bench, base_config.n_qubits, rng_bench)
    qc_bench.measure_all()
    t0 = _time.time()
    qc_bench.execute(
        simulator_type=base_config.simulator_type,
        simulation_type=maestro.SimulationType.MatrixProductState,
        shots=1,
        max_bond_dimension=base_config.chi_low,
    )
    single_circuit_time = _time.time() - t0

    rows = []
    for lx, ly in lattice_sizes:
        n = lx * ly
        bonds = get_nn_bonds(lx, ly)
        n_bonds = len(bonds)

        # Build temporary config for this size
        tmp_config = Config(
            lx=lx, ly=ly,
            j_coupling=base_config.j_coupling,
            h_field=base_config.h_field,
            t_total=base_config.t_total,
            n_trotter_steps=base_config.n_trotter_steps,
            chi_low=base_config.chi_low,
        )

        # Measure PP scan time (actual)
        qc_pp = build_tfim_trotter_circuit(
            n, bonds, tmp_config.j_coupling, tmp_config.h_field,
            tmp_config.dt, n_steps=1,
        )
        obs = [build_pauli_observable(n, {q1: 'Z', q2: 'Z'}) for q1, q2 in bonds]
        t0 = _time.time()
        qc_pp.estimate(
            simulation_type=maestro.SimulationType.PauliPropagator,
            observables=obs,
        )
        pp_time = _time.time() - t0

        # Theoretical shadow exhaustive scan (not actually run)
        shadow_circuits = n_bonds * shadow_budget_per_pair
        shadow_time_est = shadow_circuits * single_circuit_time

        rows.append({
            'lx': lx, 'ly': ly, 'n_qubits': n,
            'n_bonds': n_bonds,
            'pp_time_ms': pp_time * 1000,
            'shadow_circuits': shadow_circuits,
            'shadow_time_s': shadow_time_est,
            'shadow_time_h': shadow_time_est / 3600,
            'speedup': shadow_time_est / max(pp_time, 1e-6),
        })

        print(f"  {lx}×{ly} ({n:3d} qubits, {n_bonds:3d} bonds):  "
              f"PP={pp_time*1000:.1f}ms  |  "
              f"Shadow-scan={shadow_circuits} circuits "
              f"≈{shadow_time_est/60:.0f}min  |  "
              f"Speedup ×{shadow_time_est/max(pp_time,1e-6):.0f}")

    return rows


def plot_search_cost(rows: List[dict], save_path: str):
    """
    Two-panel figure:
    Left:  PP scan time (ms) vs lattice size — constant/flat
    Right: Shadow exhaustive-scan time (hours) — grows with n_bonds
    Plus a table of headline speedup numbers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    n_qubits = [r['n_qubits'] for r in rows]
    pp_ms = [r['pp_time_ms'] for r in rows]
    shadow_h = [r['shadow_time_h'] for r in rows]
    speedups = [r['speedup'] for r in rows]
    labels = [f"{r['lx']}×{r['ly']}" for r in rows]

    # Left: PP time
    ax1.bar(labels, pp_ms, color='#7B1FA2', alpha=0.85)
    ax1.set_ylabel('PP Scout Time (ms)', fontsize=11)
    ax1.set_xlabel('Lattice size', fontsize=11)
    ax1.set_title('PP Scout: Near-Constant Cost\n(Heisenberg picture, O(n·d))', fontsize=12)
    ax1.set_ylim(0, max(pp_ms) * 1.5)
    for i, (v, lbl) in enumerate(zip(pp_ms, labels)):
        ax1.text(i, v + max(pp_ms) * 0.05, f'{v:.1f}ms', ha='center', fontsize=9,
                 fontweight='bold', color='#7B1FA2')
    ax1.grid(axis='y', alpha=0.3)

    # Right: Shadow exhaustive-scan time
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(rows)))
    bars = ax2.bar(labels, shadow_h, color=colors)
    ax2.set_ylabel('Shadow Exhaustive Scan (hours)', fontsize=11)
    ax2.set_xlabel('Lattice size', fontsize=11)
    ax2.set_title('Exhaustive Shadow Scan: O(n²) Growth\n'
                  '(all bonds × 300 snapshots each)', fontsize=12)
    for i, (v, sx) in enumerate(zip(shadow_h, speedups)):
        ax2.text(i, v + max(shadow_h) * 0.02,
                 f'{v:.1f}h\n(×{sx:.0f} slower)', ha='center', fontsize=8,
                 fontweight='bold', color='#BF360C')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Why Use a PP Scout?\n'
                 'Pauli Propagator eliminates the O(n²) pair-search problem',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────
# Metric 3: PP ranking accuracy vs exact S₂
# ─────────────────────────────────────────────────────────────────────

def compute_exact_s2_all_bonds(config: Config, depth: int) -> List[Tuple[Tuple, float]]:
    """
    Compute exact S₂ for every nearest-neighbor bond (2-qubit subsystem)
    at a fixed Trotter depth via statevector simulation.

    Returns list of ((q1, q2), s2) sorted by s2 descending.
    Only feasible for n_qubits <= 20.
    """
    if config.n_qubits > 20:
        raise ValueError("Exact S₂ for all bonds only feasible for n <= 20 qubits")

    n = config.n_qubits
    bonds = get_nn_bonds(config.lx, config.ly)
    H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    def rz_mat(theta):
        return np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]], dtype=complex)

    def apply_single(state, gate, q):
        state = np.moveaxis(state, q, 0)
        shape = state.shape
        state = (gate @ state.reshape(2, -1)).reshape(shape)
        return np.moveaxis(state, 0, q)

    def apply_cx(state, ctrl, targ):
        n_q = len(state.shape)
        idx_1 = [slice(None)] * n_q
        idx_1[ctrl] = slice(1, 2)
        block1 = state[tuple(idx_1)].copy()
        state_t = np.moveaxis(block1, targ, 0)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        state_t = (X @ state_t.reshape(2, -1)).reshape(state_t.shape)
        result = state.copy()
        result[tuple(idx_1)] = np.moveaxis(state_t, 0, targ)
        return result

    # Evolve state to target depth
    state = np.ones((2,) * n, dtype=complex) / np.sqrt(2 ** n)
    for _ in range(depth):
        for q1, q2 in bonds:
            state = apply_cx(state, q1, q2)
            state = apply_single(state, rz_mat(2.0 * config.j_coupling * config.dt), q2)
            state = apply_cx(state, q1, q2)
        for q in range(n):
            state = apply_single(state, H_gate, q)
            state = apply_single(state, rz_mat(2.0 * config.h_field * config.dt), q)
            state = apply_single(state, H_gate, q)

    # Compute S₂ for each bond (2-qubit subsystem)
    bond_s2 = []
    for bond in bonds:
        q1, q2 = bond
        env_qubits = [q for q in range(n) if q not in (q1, q2)]
        perm = [q1, q2] + env_qubits
        psi = np.transpose(state, perm).reshape(4, -1)
        rho_A = psi @ psi.conj().T
        tr_rho_sq = max(np.real(np.trace(rho_A @ rho_A)), 0.25)
        s2 = -np.log2(tr_rho_sq)
        bond_s2.append((bond, s2))

    bond_s2.sort(key=lambda x: x[1], reverse=True)
    return bond_s2


def compute_pp_ranking_accuracy(
    config: Config,
    scout: dict,
    depth: int,
) -> dict:
    """
    Compare PP bond ranking (by score) against exact S₂ ranking.

    Computes:
    - Spearman rank correlation between PP score and true S₂
    - Top-1, Top-3, Top-5 overlap (% of true top-K bonds identified by PP)
    """
    from scipy.stats import spearmanr

    print(f"\n  Computing exact S₂ for all bonds at depth={depth} ...")
    bond_exact = compute_exact_s2_all_bonds(config, depth)

    # PP score ranking
    scored = scout['scored_bonds']
    score_by_bond = {(q1, q2): score for (q1, q2), zz, score, nn_sum in scored}
    # Also try reverse-indexed since bond ordering may differ
    score_by_bond.update({(q2, q1): score for (q1, q2), zz, score, nn_sum in scored})

    # Align: for each bond in exact ranking, get its PP score
    pp_scores = []
    exact_s2s = []
    for bond, s2 in bond_exact:
        sc = score_by_bond.get(bond, score_by_bond.get((bond[1], bond[0]), 0.0))
        pp_scores.append(sc)
        exact_s2s.append(s2)

    # Spearman rank correlation
    corr, pval = spearmanr(pp_scores, exact_s2s)

    # Top-K overlap
    pp_top_bonds_sorted = sorted(scored, key=lambda x: x[2], reverse=True)
    pp_ranking = [b[0] for b in pp_top_bonds_sorted]
    exact_ranking = [bond for bond, s2 in bond_exact]

    def normalize_bond(b):
        return tuple(sorted(b))

    overlaps = {}
    for K in [1, 3, 5]:
        exact_top_k = {normalize_bond(b) for b in exact_ranking[:K]}
        pp_top_k = {normalize_bond(b) for b in pp_ranking[:K]}
        overlap = len(exact_top_k & pp_top_k) / K * 100
        overlaps[K] = overlap

    print(f"\n  PP vs Exact S₂ Ranking:")
    print(f"  {'Rank':>5}  {'PP top bond':>12}  {'PP score':>10}  "
          f"{'Exact S₂':>10}  {'Exact rank':>10}")
    for i, ((bond, s2), (_, zz, score, _)) in enumerate(
            zip(bond_exact[:7], pp_top_bonds_sorted[:7])):
        x0, y0 = site_coords(bond[0], config.ly)
        x1, y1 = site_coords(bond[1], config.ly)
        print(f"  {i+1:>5}  ({x0},{y0})↔({x1},{y1})  "
              f"{score:>10.5f}  {s2:>10.4f}")
    print(f"\n  Spearman rank correlation: ρ = {corr:.4f}  (p={pval:.4f})")
    for K, overlap in overlaps.items():
        print(f"  Top-{K} overlap: {overlap:.0f}%")

    return {
        'bond_exact': bond_exact,
        'pp_scores': pp_scores,
        'exact_s2s': exact_s2s,
        'spearman_corr': corr,
        'spearman_pval': pval,
        'top_k_overlap': overlaps,
        'pp_ranking': pp_ranking,
        'exact_ranking': exact_ranking,
        'depth': depth,
    }


def plot_ranking_accuracy(
    ranking_result: dict,
    config: Config,
    save_path: str,
):
    """
    Two-panel figure showing PP ranking accuracy vs exact ED:
    Left:  Scatter plot of PP score vs true S₂ for every bond
    Right: Color-coded lattice showing agreement/disagreement in top-K
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    pp_scores = np.array(ranking_result['pp_scores'])
    exact_s2s = np.array(ranking_result['exact_s2s'])
    corr = ranking_result['spearman_corr']
    depth = ranking_result['depth']

    # Left: scatter plot
    sc = ax1.scatter(pp_scores, exact_s2s, c=exact_s2s, cmap='YlOrRd',
                     s=80, zorder=3, edgecolors='k', linewidths=0.5)
    plt.colorbar(sc, ax=ax1, label='True S₂ (exact ED)')
    # Add regression line
    z = np.polyfit(pp_scores, exact_s2s, 1)
    x_line = np.linspace(pp_scores.min(), pp_scores.max(), 100)
    ax1.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5, linewidth=1.5)
    ax1.set_xlabel('PP Scout Score (coord-weighted |⟨ZZ⟩|)', fontsize=11)
    ax1.set_ylabel(f'Exact S₂ at Trotter depth={depth}', fontsize=11)
    ax1.set_title(f'PP Score vs True Entanglement\n'
                  f'Spearman ρ = {corr:.3f}', fontsize=12)
    ax1.grid(alpha=0.3)

    # Right: lattice showing top-3 agreement
    lx, ly = config.lx, config.ly
    normalize = lambda b: tuple(sorted(b))
    exact_top3 = {normalize(b) for b in ranking_result['exact_ranking'][:3]}
    pp_top3 = {normalize(b) for b in ranking_result['pp_ranking'][:3]}
    both = exact_top3 & pp_top3
    only_exact = exact_top3 - pp_top3
    only_pp = pp_top3 - exact_top3

    # Draw lattice
    for x in range(lx):
        for y in range(ly):
            ax2.plot(y, x, 'o', color='#555555', markersize=6, zorder=2)
            q = site_index(x, y, ly)
            ax2.text(y, x + 0.15, str(q), ha='center', fontsize=7, color='gray')

    for bond, s2 in ranking_result['bond_exact']:
        q1, q2 = bond
        x1, y1 = site_coords(q1, ly)
        x2, y2 = site_coords(q2, ly)
        norm = normalize(bond)
        if norm in both:
            color, lw, label = '#2E7D32', 4, 'Both agree (top-3)'
        elif norm in only_exact:
            color, lw, label = '#1565C0', 3, 'Exact top-3 only'
        elif norm in only_pp:
            color, lw, label = '#F57F17', 3, 'PP top-3 only'
        else:
            color, lw, label = '#CCCCCC', 1, None
        ax2.plot([y1, y2], [x1, x2], '-', color=color, linewidth=lw,
                 zorder=3 if norm in exact_top3 | pp_top3 else 1)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E7D32', label='Top-3: PP & Exact agree'),
        Patch(facecolor='#1565C0', label='Top-3: Exact only'),
        Patch(facecolor='#F57F17', label='Top-3: PP scout only'),
        Patch(facecolor='#CCCCCC', label='Neither top-3'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax2.set_xlim(-0.5, ly - 0.5)
    ax2.set_ylim(-0.5, lx - 0.5)
    ax2.set_xticks(range(ly))
    ax2.set_yticks(range(lx))
    ax2.set_xticklabels([f'y={i}' for i in range(ly)])
    ax2.set_yticklabels([f'x={i}' for i in range(lx)])
    top1_pct = ranking_result['top_k_overlap'][1]
    top3_pct = ranking_result['top_k_overlap'][3]
    ax2.set_title(f'Top-K Bond Agreement: PP Scout vs Exact ED\n'
                  f'Top-1: {top1_pct:.0f}%  |  Top-3: {top3_pct:.0f}%', fontsize=12)
    ax2.grid(alpha=0.2)

    fig.suptitle(f'PP Scout Ranking Accuracy — {lx}×{ly} TFIM, Depth={depth}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


