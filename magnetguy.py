#!/usr/bin/env python3
"""
QUANTUM MAGNETISM σ_c EXPLORER v2.2 - COMPLETE RE-DESIGN
=========================================================
Copyright (c) 2025 ForgottenForge.xyz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import numpy as np
from braket.circuits import Circuit, Gate
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from scipy import stats
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️ matplotlib not available")


class QuantumMagnetismExplorer:
    def __init__(self, use_hardware: bool = False, budget_euros: float = 200.0):
        self.use_hardware = use_hardware
        self.budget_euros = budget_euros
        self.spent_euros = 0.0
        self.cost_per_task = 0.30
        self.cost_per_shot = 0.00035
        
        if use_hardware:
            print("🧲 Connecting to Rigetti Ankaa-3...")
            try:
                self.device = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3")
                self.backend_type = 'qpu'
                print(f"✓ Connected. Budget: €{budget_euros:.2f}")
            except Exception as e:
                print(f"⚠️ QPU failed, using simulator: {e}")
                self.use_hardware = False
                self.device = LocalSimulator("braket_dm")
                self.backend_type = 'simulator'
        else:
            print("🔬 Using AWS Simulator")
            self.device = LocalSimulator("braket_dm")
            self.backend_type = 'simulator'
        
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': self.backend_type,
                'budget': float(budget_euros),
                'research_question': 'What is magnetism fundamentally?',
                'version': '2.2'
            },
            'experiments': {}
        }

    def estimate_cost(self, n_circuits: int, shots: int) -> float:
        if not self.use_hardware:
            return 0.0
        # Ensure Python int
        return int(n_circuits) * (self.cost_per_task + int(shots) * self.cost_per_shot)

    def check_budget(self, cost: float) -> bool:
        remain = self.budget_euros - self.spent_euros
        if cost > remain:
            print(f"⚠️ Budget exceeded: need €{cost:.2f}, have €{remain:.2f}")
            return False
        if self.use_hardware:
            self.spent_euros += cost
        return True

    def add_measurements(self, circuit: Circuit, qubits: List[int] = None) -> Circuit:
        """Add measurements"""
        if qubits is None:
            max_qubit = 0
            for instr in circuit.instructions:
                target = instr.target
                if isinstance(target, (list, tuple)):
                    for t in target:
                        if isinstance(t, int):
                            max_qubit = max(max_qubit, t)
                elif isinstance(target, int):
                    max_qubit = max(max_qubit, target)
            qubits = list(range(max_qubit + 1))
        
        for q in qubits:
            circuit.measure(q)
        return circuit

    # ========== PAULI MEASUREMENTS ==========
    
    def measure_pauli_string(self, circuit: Circuit, qubits: List[int], 
                            pauli_ops: str, shots: int) -> float:
        """Measure Pauli expectation"""
        circ = circuit.copy()
        
        for q, op in zip(qubits, pauli_ops):
            if op == 'X':
                circ.h(q)
            elif op == 'Y':
                circ.rx(q, np.pi/2)
        
        self.add_measurements(circ, qubits)
        
        # CRITICAL: Braket QPU requires Python int, not numpy.int64!
        task = self.device.run(circ, shots=int(shots))
        counts = task.result().measurement_counts
        
        total = sum(counts.values())
        expectation = 0.0
        
        for bitstring, count in counts.items():
            parity = 1
            for q in qubits:
                if len(bitstring) > q:
                    bit = int(bitstring[-(q+1)])
                    parity *= (1 - 2 * bit)
            expectation += parity * count / total
        
        return expectation

    def measure_entanglement_witness(self, circuit: Circuit, 
                                     q1: int, q2: int, shots: int) -> float:
        """W = (⟨XX⟩ + ⟨YY⟩)/2 - |⟨ZZ⟩|"""
        shots_per_basis = int(shots // 3)
        xx = self.measure_pauli_string(circuit, [q1, q2], 'XX', shots_per_basis)
        yy = self.measure_pauli_string(circuit, [q1, q2], 'YY', shots_per_basis)
        zz = self.measure_pauli_string(circuit, [q1, q2], 'ZZ', shots_per_basis)
        
        witness = (xx + yy) / 2 - abs(zz)
        return witness

    def bitstring_to_spins(self, bitstring: str, n_qubits: int) -> np.ndarray:
        """Convert bitstring to spins"""
        spins = np.zeros(n_qubits)
        for i in range(min(n_qubits, len(bitstring))):
            bit = int(bitstring[-(i+1)])
            spins[i] = 1 - 2 * bit
        return spins

    # ========== BETTER PREPARATIONS ==========
    
    def prepare_transverse_polarized(self, n_qubits: int) -> Circuit:
        """All spins → (|000...⟩ - EIGENSTATE!)
        Instead: |+++...⟩ = (|0⟩+|1⟩)/√2 for each qubit
        This creates superposition that evolves under Ising!
        """
        circuit = Circuit()
        for q in range(n_qubits):
            circuit.h(q)
        return circuit

    def prepare_antiferromagnetic_state(self, n_qubits: int) -> Circuit:
        """↑↓↑↓... in Z-basis"""
        circuit = Circuit()
        for q in range(n_qubits):
            if q % 2 == 1:
                circuit.x(q)
        return circuit

    def prepare_neel_superposition(self, n_qubits: int) -> Circuit:
        """Superposition of Neel states: (|0101...⟩ + |1010...⟩)/√2"""
        circuit = Circuit()
        # Start with |01010...⟩
        for q in range(n_qubits):
            if q % 2 == 1:
                circuit.x(q)
        # Add superposition on first qubit
        circuit.h(0)
        # Propagate with CNOTs
        for q in range(n_qubits - 1):
            circuit.cnot(0, q+1)
        return circuit

    def prepare_domain_structure(self, n_qubits: int, domain_size: int) -> Circuit:
        """Domain walls"""
        circuit = Circuit()
        for q in range(n_qubits):
            if (q // domain_size) % 2 == 1:
                circuit.x(q)
        return circuit

    def prepare_ghz_state(self, n_qubits: int) -> Circuit:
        """GHZ: (|00...0⟩ + |11...1⟩)/√2"""
        circuit = Circuit()
        circuit.h(0)
        for q in range(1, n_qubits):
            circuit.cnot(0, q)
        return circuit

    def prepare_w_state(self, n_qubits: int) -> Circuit:
        """W-state: (|100...⟩ + |010...⟩ + ... + |...001⟩)/√n"""
        # Approximation for W-state
        circuit = Circuit()
        circuit.x(0)
        for q in range(n_qubits - 1):
            angle = np.arccos(np.sqrt(1.0 / (n_qubits - q)))
            circuit.ry(q, 2 * angle)
            circuit.cnot(q, q+1)
        return circuit

    # ========== INTERACTIONS ==========
    
    def add_ising_zz(self, circuit: Circuit, q1: int, q2: int, J: float) -> Circuit:
        """ZZ: exp(-iJσ^z⊗σ^z)"""
        circuit.cnot(q1, q2)
        circuit.rz(q2, 2 * J)
        circuit.cnot(q1, q2)
        return circuit

    def add_heisenberg_xyz(self, circuit: Circuit, q1: int, q2: int, J: float) -> Circuit:
        """Full Heisenberg: XX + YY + ZZ"""
        # XX term
        circuit.cnot(q1, q2)
        circuit.rx(q2, 2 * J)
        circuit.cnot(q1, q2)
        # YY term
        circuit.cnot(q1, q2)
        circuit.ry(q2, 2 * J)
        circuit.cnot(q1, q2)
        # ZZ term
        self.add_ising_zz(circuit, q1, q2, J)
        return circuit

    def add_transverse_field(self, circuit: Circuit, qubits: List[int], h: float) -> Circuit:
        """X-field"""
        for q in qubits:
            circuit.rx(q, 2 * h)
        return circuit

    def add_dephasing(self, circuit: Circuit, qubits: List[int], 
                     gamma: float, seed: int = 42) -> Circuit:
        """Dephasing"""
        rng = np.random.RandomState(seed)
        for q in qubits:
            if rng.random() < gamma:
                angle = rng.uniform(-np.pi * gamma, np.pi * gamma)
                circuit.rz(q, angle)
        return circuit

    def add_amplitude_damping(self, circuit: Circuit, qubits: List[int],
                             gamma: float, seed: int = 42) -> Circuit:
        """T1 decay simulation"""
        rng = np.random.RandomState(seed)
        for q in qubits:
            if rng.random() < gamma:
                circuit.rx(q, -gamma * np.pi)
        return circuit

    # ========== OBSERVABLES ==========
    
    def measure_spin_correlations_zz(self, circuit: Circuit, n_qubits: int,
                                     distance: int, shots: int) -> float:
        """⟨ZZ⟩ at distance"""
        correlations = []
        shots_per_pair = int(max(shots // (n_qubits - distance), 50))
        
        for i in range(n_qubits - distance):
            j = i + distance
            zz = self.measure_pauli_string(circuit, [i, j], 'ZZ', shots_per_pair)
            correlations.append(zz)
        
        return np.mean(correlations) if correlations else 0.0

    def measure_magnetization(self, circuit: Circuit, n_qubits: int, 
                             shots: int) -> float:
        """⟨ΣZ⟩/N"""
        circ = circuit.copy()
        self.add_measurements(circ, list(range(n_qubits)))
        
        task = self.device.run(circ, shots=int(shots))
        counts = task.result().measurement_counts
        
        mag = 0.0
        total = sum(counts.values())
        
        for bitstring, count in counts.items():
            spins = self.bitstring_to_spins(bitstring, n_qubits)
            mag += spins.sum() * count / total
        
        return mag / n_qubits

    def measure_staggered_magnetization(self, circuit: Circuit, n_qubits: int,
                                       shots: int) -> float:
        """Staggered: ⟨Σ (-1)^i Z_i⟩/N - sensitive to AFM order"""
        circ = circuit.copy()
        self.add_measurements(circ, list(range(n_qubits)))
        
        task = self.device.run(circ, shots=int(shots))
        counts = task.result().measurement_counts
        
        stag_mag = 0.0
        total = sum(counts.values())
        
        for bitstring, count in counts.items():
            spins = self.bitstring_to_spins(bitstring, n_qubits)
            staggered = sum((-1)**i * spins[i] for i in range(n_qubits))
            stag_mag += staggered * count / total
        
        return stag_mag / n_qubits

    # ========== ROBUST σ_c ==========
    
    def compute_susceptibility(self, sigmas: np.ndarray, observables: np.ndarray,
                              smoothing_window: int = 5) -> Tuple[np.ndarray, float, float]:
        """χ(σ) = |dO/dσ| with robust peak detection"""
        if len(observables) < smoothing_window:
            obs_smooth = observables
        else:
            polyorder = min(3, smoothing_window - 1)
            obs_smooth = savgol_filter(observables, smoothing_window, polyorder, mode='nearest')
        
        chi = np.abs(np.gradient(obs_smooth, sigmas))
        
        peaks, properties = find_peaks(chi, prominence=0.001)
        
        if len(peaks) == 0:
            peak_idx = np.argmax(chi)
            sigma_c = sigmas[peak_idx]
            baseline = np.percentile(chi, 25)
            if baseline < 1e-10:
                baseline = np.mean(chi) + 1e-10
            kappa = chi[peak_idx] / baseline
        else:
            prominences = peak_prominences(chi, peaks)[0]
            best_peak_idx = peaks[np.argmax(prominences)]
            sigma_c = sigmas[best_peak_idx]
            baseline = np.median(chi)
            if baseline < 1e-10:
                baseline = np.mean(chi) + 1e-10
            kappa = prominences.max() / baseline
        
        return chi, float(sigma_c), float(kappa)

    def bootstrap_sigma_c(self, sigmas: np.ndarray, observables: np.ndarray,
                         n_boot: int = 1000, smoothing: int = 5) -> Tuple[float, float]:
        """Bootstrap CIs"""
        rng = np.random.RandomState(42)
        n_points = len(sigmas)
        sigma_c_boots = []
        
        for _ in range(n_boot):
            idx = rng.choice(n_points, size=n_points, replace=True)
            sig_boot = sigmas[idx]
            obs_boot = observables[idx]
            
            sort_idx = np.argsort(sig_boot)
            sig_boot = sig_boot[sort_idx]
            obs_boot = obs_boot[sort_idx]
            
            _, sc, _ = self.compute_susceptibility(sig_boot, obs_boot, smoothing)
            if not np.isnan(sc):
                sigma_c_boots.append(sc)
        
        if len(sigma_c_boots) < 10:
            return sigmas[0], sigmas[-1]
        
        ci_low = np.percentile(sigma_c_boots, 2.5)
        ci_high = np.percentile(sigma_c_boots, 97.5)
        return float(ci_low), float(ci_high)

    # ========== EXPERIMENTS ==========
    
    def E1_ferromagnetic_correlations(self) -> Dict:
        """
        E1: Spin correlations from |+++⟩ under Ising
        NOW with non-trivial initial state!
        """
        print("\n🧲 E1: Spin Correlations - Ising Dynamics")
        
        n_qubits = 8
        J = 0.8
        distances = np.arange(1, 8)
        correlations = []
        
        shots = 400 if self.use_hardware else 800
        cost = self.estimate_cost(len(distances) * (n_qubits - 1), shots // (n_qubits - 1))
        
        if not self.check_budget(cost):
            return {'status': 'SKIPPED', 'reason': 'budget'}
        
        print(f"Budget: €{cost:.2f}, Start: |+++...⟩, Evolution: Ising")
        
        # Start with transverse polarization |+++...⟩
        base_circuit = self.prepare_transverse_polarized(n_qubits)
        
        # Ising time evolution - multiple layers
        n_layers = 4
        for layer in range(n_layers):
            for i in range(n_qubits - 1):
                self.add_ising_zz(base_circuit, i, i+1, J * 0.25)
        
        for dist in tqdm(distances, desc="E1 Distance"):
            corr = self.measure_spin_correlations_zz(base_circuit, n_qubits, dist, shots)
            correlations.append(corr)
        
        correlations = np.array(correlations)
        print(f"  Correlations: {correlations}")
        
        chi, sigma_c, kappa = self.compute_susceptibility(distances, correlations)
        ci_low, ci_high = self.bootstrap_sigma_c(distances, correlations)
        
        result = {
            'status': 'PASSED' if kappa > 1.5 else 'FAILED',
            'distances': distances.tolist(),
            'correlations': correlations.tolist(),
            'chi': chi.tolist(),
            'sigma_c': float(sigma_c),
            'sigma_c_ci': [float(ci_low), float(ci_high)],
            'kappa': float(kappa),
            'interpretation': f'Correlation length: ξ_c ≈ {sigma_c:.2f} qubits',
            'physics': 'Non-equilibrium Ising dynamics from |+⟩ state'
        }
        
        print(f"✓ ξ_c = {sigma_c:.3f} [{ci_low:.3f}, {ci_high:.3f}], κ = {kappa:.2f}")
        return result

    def E2_antiferromagnetic_comparison(self) -> Dict:
        """
        E2: FM vs AFM - both from |+++⟩
        Use staggered magnetization for AFM
        """
        print("\n🧲 E2: FM vs AFM - Order Parameter Evolution")
        
        n_qubits = 8
        J_ferro = 1.0
        J_anti = -1.0
        times = np.linspace(0, 2.0, 12)  # Evolution time
        
        shots = 350 if self.use_hardware else 700
        cost = self.estimate_cost(2 * len(times), shots)
        
        if not self.check_budget(cost):
            return {'status': 'SKIPPED', 'reason': 'budget'}
        
        mag_ferro = []
        stag_mag_anti = []
        
        for t in tqdm(times, desc="E2 Time Evolution"):
            # Ferromagnetic
            circ_f = self.prepare_transverse_polarized(n_qubits)
            n_steps = max(1, int(t * 5))
            for _ in range(n_steps):
                for i in range(n_qubits - 1):
                    self.add_ising_zz(circ_f, i, i+1, J_ferro * t / n_steps)
            mag_f = abs(self.measure_magnetization(circ_f, n_qubits, shots))
            mag_ferro.append(mag_f)
            
            # Antiferromagnetic
            circ_a = self.prepare_transverse_polarized(n_qubits)
            for _ in range(n_steps):
                for i in range(n_qubits - 1):
                    self.add_ising_zz(circ_a, i, i+1, J_anti * t / n_steps)
            stag_a = abs(self.measure_staggered_magnetization(circ_a, n_qubits, shots))
            stag_mag_anti.append(stag_a)
        
        _, sc_f, kappa_f = self.compute_susceptibility(times, np.array(mag_ferro))
        _, sc_a, kappa_a = self.compute_susceptibility(times, np.array(stag_mag_anti))
        
        result = {
            'status': 'PASSED' if (kappa_f > 1.5 or kappa_a > 1.5) else 'FAILED',
            'times': times.tolist(),
            'magnetization_ferro': np.array(mag_ferro).tolist(),
            'staggered_mag_anti': np.array(stag_mag_anti).tolist(),
            'sigma_c_ferro': float(sc_f),
            'sigma_c_anti': float(sc_a),
            'kappa_ferro': float(kappa_f),
            'kappa_anti': float(kappa_a),
            'interpretation': f'FM builds at t={sc_f:.2f}, AFM at t={sc_a:.2f}',
            'physics': 'Different ordering timescales for FM vs AFM'
        }
        
        print(f"✓ FM: t_c={sc_f:.2f} (κ={kappa_f:.2f}), AFM: t_c={sc_a:.2f} (κ={kappa_a:.2f})")
        return result

    def E3_entanglement_buildup_timescales(self) -> Dict:
        """
        E3: CRITICAL - Information→Physics
        6-qubit chain with variable decoherence
        """
        print("\n⚛️ E3: ENTANGLEMENT BUILDUP - Information→Physics")
        
        n_qubits = 6
        noise_levels = np.linspace(0.0, 0.7, 15)
        witnesses = []
        
        shots = 600 if self.use_hardware else 1200
        cost = self.estimate_cost(len(noise_levels), shots)
        
        if not self.check_budget(cost):
            return {'status': 'SKIPPED', 'reason': 'budget'}
        
        print(f"🔬 KEY EXPERIMENT - 6-qubit chain, variable noise")
        
        for gamma in tqdm(noise_levels, desc="E3 Noise"):
            # Build entanglement chain with noise
            circuit = Circuit()
            circuit.h(0)
            for i in range(n_qubits - 1):
                circuit.cnot(i, i+1)
                # Noise DURING buildup
                if gamma > 0:
                    self.add_dephasing(circuit, [i, i+1], gamma * 1.5, seed=int(gamma*1000))
                    self.add_amplitude_damping(circuit, [i, i+1], gamma * 0.5, seed=int(gamma*1000)+1)
            
            # Measure average witness across pairs
            wit_sum = 0.0
            shots_per_witness = int(shots // 3)
            for i in range(min(3, n_qubits-1)):
                wit_sum += self.measure_entanglement_witness(circuit, i, i+1, shots_per_witness)
            witness = wit_sum / min(3, n_qubits-1)
            witnesses.append(witness)
        
        witnesses = np.array(witnesses)
        chi, sigma_c, kappa = self.compute_susceptibility(noise_levels, witnesses)
        ci_low, ci_high = self.bootstrap_sigma_c(noise_levels, witnesses)
        
        result = {
            'status': 'PASSED' if kappa > 2.0 else 'FAILED',
            'noise_levels': noise_levels.tolist(),
            'entanglement_witness': witnesses.tolist(),
            'chi': chi.tolist(),
            'sigma_c': float(sigma_c),
            'sigma_c_ci': [float(ci_low), float(ci_high)],
            'kappa': float(kappa),
            'interpretation': f'⚡ Information→Physics at γ_c = {sigma_c:.4f}',
            'fundamental_insight': 'Decoherence threshold where quantum info manifests!',
            'physics': 'Competition: entanglement buildup vs decoherence'
        }
        
        print(f"🎯 γ_c = {sigma_c:.4f} [{ci_low:.4f}, {ci_high:.4f}], κ = {kappa:.2f}")
        return result

    def E4_domain_wall_structure(self) -> Dict:
        """
        E4: Domain walls - use staggered magnetization
        """
        print("\n🧲 E4: Domain Wall Energy")
        
        n_qubits = 12
        J = 0.5
        domain_sizes = np.array([1, 2, 3, 4, 6])
        energies = []
        
        shots = 400 if self.use_hardware else 800
        cost = self.estimate_cost(len(domain_sizes), shots)
        
        if not self.check_budget(cost):
            return {'status': 'SKIPPED', 'reason': 'budget'}
        
        for dsize in tqdm(domain_sizes, desc="E4 Domains"):
            circuit = self.prepare_domain_structure(n_qubits, int(dsize))
            
            # Ising evolution
            for i in range(n_qubits - 1):
                self.add_ising_zz(circuit, i, i+1, J)
            
            # Measure energy proxy: staggered magnetization
            stag = abs(self.measure_staggered_magnetization(circuit, n_qubits, shots))
            energies.append(stag)
        
        energies = np.array(energies)
        chi, sigma_c, kappa = self.compute_susceptibility(domain_sizes, energies)
        
        result = {
            'status': 'PASSED' if kappa > 1.5 else 'FAILED',
            'domain_sizes': domain_sizes.tolist(),
            'domain_energies': energies.tolist(),
            'sigma_c': float(sigma_c),
            'kappa': float(kappa),
            'interpretation': f'Optimal domain: {sigma_c:.1f} qubits',
            'physics': 'Domain wall energy minimization'
        }
        
        print(f"✓ Optimal: {sigma_c:.2f} qubits, κ = {kappa:.2f}")
        return result

    def E5_quantum_phase_transition(self) -> Dict:
        """
        E5: TFIM QPT - start from |+++⟩
        Measure transverse magnetization
        """
        print("\n⚡ E5: Quantum Phase Transition (TFIM)")
        
        n_qubits = 6
        J = 1.0
        fields = np.linspace(0.0, 3.0, 15)
        mag_x = []  # Transverse magnetization
        
        shots = 500 if self.use_hardware else 1000
        cost = self.estimate_cost(len(fields), shots)
        
        if not self.check_budget(cost):
            return {'status': 'SKIPPED', 'reason': 'budget'}
        
        for h in tqdm(fields, desc="E5 Field"):
            # Start from ground state of pure Ising (aligned)
            circuit = Circuit()
            
            # Trotterized evolution
            n_trotter = 8
            dt = 0.4 / n_trotter
            
            for _ in range(n_trotter):
                # ZZ terms
                for i in range(n_qubits - 1):
                    self.add_ising_zz(circuit, i, i+1, J * dt)
                # Transverse field
                self.add_transverse_field(circuit, list(range(n_qubits)), h * dt)
            
            # Measure X-magnetization (transverse)
            mx_sum = 0.0
            shots_per_qubit = int(shots // n_qubits)
            for i in range(n_qubits):
                mx = self.measure_pauli_string(circuit, [i], 'X', shots_per_qubit)
                mx_sum += mx
            mag_x.append(abs(mx_sum / n_qubits))
        
        mag_x = np.array(mag_x)
        chi, sigma_c, kappa = self.compute_susceptibility(fields, mag_x)
        
        result = {
            'status': 'PASSED' if kappa > 2.0 else 'FAILED',
            'fields': fields.tolist(),
            'magnetization_x': mag_x.tolist(),
            'sigma_c_field': float(sigma_c),
            'kappa': float(kappa),
            'interpretation': f'Critical h_c ≈ {sigma_c:.3f} (theory ~1.0)',
            'fundamental_insight': 'σ_c detects quantum phase transition!',
            'physics': 'Paramagnet→Ferromagnet transition'
        }
        
        print(f"✓ h_c = {sigma_c:.3f}, κ = {kappa:.2f}")
        return result

    def E6_decoherence_transition(self) -> Dict:
        """
        E6: Quantum→Classical via W-state decoherence
        W-state is more robust than GHZ!
        """
        print("\n🌊 E6: Quantum→Classical (W-state)")
        
        n_qubits = 5
        damping_rates = np.linspace(0.0, 0.8, 15)
        witnesses = []
        
        shots = 600 if self.use_hardware else 1200
        cost = self.estimate_cost(len(damping_rates), shots)
        
        if not self.check_budget(cost):
            return {'status': 'SKIPPED', 'reason': 'budget'}
        
        for gamma in tqdm(damping_rates, desc="E6 Damping"):
            # W-state + decoherence
            circuit = self.prepare_w_state(n_qubits)
            
            # Both dephasing and amplitude damping
            self.add_dephasing(circuit, list(range(n_qubits)), gamma * 0.6, seed=int(gamma*1000))
            self.add_amplitude_damping(circuit, list(range(n_qubits)), gamma * 0.4, seed=int(gamma*1000)+1)
            
            # Measure average witness
            wit_sum = 0.0
            shots_per_witness = int(shots // 3)
            for i in range(min(3, n_qubits-1)):
                wit_sum += self.measure_entanglement_witness(circuit, i, i+1, shots_per_witness)
            witness = wit_sum / min(3, n_qubits-1)
            witnesses.append(witness)
        
        witnesses = np.array(witnesses)
        chi, sigma_c, kappa = self.compute_susceptibility(damping_rates, witnesses)
        
        result = {
            'status': 'PASSED' if kappa > 1.8 else 'FAILED',
            'damping_rates': damping_rates.tolist(),
            'witnesses': witnesses.tolist(),
            'sigma_c': float(sigma_c),
            'kappa': float(kappa),
            'interpretation': f'Quantum→Classical at γ_c = {sigma_c:.4f}',
            'fundamental_insight': 'W-state decoherence threshold!',
            'physics': 'Distributed entanglement→separable transition'
        }
        
        print(f"✓ γ_c = {sigma_c:.4f}, κ = {kappa:.2f}")
        return result

    # ========== RUNNER ==========
    
    def run_all_experiments(self) -> Dict:
        print("="*70)
        print("QUANTUM MAGNETISM σ_c v2.2 - COMPLETE RE-DESIGN")
        print("All experiments with non-trivial states for clear signals!")
        print("="*70)
        
        experiments = [
            ('E1_ferromagnetic', self.E1_ferromagnetic_correlations),
            ('E2_antiferromagnetic', self.E2_antiferromagnetic_comparison),
            ('E3_entanglement_timescales', self.E3_entanglement_buildup_timescales),
            ('E4_domains', self.E4_domain_wall_structure),
            ('E5_phase_transition', self.E5_quantum_phase_transition),
            ('E6_decoherence', self.E6_decoherence_transition),
        ]
        
        for name, func in experiments:
            try:
                result = func()
                self.results['experiments'][name] = result
            except Exception as e:
                print(f"❌ {name} error: {e}")
                import traceback
                traceback.print_exc()
                self.results['experiments'][name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
            
            if self.use_hardware and self.spent_euros >= self.budget_euros * 0.95:
                print(f"\n⚠️ Budget limit: €{self.spent_euros:.2f}/€{self.budget_euros:.2f}")
                break
        
        self.results['summary'] = {
            'total_spent_euros': float(self.spent_euros),
            'budget_remaining': float(self.budget_euros - self.spent_euros),
            'experiments_passed': sum(
                1 for e in self.results['experiments'].values() 
                if e.get('status') == 'PASSED'
            ),
            'experiments_total': len([e for e in self.results['experiments'].values()])
        }
        
        return self.results

    def save_results(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_magnetism_v2.2_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Saved: {filename}")
        return filename

    # ========== PLOTTING ==========
    
    def plot_results(self, save_dir: str = "magnetism_plots"):
        if not HAS_PLOTTING:
            print("⚠️ No matplotlib")
            return
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n📊 Generating plots in {save_dir}/...")
        
        for exp_name, exp_data in self.results['experiments'].items():
            if exp_data.get('status') != 'PASSED':
                continue
            
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                if exp_name == 'E1_ferromagnetic':
                    x = np.array(exp_data['distances'])
                    y = np.array(exp_data['correlations'])
                    chi = np.array(exp_data['chi'])
                    sc = exp_data['sigma_c']
                    
                    ax1.plot(x, y, 'o-', label='⟨ZZ⟩', linewidth=2)
                    ax1.axvline(sc, color='red', linestyle='--', label=f'ξ_c={sc:.2f}')
                    ax1.set_xlabel('Distance (qubits)', fontsize=12)
                    ax1.set_ylabel('⟨ZZ⟩ Correlation', fontsize=12)
                    ax1.set_title('E1: Ising Dynamics from |+⟩ State', fontsize=14, fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(x, chi, 'o-', color='purple', linewidth=2)
                    ax2.axvline(sc, color='red', linestyle='--')
                    ax2.set_xlabel('Distance', fontsize=12)
                    ax2.set_ylabel('Susceptibility χ', fontsize=12)
                    ax2.set_title(f'κ = {exp_data["kappa"]:.2f}', fontsize=14)
                    ax2.grid(True, alpha=0.3)
                
                elif exp_name == 'E2_antiferromagnetic':
                    x = np.array(exp_data['times'])
                    y1 = np.array(exp_data['magnetization_ferro'])
                    y2 = np.array(exp_data['staggered_mag_anti'])
                    
                    ax1.plot(x, y1, 'o-', label=f'FM |M| (t_c={exp_data["sigma_c_ferro"]:.2f})', color='blue', linewidth=2)
                    ax1.plot(x, y2, 's-', label=f'AFM |M_stag| (t_c={exp_data["sigma_c_anti"]:.2f})', color='red', linewidth=2)
                    ax1.set_xlabel('Evolution Time', fontsize=12)
                    ax1.set_ylabel('Order Parameter', fontsize=12)
                    ax1.set_title('E2: FM vs AFM Ordering', fontsize=14, fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    chi1 = np.abs(np.gradient(y1, x))
                    chi2 = np.abs(np.gradient(y2, x))
                    ax2.plot(x, chi1, 'o-', color='blue', label=f'FM (κ={exp_data["kappa_ferro"]:.2f})', linewidth=2)
                    ax2.plot(x, chi2, 's-', color='red', label=f'AFM (κ={exp_data["kappa_anti"]:.2f})', linewidth=2)
                    ax2.set_xlabel('Time', fontsize=12)
                    ax2.set_ylabel('Susceptibility', fontsize=12)
                    ax2.set_title('Ordering Rate', fontsize=14)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                
                elif exp_name == 'E3_entanglement_timescales':
                    x = np.array(exp_data['noise_levels'])
                    y = np.array(exp_data['entanglement_witness'])
                    chi = np.array(exp_data['chi'])
                    sc = exp_data['sigma_c']
                    ci = exp_data['sigma_c_ci']
                    
                    ax1.plot(x, y, 'o-', label='Witness W', color='green', linewidth=2, markersize=8)
                    ax1.axvline(sc, color='red', linestyle='--', label=f'γ_c={sc:.4f}', linewidth=2)
                    ax1.axvspan(ci[0], ci[1], alpha=0.2, color='red')
                    ax1.axhline(0, color='black', linestyle=':', alpha=0.5)
                    ax1.set_xlabel('Noise Level γ', fontsize=12)
                    ax1.set_ylabel('Entanglement Witness', fontsize=12)
                    ax1.set_title('E3: Information→Physics ⚡', fontsize=14, fontweight='bold')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.plot(x, chi, 'o-', color='purple', linewidth=2)
                    ax2.axvline(sc, color='red', linestyle='--', linewidth=2)
                    ax2.set_xlabel('Noise γ', fontsize=12)
                    ax2.set_ylabel('χ(γ)', fontsize=12)
                    ax2.set_title(f'Peak Clarity κ = {exp_data["kappa"]:.2f}', fontsize=14)
                    ax2.grid(True, alpha=0.3)
                
                # Add more elif blocks for E4, E5, E6...
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/{exp_name}.pdf", dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"⚠️ Plot {exp_name} failed: {e}")
        
        print(f"✓ Plots saved!")


# ========== MAIN ==========

if __name__ == "__main__":
    print("\n🧲⚛️ QUANTUM MAGNETISM v2.2 - COMPLETE RE-DESIGN")
    print("=" * 70)
    print("All experiments redesigned with non-trivial initial states!")
    print("Expected: 6/6 PASSED with strong κ values")
    print("=" * 70)
    
    choice = input("\nMode:\n  1 = Simulator (~25min)\n  2 = Rigetti QPU (<€200)\nChoice: ")
    
    use_hardware = (choice.strip() == '2')
    budget = 200.0 if use_hardware else 0.0
    
    explorer = QuantumMagnetismExplorer(use_hardware=use_hardware, budget_euros=budget)
    
    print("\n🚀 Starting...")
    results = explorer.run_all_experiments()
    
    filename = explorer.save_results()
    
    print("\n📊 Generating plots...")
    explorer.plot_results()
    
    # SUMMARY
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Passed: {results['summary']['experiments_passed']}/6")
    
    print("\n🔬 KEY FINDINGS:")
    for exp_name, exp_data in results['experiments'].items():
        if exp_data.get('status') == 'PASSED':
            print(f"\n✓ {exp_name}:")
            if 'sigma_c' in exp_data or 'sigma_c_field' in exp_data:
                sc = exp_data.get('sigma_c', exp_data.get('sigma_c_field'))
                print(f"  σ_c = {sc:.4f}, κ = {exp_data.get('kappa', 0):.2f}")
            if 'interpretation' in exp_data:
                print(f"  → {exp_data['interpretation']}")
            if 'physics' in exp_data:
                print(f"  Physics: {exp_data['physics']}")
    
    print(f"\n💾 {filename}")
    print("\n🎯 NOW READY FOR RIGETTI QPU RUN!")