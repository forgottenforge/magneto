"""
Fix 13
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
import matplotlib.pyplot as plt
import json

# Load data
with open('quantum_magnetism_v2.3_FINAL_20251116_043644.json', 'r') as f:
    data = json.load(f)

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# E4: Domain Structure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

e4 = data['experiments']['E4_domains']
domain_sizes = np.array(e4['domain_sizes'])
energies = np.array(e4['domain_energies'])

ax1.plot(domain_sizes, energies, 'o-', color='brown', linewidth=2, markersize=8)
ax1.axvline(e4['sigma_c'], color='red', linestyle='--', 
           label=f"σ_c = {e4['sigma_c']:.1f} qubits")
ax1.set_xlabel('Domain Size (qubits)')
ax1.set_ylabel('Staggered Magnetization')
ax1.set_title(f"(a) Domain Wall Energy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Compute susceptibility for E4
chi = np.abs(np.gradient(energies, domain_sizes))
ax2.plot(domain_sizes, chi, 's-', color='purple', linewidth=2, markersize=8)
ax2.axvline(e4['sigma_c'], color='red', linestyle='--')
ax2.axhline(1.5, color='orange', linestyle=':', alpha=0.5, label='κ = 1.5 threshold')
ax2.set_xlabel('Domain Size (qubits)')
ax2.set_ylabel('Susceptibility χ')
ax2.set_title(f"(b) Peak Clarity κ = {e4['kappa']:.2f}")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E4_domains.pdf', dpi=300, bbox_inches='tight')
plt.close()

# E6: GHZ Decoherence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

e6 = data['experiments']['E6_decoherence']
damping = np.array(e6['damping_rates'])
witnesses = np.array(e6['witnesses'])

ax1.plot(damping, witnesses, 'o-', color='indigo', linewidth=2, markersize=8)
ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
ax1.axvline(e6['sigma_c'], color='red', linestyle='--', 
           label=f"γ_c = {e6['sigma_c']:.3f}")
ax1.set_xlabel('Damping Rate γ')
ax1.set_ylabel('Entanglement Witness W')
ax1.set_title('(a) GHZ State Decoherence')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Compute susceptibility
chi = np.abs(np.gradient(witnesses, damping))
ax2.plot(damping, chi, 's-', color='purple', linewidth=2, markersize=8)
ax2.axvline(e6['sigma_c'], color='red', linestyle='--')
ax2.axhline(1.5, color='orange', linestyle=':', alpha=0.5, label='κ = 1.5 threshold')
ax2.set_xlabel('Damping Rate γ')
ax2.set_ylabel('Susceptibility χ')
ax2.set_title(f"(b) Peak Clarity κ = {e6['kappa']:.2f}")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E6_decoherence.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("✓ E4 and E6 plots generated!")