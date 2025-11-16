"""
Fix 1
Copyright (c) 2025 ForgottenForge.xyz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""


import matplotlib.pyplot as plt
import numpy as np
import json

# Lade die Daten
with open('quantum_magnetism_v2.3_FINAL_20251116_043644.json', 'r') as f:
    data = json.load(f)

if 'E5_phase_transition' in data['experiments']:
    exp = data['experiments']['E5_phase_transition']
    if exp['status'] == 'PASSED':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        fields = np.array(exp['fields'])
        correlations = np.array(exp['zz_correlations'])
        
        # Plot 1: Correlations
        ax1.plot(fields, correlations, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.axvline(exp['sigma_c_field'], color='red', linestyle='--', linewidth=2, 
                   label=f"h_c={exp['sigma_c_field']:.3f}")
        ax1.axvline(1.0, color='green', linestyle=':', linewidth=2, 
                   label='Theory h_c=1.0', alpha=0.7)
        ax1.set_xlabel('Transverse Field h', fontsize=12)
        ax1.set_ylabel('⟨ZZ⟩ Correlations', fontsize=12)
        ax1.set_title('E5: Quantum Phase Transition', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Susceptibility
        chi = np.abs(np.gradient(correlations, fields))
        ax2.plot(fields, chi, 'o-', color='purple', linewidth=2, markersize=8)
        ax2.axvline(exp['sigma_c_field'], color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Field h', fontsize=12)
        ax2.set_ylabel('dM/dh', fontsize=12)
        ax2.set_title(f'κ = {exp["kappa"]:.2f}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('E5_phase_transition_FIXED.pdf', dpi=300)
        plt.show()
        print("✓ E5 Plot erstellt!")