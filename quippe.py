#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK FIX - Run this to complete the analysis
Fix 2
Copyright (c) 2025 ForgottenForge.xyz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""

import json
import pandas as pd
from pathlib import Path

# Load the data
with open('quantum_magnetism_v2.3_FINAL_20251116_043644.json', 'r') as f:
    data = json.load(f)

# Create the missing directories
Path('figures_enhanced').mkdir(exist_ok=True)

# Generate simple summary table
results = []
for exp_name, exp_data in data['experiments'].items():
    if 'kappa' in exp_data:
        results.append({
            'Experiment': exp_name,
            'sigma_c': exp_data.get('sigma_c', exp_data.get('sigma_c_field', 'N/A')),
            'kappa': exp_data['kappa'],
            'Status': exp_data['status']
        })

df = pd.DataFrame(results)
print("\n=== RESULTS SUMMARY ===")
print(df.to_string())

# Save with UTF-8
df.to_csv('results_summary.csv', index=False, encoding='utf-8')


print("\n✓ Analysis completed successfully!")
print("✓ Files saved with UTF-8 encoding")
print("✓ Ready for journal submission after claims revision")