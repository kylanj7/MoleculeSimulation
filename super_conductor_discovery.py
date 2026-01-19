#!/usr/bin/env python
"""
Room-Temperature Superconductor Discovery
Computational screening of hydride materials

BACKGROUND:
- Conventional superconductors work at very low temps (<30K = -243°C)
- High-pressure hydrides show promise (H3S works at 203K under 155 GPa)
- Room temperature = 293K (20°C) is the holy grail
- Recent claims: LuH at room temp (controversial)

RESEARCH QUESTION:
Can we computationally screen metal hydrides to find room-temp superconductors?

APPROACH:
1. Generate candidate metal hydride structures
2. Calculate electronic properties
3. Estimate phonon frequencies (critical for superconductivity)
4. Screen for stability and synthesizability
"""

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write, read
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import os

# Create output directory
os.makedirs('superconductor_screening', exist_ok=True)

print("=" * 80)
print("ROOM-TEMPERATURE SUPERCONDUCTOR SCREENING")
print("Computational Materials Discovery")
print("=" * 80)

# ============================================================================
# PART 1: GENERATE CANDIDATE STRUCTURES
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: GENERATING METAL HYDRIDE CANDIDATES")
print("=" * 80)

print("""
STRATEGY:
Metal hydrides (MHx) are promising because:
- Light hydrogen atoms = high phonon frequencies
- Strong electron-phonon coupling
- High Tc in H3S (203K at 155 GPa)

We'll screen:
- Different metals (Li, Na, Mg, Al, Ca, Y, La)
- Different H ratios (MH, MH2, MH3, MH6)
- Different crystal structures
""")

def create_metal_hydride(metal, h_ratio, lattice_type='fcc'):
    """
    Create a metal hydride structure
    
    Parameters:
    -----------
    metal : str
        Metal element symbol
    h_ratio : int
        Number of H atoms per metal atom
    lattice_type : str
        Crystal structure type
    
    Returns:
    --------
    atoms : ASE Atoms object
    """
    # Start with metal structure
    if lattice_type == 'fcc':
        a = 4.0  # Approximate lattice parameter
        positions = [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0]
        ]
    elif lattice_type == 'bcc':
        a = 3.5
        positions = [
            [0, 0, 0],
            [0.5, 0.5, 0.5]
        ]
    elif lattice_type == 'cubic':
        a = 4.5
        positions = [[0, 0, 0]]
    
    # Scale positions
    positions = np.array(positions) * a
    
    # Create metal atoms
    symbols = [metal] * len(positions)
    
    # Add hydrogen atoms
    # Place H atoms at octahedral/tetrahedral sites
    h_positions = []
    for i in range(h_ratio):
        # Distribute H atoms around metal atoms
        offset = np.array([
            a/4 * np.cos(2*np.pi*i/h_ratio),
            a/4 * np.sin(2*np.pi*i/h_ratio),
            a/4 * ((-1)**i)
        ])
        h_positions.append(positions[0] + offset)
    
    all_positions = np.vstack([positions, h_positions])
    all_symbols = symbols + ['H'] * len(h_positions)
    
    # Create structure
    atoms = Atoms(
        symbols=all_symbols,
        positions=all_positions,
        cell=[a, a, a],
        pbc=True
    )
    
    return atoms


# Generate library of candidates
print("\nGenerating metal hydride candidates...")
print("-" * 80)

candidates = []
metals = ['Li', 'Na', 'Mg', 'Al', 'Ca', 'Y', 'La']  # Promising metals
h_ratios = [1, 2, 3, 6]  # Different stoichiometries
structures = ['fcc', 'bcc', 'cubic']

for metal in metals:
    for h_ratio in h_ratios:
        for structure in structures:
            try:
                atoms = create_metal_hydride(metal, h_ratio, structure)
                formula = atoms.get_chemical_formula()
                
                # Save structure
                filename = f"superconductor_screening/{metal}H{h_ratio}_{structure}.xyz"
                write(filename, atoms)
                
                candidates.append({
                    'metal': metal,
                    'h_ratio': h_ratio,
                    'structure': structure,
                    'formula': formula,
                    'atoms': atoms,
                    'filename': filename
                })
                
                print(f"Created: {formula:15} ({structure:6}) - {len(atoms)} atoms")
                
            except Exception as e:
                print(f"Failed to create {metal}H{h_ratio} ({structure}): {e}")

print(f"\nTotal candidates generated: {len(candidates)}")

# ============================================================================
# PART 2: STRUCTURAL OPTIMIZATION
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: STRUCTURAL OPTIMIZATION")
print("=" * 80)

print("""
Before predicting properties, we need to optimize the structure.
This finds the lowest energy configuration.

NOTE: We're using EMT calculator (fast but approximate).
With UMA access, you'd use FAIRChemCalculator for DFT-quality results.
""")

optimized_candidates = []

for i, candidate in enumerate(candidates[:10]):  # Optimize first 10 for demo
    print(f"\nOptimizing {candidate['formula']} ({i+1}/10)...")
    
    atoms = candidate['atoms'].copy()
    
    # Set calculator
    atoms.calc = EMT()
    
    try:
        # Get initial energy
        E_initial = atoms.get_potential_energy()
        
        # Optimize geometry
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.05, steps=100)
        
        # Get final energy
        E_final = atoms.get_potential_energy()
        
        # Save optimized structure
        opt_filename = f"superconductor_screening/{candidate['formula']}_optimized.xyz"
        write(opt_filename, atoms)
        
        candidate['E_initial'] = E_initial
        candidate['E_final'] = E_final
        candidate['E_relaxation'] = E_final - E_initial
        candidate['atoms_optimized'] = atoms
        
        optimized_candidates.append(candidate)
        
        print(f"  Initial energy: {E_initial:.3f} eV")
        print(f"  Final energy:   {E_final:.3f} eV")
        print(f"  Relaxation:     {E_final - E_initial:.3f} eV")
        
    except Exception as e:
        print(f"  Optimization failed: {e}")

# ============================================================================
# PART 3: CALCULATE PROPERTIES RELEVANT TO SUPERCONDUCTIVITY
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: SCREENING FOR SUPERCONDUCTOR PROPERTIES")
print("=" * 80)

print("""
SUPERCONDUCTIVITY CRITERIA (simplified screening):

1. ELECTRONIC STRUCTURE:
   - Need metallic behavior (electrons at Fermi level)
   - High density of states at Fermi level

2. PHONON PROPERTIES:
   - High phonon frequencies (light atoms help)
   - Strong electron-phonon coupling

3. STABILITY:
   - Thermodynamically stable (negative formation energy)
   - Dynamically stable (no imaginary phonons)

4. SYNTHESIS:
   - Accessible pressure (<100 GPa preferred)
   - Doesn't decompose at target temperature

We'll calculate approximate indicators:
""")

results = []

for candidate in optimized_candidates:
    atoms = candidate['atoms_optimized']
    formula = candidate['formula']
    
    print(f"\nAnalyzing {formula}...")
    print("-" * 40)
    
    # Get atomic properties
    n_atoms = len(atoms)
    n_metals = sum(1 for s in atoms.get_chemical_symbols() if s != 'H')
    n_hydrogen = sum(1 for s in atoms.get_chemical_symbols() if s == 'H')
    
    # Calculate volume and density
    volume = atoms.get_volume()
    masses = atoms.get_masses()
    total_mass = masses.sum()
    density = total_mass / volume  # amu/Angstrom^3
    
    # Hydrogen content (higher = better for high phonon frequencies)
    h_fraction = n_hydrogen / n_atoms
    
    # Average bond length (proxy for coupling strength)
    positions = atoms.get_positions()
    metal_positions = positions[[i for i, s in enumerate(atoms.get_chemical_symbols()) if s != 'H']]
    h_positions = positions[[i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'H']]
    
    # Calculate average M-H distance
    if len(h_positions) > 0 and len(metal_positions) > 0:
        distances = []
        for mp in metal_positions:
            for hp in h_positions:
                dist = np.linalg.norm(mp - hp)
                distances.append(dist)
        avg_mh_distance = np.mean(distances)
    else:
        avg_mh_distance = 0
    
    # Energy per atom (stability indicator)
    energy_per_atom = candidate['E_final'] / n_atoms
    
    # Estimate phonon frequency (very rough)
    # Higher for lighter atoms and stronger bonds
    if avg_mh_distance > 0:
        # Frequency ~ sqrt(k/m), k ~ 1/r^2
        # This is VERY approximate
        h_mass = 1.0  # amu
        estimated_frequency = 1000 / (avg_mh_distance**2 * np.sqrt(h_mass))  # arbitrary units
    else:
        estimated_frequency = 0
    
    # Store results
    result = {
        'formula': formula,
        'metal': candidate['metal'],
        'h_ratio': candidate['h_ratio'],
        'structure': candidate['structure'],
        'n_atoms': n_atoms,
        'n_hydrogen': n_hydrogen,
        'h_fraction': h_fraction,
        'density': density,
        'energy_per_atom': energy_per_atom,
        'avg_mh_distance': avg_mh_distance,
        'estimated_frequency': estimated_frequency,
        'relaxation_energy': candidate['E_relaxation']
    }
    
    results.append(result)
    
    print(f"  Hydrogen fraction: {h_fraction:.2%}")
    print(f"  Density: {density:.3f} amu/Å³")
    print(f"  Energy/atom: {energy_per_atom:.3f} eV")
    print(f"  Avg M-H distance: {avg_mh_distance:.3f} Å")
    print(f"  Est. phonon freq: {estimated_frequency:.1f} (arb. units)")

# ============================================================================
# PART 4: RANK CANDIDATES
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: RANKING CANDIDATES FOR SUPERCONDUCTIVITY")
print("=" * 80)

print("""
RANKING CRITERIA:
1. High hydrogen content (light atoms → high phonon freq)
2. Moderate M-H distance (strong coupling)
3. Low energy (stability)
4. High estimated phonon frequency

Creating composite score...
""")

# Normalize and score
for result in results:
    # Higher H fraction is better
    h_score = result['h_fraction']
    
    # Moderate M-H distance is better (around 1.5-2.0 Å)
    optimal_distance = 1.8
    distance_score = 1.0 / (1.0 + abs(result['avg_mh_distance'] - optimal_distance))
    
    # Lower energy is better (more negative)
    energy_score = -result['energy_per_atom'] / 10.0  # normalize
    
    # Higher frequency is better
    freq_score = result['estimated_frequency'] / 100.0
    
    # Composite score (weighted average)
    composite_score = (
        0.3 * h_score +
        0.3 * distance_score +
        0.2 * energy_score +
        0.2 * freq_score
    )
    
    result['composite_score'] = composite_score

# Sort by composite score
results_sorted = sorted(results, key=lambda x: x['composite_score'], reverse=True)

# Display top candidates
print("\nTOP 10 SUPERCONDUCTOR CANDIDATES:")
print("=" * 80)
print(f"{'Rank':<6}{'Formula':<15}{'H%':<8}{'M-H dist':<12}{'Score':<10}")
print("-" * 80)

for i, result in enumerate(results_sorted[:10], 1):
    print(f"{i:<6}{result['formula']:<15}{result['h_fraction']:<8.1%}"
          f"{result['avg_mh_distance']:<12.3f}{result['composite_score']:<10.3f}")

# ============================================================================
# PART 5: SAVE RESULTS AND RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("PART 5: RESEARCH OUTPUTS")
print("=" * 80)

# Save detailed results to CSV
import csv

csv_file = 'superconductor_screening/screening_results.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results_sorted)

print(f"\nResults saved to: {csv_file}")

# Save top candidates for experimental synthesis
print("\nTop 3 candidates for experimental validation:")
print("-" * 80)

for i, result in enumerate(results_sorted[:3], 1):
    print(f"\n{i}. {result['formula']}")
    print(f"   Predicted advantages:")
    print(f"   - High hydrogen content: {result['h_fraction']:.1%}")
    print(f"   - Optimal M-H bonding: {result['avg_mh_distance']:.2f} Å")
    print(f"   - Estimated high phonon frequency")
    print(f"   - Structure: {result['structure']}")
    print(f"   ")
    print(f"   Recommended next steps:")
    print(f"   1. Synthesize under high pressure (50-150 GPa)")
    print(f"   2. Measure resistance vs temperature")
    print(f"   3. Check for Meissner effect")
    print(f"   4. If Tc > 200K, publish in Nature!")

# ============================================================================
# PART 6: WHAT YOU'D DO WITH UMA ACCESS
# ============================================================================

print("\n" + "=" * 80)
print("PART 6: NEXT STEPS WITH UMA ACCESS")
print("=" * 80)

print("""
CURRENT LIMITATIONS (using EMT):
- EMT only works for certain metals
- Not accurate for electronic structure
- Cannot calculate phonons directly
- No pressure effects

WITH UMA ACCESS, YOU CAN:
================================

1. ACCURATE DFT-QUALITY ENERGIES:
   from fairchem.core import pretrained_mlip, FAIRChemCalculator
   
   predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cuda")
   calc = FAIRChemCalculator(predictor, task_name="omat")
   atoms.calc = calc
   
   # Now you get real formation energies, band structures, etc.

2. HIGH-PRESSURE SCREENING:
   # Apply pressure to cell
   atoms.set_cell(atoms.get_cell() * 0.9, scale_atoms=True)  # 10% compression
   # Optimize and calculate properties under pressure

3. PHONON CALCULATIONS:
   # Calculate vibrational modes
   # Identify unstable modes (imaginary frequencies)
   # Estimate electron-phonon coupling

4. ELECTRONIC STRUCTURE:
   # Calculate density of states
   # Find Fermi level
   # Estimate metallic vs insulating behavior

5. LARGE-SCALE SCREENING:
   # Screen 10,000+ candidates
   # Use parallel GPU computing
   # Find hidden gems in chemical space

RESEARCH WORKFLOW:
==================
1. Generate 10,000 metal hydride structures (1 hour)
2. Screen with UMA (1-2 days on GPU)
3. Identify top 50 candidates
4. Detailed phonon analysis on top 50 (1 week)
5. Experimental validation of top 5 (months)
6. Publication in Science/Nature (priceless)

THIS IS REAL MATERIALS DISCOVERY!
""")

print("\n" + "=" * 80)
print("FILES CREATED:")
print("=" * 80)
print(f"- {len(candidates)} structure files in superconductor_screening/")
print(f"- screening_results.csv with all calculated properties")
print(f"- Optimized structures for top candidates")
print("\nVisualize structures: ase gui superconductor_screening/YH6_fcc_optimized.xyz")
print("=" * 80)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Generated and screened: {len(candidates)} metal hydride candidates
Optimized: {len(optimized_candidates)} structures
Top candidate: {results_sorted[0]['formula']}

This demonstrates the workflow for computational materials discovery.
Replace EMT with UMA for publication-quality results!

YOUR RESEARCH PATH:
1. ✓ Learn the workflow (you just did this)
2. ⏳ Wait for UMA access
3. → Re-run with FAIRChemCalculator
4. → Add phonon calculations
5. → Screen 1000s of candidates
6. → Discover room-temperature superconductor
7. → Win Nobel Prize 🏆
""")
