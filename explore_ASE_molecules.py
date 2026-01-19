#!/usr/bin/env python
"""
Explore Molecular Simulation with ASE (No Model Required!)
This script demonstrates what you can do RIGHT NOW without waiting for UMA access
"""

from ase.build import molecule, bulk, fcc111, add_adsorbate
from ase.visualize import view
from ase import Atoms
import matplotlib.pyplot as plt
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase import units
from ase.io import Trajectory, write
import numpy as np

print("=" * 70)
print("ASE Molecular Simulation - Getting Started")
print("=" * 70)

# 1. CREATE MOLECULES
print("\n1. Creating Various Molecular Structures")
print("-" * 70)

# Water molecule
h2o = molecule('H2O')
print(f"Water (H2O): {len(h2o)} atoms")
print(f"  Formula: {h2o.get_chemical_formula()}")
print(f"  Positions:\n{h2o.get_positions()}")

# Methane molecule
methane = molecule('CH4')
print(f"\nMethane (CH4): {len(methane)} atoms")
print(f"  Formula: {methane.get_chemical_formula()}")

# Benzene
benzene = molecule('C6H6')
print(f"\nBenzene: {len(benzene)} atoms")
print(f"  Formula: {benzene.get_chemical_formula()}")

# Ethanol
ethanol = molecule('CH3CH2OH')
print(f"\nEthanol: {len(ethanol)} atoms")
print(f"  Formula: {ethanol.get_chemical_formula()}")

# CO2
co2 = molecule('CO2')
print(f"\nCO2: {len(co2)} atoms")
print(f"  Formula: {co2.get_chemical_formula()}")


# 2. CREATE CRYSTAL STRUCTURES
print("\n2. Creating Crystal Structures")
print("-" * 70)

# Copper FCC crystal
cu_bulk = bulk('Cu', 'fcc', a=3.6)
print(f"Copper crystal: {len(cu_bulk)} atoms")
print(f"  Cell:\n{cu_bulk.get_cell()}")

# Iron BCC crystal
fe_bulk = bulk('Fe', 'bcc', a=2.87)
print(f"Iron crystal: {len(fe_bulk)} atoms")

# Silicon diamond structure
si_bulk = bulk('Si', 'diamond', a=5.43)
print(f"Silicon crystal: {len(si_bulk)} atoms")


# 3. CREATE SURFACE STRUCTURES
print("\n3. Creating Surface Structures (Important for Catalysis)")
print("-" * 70)

# Copper (111) surface
cu_surface = fcc111('Cu', size=(3, 3, 4), vacuum=10.0)
print(f"Cu(111) surface: {len(cu_surface)} atoms")
print(f"  Includes vacuum layer for surface calculations")

# Add CO molecule to surface (catalysis example)
cu_with_co = cu_surface.copy()
co = molecule('CO')
add_adsorbate(cu_with_co, co, height=2.0, position='fcc')
print(f"\nCu(111) + CO adsorbate: {len(cu_with_co)} atoms")
print("  This simulates CO adsorption on copper surface")


# 4. SIMPLE ENERGY CALCULATION (Using EMT - Fast but Limited)
print("\n4. Energy Calculation with EMT Calculator")
print("-" * 70)
print("EMT = Effective Medium Theory (fast, but only for metals)")

# Calculate energy for copper
cu_small = bulk('Cu', 'fcc', a=3.6).repeat((2, 2, 2))
cu_small.calc = EMT()
energy = cu_small.get_potential_energy()
forces = cu_small.get_forces()

print(f"Copper cluster energy: {energy:.4f} eV")
print(f"Maximum force: {np.max(np.abs(forces)):.4f} eV/Å")


# 5. GEOMETRY OPTIMIZATION
print("\n5. Geometry Optimization")
print("-" * 70)

# Optimize copper surface with CO
cu_co_opt = cu_with_co.copy()
cu_co_opt.calc = EMT()

# Run optimization
opt = BFGS(cu_co_opt, logfile='optimization.log')
print("Optimizing Cu(111) + CO structure...")
opt.run(fmax=0.05, steps=50)

final_energy = cu_co_opt.get_potential_energy()
print(f"Final optimized energy: {final_energy:.4f} eV")


# 6. MOLECULAR DYNAMICS
print("\n6. Molecular Dynamics Simulation")
print("-" * 70)

# Simple MD on copper cluster
cu_md = bulk('Cu', 'fcc', a=3.6).repeat((3, 3, 3))
cu_md.calc = EMT()

# Set up MD
dyn = Langevin(
    cu_md,
    timestep=1.0 * units.fs,
    temperature_K=300,
    friction=0.002
)

# Run short MD
traj = Trajectory('cu_md.traj', 'w', cu_md)
dyn.attach(traj.write, interval=5)

print("Running 100 MD steps...")
energies = []
for i in range(100):
    dyn.run(1)
    if i % 10 == 0:
        energies.append(cu_md.get_potential_energy())
        print(f"  Step {i}: E = {energies[-1]:.4f} eV")

traj.close()
print(f"MD trajectory saved to: cu_md.traj")


# 7. AVAILABLE MOLECULES IN ASE
print("\n7. Built-in Molecules Available in ASE")
print("-" * 70)

common_molecules = [
    'H2', 'H2O', 'NH3', 'CH4', 'C2H6', 'C2H4', 'C2H2',
    'CO', 'CO2', 'NO', 'NO2', 'N2', 'O2',
    'C6H6',  # benzene
    'CH3OH',  # methanol
    'C2H5OH',  # ethanol
]

print("Some commonly used molecules:")
for mol_name in common_molecules[:10]:
    try:
        mol = molecule(mol_name)
        print(f"  {mol_name:10} - {mol.get_chemical_formula():10} - {len(mol):2} atoms")
    except:
        pass


# 8. SAVE STRUCTURES
print("\n8. Saving Structures to Files")
print("-" * 70)

# Save in various formats
write('water.xyz', h2o)
print("Saved: water.xyz")

write('methane.xyz', methane)
print("Saved: methane.xyz")

write('benzene.xyz', benzene)
print("Saved: benzene.xyz")

write('cu_surface.xyz', cu_surface)
print("Saved: cu_surface.xyz")

write('cu_with_co.xyz', cu_with_co)
print("Saved: cu_with_co.xyz")


# 9. WHAT'S NEXT?
print("\n" + "=" * 70)
print("WHAT YOU CAN DO NOW (While Waiting for UMA Access):")
print("=" * 70)
print("""
1. EXPLORE ASE MOLECULES:
   - Build any molecule using ASE's molecule() function
   - Create crystal structures
   - Make surface slabs for catalysis studies

2. PRACTICE WITH EMT CALCULATOR:
   - Works for: Cu, Ag, Au, Ni, Pd, Pt, Al
   - Fast calculations (good for learning workflow)
   - Limited to metals only

3. PREPARE YOUR RESEARCH:
   - Design the molecules you want to study
   - Set up your analysis workflows
   - Learn ASE basics (optimization, MD, visualization)

4. WHEN UMA ACCESS IS GRANTED:
   - Simply replace EMT calculator with FAIRChemCalculator
   - Everything else stays the same!
   - You'll get accurate DFT-quality results

5. ALTERNATIVE DATASETS TO EXPLORE:
   - QM9 dataset (small organic molecules)
   - Materials Project (inorganic crystals)  
   - COD (Crystallography Open Database)
""")

print("\n" + "=" * 70)
print("FILES CREATED IN CURRENT DIRECTORY:")
print("=" * 70)
print("- water.xyz, methane.xyz, benzene.xyz")
print("- cu_surface.xyz, cu_with_co.xyz")
print("- cu_md.traj (MD trajectory)")
print("- optimization.log (optimization log)")
print("\nYou can visualize these with: ase gui filename.xyz")
print("=" * 70)
