#!/usr/bin/env python
"""
SUPERCONDUCTOR MATERIALS EXPLORATION
Using Materials Project Database

This script works RIGHT NOW - no UMA required!
Explores known metal hydrides and prepares candidates for UMA screening

Get your FREE API key: https://materialsproject.org/api
Then: export MP_API_KEY='your_key_here'
"""

import numpy as np
from ase.io import write, read
from ase import Atoms
import os
import json
import csv

print("=" * 80)
print("SUPERCONDUCTOR MATERIALS EXPLORATION")
print("Materials Project Database Analysis")
print("=" * 80)

# Create output directory
os.makedirs('materials_project_exploration', exist_ok=True)

# ============================================================================
# PART 1: CHECK MATERIALS PROJECT ACCESS
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: CHECKING MATERIALS PROJECT ACCESS")
print("=" * 80)

mp_available = False
mp_api_key = None

# Check for API key in environment
mp_api_key = os.environ.get('MP_API_KEY')

if not mp_api_key:
    print("\n⚠ MP_API_KEY not found in environment")
    print("\nTo get started with Materials Project:")
    print("=" * 80)
    print("1. Go to: https://materialsproject.org/")
    print("2. Click 'Sign Up' (top right)")
    print("3. Register with your email")
    print("4. Go to: https://materialsproject.org/api")
    print("5. Click 'Generate API Key'")
    print("6. Copy your key")
    print("\n7. Set environment variable:")
    print("   export MP_API_KEY='your_key_here'")
    print("\n   Or add to ~/.bashrc for permanent:")
    print("   echo 'export MP_API_KEY=\"your_key_here\"' >> ~/.bashrc")
    print("   source ~/.bashrc")
    print("=" * 80)
    
    # Ask user if they want to enter it now
    print("\nDo you have an API key you want to use right now?")
    user_input = input("Enter your API key (or press Enter to skip): ").strip()
    
    if user_input:
        mp_api_key = user_input
        print(f"Using provided API key: {mp_api_key[:8]}...")
    else:
        print("\nSkipping Materials Project for now.")
        print("You can still explore the generated structures!")
        mp_api_key = None

# Try to import and use Materials Project
if mp_api_key:
    try:
        from mp_api.client import MPRester
        print("✓ mp-api installed")
        
        # Test the API key
        print("Testing API key...")
        with MPRester(mp_api_key) as mpr:
            # Simple test query
            test = mpr.materials.summary.search(
                formula="H2O",
                num_elements=(1, 2),
                fields=["material_id"]
            )
            if test:
                print(f"✓ API key valid! (Found {len(test)} H2O structures)")
                mp_available = True
            else:
                print("✗ API key seems invalid or query failed")
                
    except ImportError:
        print("✗ mp-api not installed")
        print("\nInstall with:")
        print("  conda install -c conda-forge mp-api")
        print("\nOr:")
        print("  pip install mp-api")
        
    except Exception as e:
        print(f"✗ Error with Materials Project: {e}")
        print("Check your API key and internet connection")

if not mp_available:
    print("\n" + "!" * 80)
    print("Continuing without Materials Project")
    print("Will generate structures only")
    print("!" * 80)

# ============================================================================
# PART 2: QUERY MATERIALS PROJECT FOR KNOWN HYDRIDES
# ============================================================================

if mp_available:
    print("\n" + "=" * 80)
    print("PART 2: QUERYING MATERIALS PROJECT FOR METAL HYDRIDES")
    print("=" * 80)
    
    print("""
    SEARCHING FOR:
    - Metal hydrides (M-H systems)
    - Promising superconductor candidates
    - Known experimental structures
    
    This will help us:
    1. See what's already been discovered
    2. Use real structures as templates
    3. Identify gaps to fill with new candidates
    """)
    
    known_hydrides = []
    
    # Metals known for high-Tc hydrides
    target_metals = {
        'La': 'Lanthanum (LaH10: Tc=250K @ 170GPa)',
        'Y': 'Yttrium (YH6: Tc=220K @ 166GPa)',
        'Ce': 'Cerium (CeH9: Tc=195K @ 100GPa)',
        'Ca': 'Calcium (CaH6: Tc=215K @ 172GPa)',
        'Th': 'Thorium (ThH10: predicted high-Tc)',
        'Sc': 'Scandium (ScH3: superconductor)',
        'Mg': 'Magnesium (MgH2: hydrogen storage)',
        'Li': 'Lithium (LiH: simple hydride)',
    }
    
    print("\nSearching Materials Project database...")
    print("-" * 80)
    
    all_results = {}
    
    with MPRester(mp_api_key) as mpr:
        for metal, description in target_metals.items():
            print(f"\n{metal} ({description}):")
            
            try:
                # Search for binary M-H compounds
                docs = mpr.materials.summary.search(
                    elements=[metal, 'H'],
                    num_elements=(1, 2),  # Binary only
                    fields=["material_id", "formula_pretty", "energy_per_atom",
                           "formation_energy_per_atom", "band_gap", "structure",
                           "nsites", "volume", "density"]
                )
                
                metal_results = []
                
                for doc in docs:
                    # Extract info
                    hydride_info = {
                        'mp_id': str(doc.material_id),
                        'formula': doc.formula_pretty,
                        'energy_per_atom': float(doc.energy_per_atom),
                        'formation_energy': float(doc.formation_energy_per_atom),
                        'band_gap': float(doc.band_gap),
                        'n_sites': int(doc.nsites),
                        'volume': float(doc.volume),
                        'density': float(doc.density),
                        'metal': metal,
                        'structure': doc.structure
                    }
                    
                    # Calculate H ratio
                    formula = doc.formula_pretty
                    # Simple parsing (works for MHx format)
                    if 'H' in formula:
                        parts = formula.replace(metal, '').replace('H', ' H').strip().split()
                        if len(parts) > 1 and parts[1].isdigit():
                            h_ratio = int(parts[1])
                        elif 'H' in parts:
                            h_ratio = 1
                        else:
                            h_ratio = 0
                    else:
                        h_ratio = 0
                    
                    hydride_info['h_ratio'] = h_ratio
                    
                    metal_results.append(hydride_info)
                    known_hydrides.append(hydride_info)
                    
                    # Print info
                    is_metal = "METAL" if doc.band_gap < 0.1 else f"Gap={doc.band_gap:.2f}eV"
                    print(f"  {doc.formula_pretty:10} (mp-{doc.material_id:8}) "
                          f"E={doc.formation_energy_per_atom:6.3f} eV/atom  {is_metal}")
                
                all_results[metal] = metal_results
                print(f"  → Found {len(metal_results)} {metal}-H compounds")
                
            except Exception as e:
                print(f"  Error querying {metal}: {e}")
    
    print(f"\n{'=' * 80}")
    print(f"TOTAL HYDRIDES FOUND: {len(known_hydrides)}")
    print(f"{'=' * 80}")
    
    # ========================================================================
    # PART 3: ANALYZE KNOWN HYDRIDES
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PART 3: ANALYZING KNOWN HYDRIDES")
    print("=" * 80)
    
    print("\nHYDROGEN CONTENT DISTRIBUTION:")
    print("-" * 80)
    
    h_ratio_counts = {}
    for h in known_hydrides:
        ratio = h['h_ratio']
        if ratio not in h_ratio_counts:
            h_ratio_counts[ratio] = []
        h_ratio_counts[ratio].append(h['formula'])
    
    for ratio in sorted(h_ratio_counts.keys()):
        compounds = h_ratio_counts[ratio]
        print(f"MH{ratio if ratio > 0 else '?'}: {len(compounds):3} compounds")
        print(f"  Examples: {', '.join(compounds[:5])}")
    
    print("\nMETALLIC vs INSULATING:")
    print("-" * 80)
    
    metallic = [h for h in known_hydrides if h['band_gap'] < 0.1]
    insulating = [h for h in known_hydrides if h['band_gap'] >= 0.1]
    
    print(f"Metallic (gap < 0.1 eV): {len(metallic)}")
    print(f"Insulating (gap ≥ 0.1 eV): {len(insulating)}")
    
    if metallic:
        print("\nMETALLIC HYDRIDES (Superconductor candidates):")
        for h in sorted(metallic, key=lambda x: x['h_ratio'], reverse=True)[:10]:
            print(f"  {h['formula']:10} H-ratio={h['h_ratio']} "
                  f"E={h['formation_energy']:6.3f} eV/atom")
    
    print("\nSTABILITY ANALYSIS:")
    print("-" * 80)
    
    stable = [h for h in known_hydrides if h['formation_energy'] < -0.5]
    metastable = [h for h in known_hydrides if -0.5 <= h['formation_energy'] < 0]
    unstable = [h for h in known_hydrides if h['formation_energy'] >= 0]
    
    print(f"Stable (E < -0.5 eV/atom): {len(stable)}")
    print(f"Metastable (-0.5 to 0 eV/atom): {len(metastable)}")
    print(f"Unstable (E ≥ 0 eV/atom): {len(unstable)}")
    
    # ========================================================================
    # PART 4: SAVE KNOWN STRUCTURES
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PART 4: SAVING KNOWN STRUCTURES")
    print("=" * 80)
    
    print("\nConverting structures to XYZ format...")
    
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        
        saved_count = 0
        for hydride in known_hydrides:
            try:
                # Convert pymatgen structure to ASE atoms
                atoms = AseAtomsAdaptor.get_atoms(hydride['structure'])
                
                # Save to file
                filename = f"materials_project_exploration/MP_{hydride['formula']}_{hydride['mp_id']}.xyz"
                write(filename, atoms)
                saved_count += 1
                
            except Exception as e:
                print(f"  Failed to save {hydride['formula']}: {e}")
        
        print(f"✓ Saved {saved_count} structure files")
        
    except ImportError:
        print("⚠ pymatgen not found - cannot save structures")
        print("  Install with: conda install -c conda-forge pymatgen")
    
    # ========================================================================
    # PART 5: IDENTIFY RESEARCH OPPORTUNITIES
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PART 5: RESEARCH OPPORTUNITIES")
    print("=" * 80)
    
    print("""
    Based on Materials Project data, here are promising research directions:
    """)
    
    print("\n1. HIGH HYDROGEN CONTENT (Unexplored):")
    print("-" * 80)
    
    # Check which high-H compounds are missing
    high_h_targets = {}
    for metal in target_metals.keys():
        metal_hydrides = [h for h in known_hydrides if h['metal'] == metal]
        existing_ratios = [h['h_ratio'] for h in metal_hydrides]
        
        # High-Tc hydrides typically have H6, H9, H10
        target_ratios = [6, 9, 10, 12]
        missing = [r for r in target_ratios if r not in existing_ratios]
        
        if missing:
            high_h_targets[metal] = missing
    
    print("These high-H compositions are NOT in Materials Project:")
    for metal, ratios in high_h_targets.items():
        print(f"  {metal}: {', '.join([f'{metal}H{r}' for r in ratios])}")
    
    print("\n2. METALLIC HYDRIDES (Superconductor Candidates):")
    print("-" * 80)
    
    metallic_sorted = sorted(metallic, key=lambda x: x['h_ratio'], reverse=True)
    print(f"\nTop metallic hydrides by H content:")
    for h in metallic_sorted[:5]:
        print(f"  {h['formula']:10} - Use as template for variations")
    
    print("\n3. UNSTABLE BUT INTERESTING:")
    print("-" * 80)
    print("These are unstable at ambient pressure but might work under compression:")
    
    for h in sorted(unstable, key=lambda x: x['h_ratio'], reverse=True)[:5]:
        if h['h_ratio'] >= 6:  # High H content
            print(f"  {h['formula']:10} E={h['formation_energy']:6.3f} eV/atom")
            print(f"    → Try at high pressure (100-200 GPa)")

# ============================================================================
# PART 6: GENERATE NEW CANDIDATE STRUCTURES
# ============================================================================

print("\n" + "=" * 80)
print("PART 6: GENERATING NEW SUPERCONDUCTOR CANDIDATES")
print("=" * 80)

print("""
Based on known high-Tc superconductors and Materials Project gaps,
generating promising new candidates for UMA screening...
""")

def create_clathrate_hydride(metal, h_count, lattice=4.5):
    """
    Create hydrogen clathrate structure (cage around metal)
    This geometry is found in high-Tc hydrides like LaH10
    """
    # Metal at origin
    metal_pos = np.array([[0, 0, 0]])
    
    # Create H cage
    h_positions = []
    
    if h_count == 10:
        # Dodecahedral cage (LaH10-type)
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        scale = lattice * 0.3
        h_positions = [
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
            [0, 1, phi], [0, -1, phi]
        ]
        h_positions = (np.array(h_positions) * scale).tolist()
        
    elif h_count == 9:
        # Similar but one vertex removed
        phi = (1 + np.sqrt(5)) / 2
        scale = lattice * 0.3
        h_positions = [
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
            [0, 1, phi]
        ]
        h_positions = (np.array(h_positions) * scale).tolist()
        
    elif h_count == 6:
        # Octahedral cage
        scale = lattice * 0.35
        h_positions = [
            [scale, 0, 0], [-scale, 0, 0],
            [0, scale, 0], [0, -scale, 0],
            [0, 0, scale], [0, 0, -scale]
        ]
        
    elif h_count == 12:
        # Icosahedral cage
        phi = (1 + np.sqrt(5)) / 2
        scale = lattice * 0.28
        h_positions = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        h_positions = (np.array(h_positions) * scale).tolist()
    
    else:
        # Default: distribute on sphere
        scale = lattice * 0.3
        for i in range(h_count):
            theta = 2 * np.pi * i / h_count
            phi_angle = np.pi * (i % (h_count // 2)) / (h_count // 2)
            h_positions.append([
                scale * np.sin(phi_angle) * np.cos(theta),
                scale * np.sin(phi_angle) * np.sin(theta),
                scale * np.cos(phi_angle)
            ])
    
    all_positions = np.vstack([metal_pos, h_positions])
    symbols = [metal] + ['H'] * h_count
    
    atoms = Atoms(
        symbols=symbols,
        positions=all_positions,
        cell=[lattice, lattice, lattice],
        pbc=True
    )
    
    return atoms

# Generate candidates based on research opportunities
candidates = []

# Target compositions based on gaps in Materials Project
target_compositions = [
    ('La', 10), ('La', 9), ('La', 12),
    ('Y', 10), ('Y', 9), ('Y', 12),
    ('Ce', 10), ('Ce', 9), ('Ce', 6),
    ('Ca', 10), ('Ca', 9), ('Ca', 12),
    ('Th', 10), ('Th', 9), ('Th', 6),
    ('Sc', 6), ('Sc', 9), ('Sc', 10),
]

# Different "pressures" (lattice compression)
lattice_constants = [4.0, 4.5, 5.0, 5.5]  # Å

print("\nGenerating high-Tc candidates:")
print("-" * 80)

for metal, h_ratio in target_compositions:
    for latt in lattice_constants:
        atoms = create_clathrate_hydride(metal, h_ratio, latt)
        formula = atoms.get_chemical_formula()
        
        # Estimate pressure from compression
        # Smaller lattice = higher pressure (rough estimate)
        pressure_gpa = int((5.5 - latt) / 0.5 * 50)
        
        filename = f"materials_project_exploration/CANDIDATE_{formula}_a{latt:.1f}.xyz"
        write(filename, atoms)
        
        candidates.append({
            'metal': metal,
            'h_ratio': h_ratio,
            'formula': formula,
            'lattice': latt,
            'pressure_gpa': pressure_gpa,
            'filename': filename
        })
        
        # Print if it's a gap in Materials Project
        if mp_available:
            exists = any(h['formula'] == f"{metal}H{h_ratio}" for h in known_hydrides)
            status = "EXISTS in MP" if exists else "NEW candidate!"
        else:
            status = "NEW candidate!"
        
        print(f"{formula:10} (a={latt:.1f}Å, ~{pressure_gpa:3}GPa) - {status}")

print(f"\n✓ Generated {len(candidates)} candidate structures")

# ============================================================================
# PART 7: SAVE RESULTS AND SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PART 7: SAVING RESEARCH OUTPUTS")
print("=" * 80)

# Save candidate list
candidates_file = 'materials_project_exploration/candidates_for_uma.json'
with open(candidates_file, 'w') as f:
    json.dump(candidates, f, indent=2)

print(f"✓ Candidate list saved: {candidates_file}")

# Save known hydrides data if available
if mp_available:
    mp_data_file = 'materials_project_exploration/known_hydrides.json'
    with open(mp_data_file, 'w') as f:
        export_hydrides = []
        for h in known_hydrides:
            export = {k: v for k, v in h.items() if k != 'structure'}
            export_hydrides.append(export)
        json.dump(export_hydrides, f, indent=2)
    
    print(f"✓ Known hydrides data saved: {mp_data_file}")
    
    # Save CSV for easy viewing
    csv_file = 'materials_project_exploration/known_hydrides.csv'
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['formula', 'mp_id', 'metal', 'h_ratio', 'formation_energy', 
                     'band_gap', 'energy_per_atom', 'density']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in known_hydrides:
            row = {k: h[k] for k in fieldnames}
            writer.writerow(row)
    
    print(f"✓ Known hydrides CSV saved: {csv_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if mp_available:
    print(f"""
Materials Project Analysis:
  - Queried {len(target_metals)} metal-H systems
  - Found {len(known_hydrides)} known hydrides
  - {len(metallic)} metallic (superconductor candidates)
  - {len(stable)} thermodynamically stable

Generated {len(candidates)} new candidates for screening

FILES CREATED:
  - {len(candidates)} candidate structure XYZ files
  - {len(known_hydrides) if mp_available else 0} known structure XYZ files
  - candidates_for_uma.json (ready for UMA screening)
  - known_hydrides.json (Materials Project data)
  - known_hydrides.csv (spreadsheet format)

NEXT STEPS:
""")
else:
    print(f"""
Generated {len(candidates)} new candidates for screening

FILES CREATED:
  - {len(candidates)} candidate structure XYZ files  
  - candidates_for_uma.json (ready for UMA screening)

TO ENABLE MATERIALS PROJECT:
  1. Get free API key: https://materialsproject.org/api
  2. Set: export MP_API_KEY='your_key_here'
  3. Re-run this script

NEXT STEPS:
""")

print("""
1. VISUALIZE STRUCTURES (Do this now!):
   ase gui materials_project_exploration/CANDIDATE_LaH10_a4.5.xyz
   
2. WHEN UMA ACCESS GRANTED:
   - Run: python superconductor_uma_discovery.py
   - It will screen all these candidates
   - Get DFT-quality predictions
   - Rank by superconductivity potential

3. EXPERIMENTAL VALIDATION:
   - Top 3 candidates go to diamond anvil cell
   - Synthesize at predicted pressure
   - Measure resistance vs temperature
   - Look for Tc > 200K!

4. PUBLICATION:
   - If Tc > room temperature → Nature/Science
   - Change the world 🚀
""")

print("=" * 80)
print("EXPLORATION COMPLETE!")
print("=" * 80)

if not mp_available:
    print("\n💡 TIP: Get Materials Project access to unlock:")
    print("  - 150,000+ known structures")
    print("  - Experimental validation data")
    print("  - Better starting points for optimization")
    print("  - Identify what's never been tried!")
