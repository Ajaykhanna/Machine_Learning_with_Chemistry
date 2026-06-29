#!/usr/bin/env python3
import os
import re

def extract_optimized_geometry(log_path):
    """
    Parses a Gaussian log file to isolate the final 'Standard orientation'
    belonging to the completely optimized geometry block.
    
    Returns:
        list of tuples: [(atomic_number_int, "formatted coordinate line string"), ...]
    """
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at '{log_path}'")
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Locate all 'Standard orientation' blocks to trace the final converged geometry step
    blocks = list(re.finditer(
        r"Standard orientation:\s*\n\s*-+\s*\n\s*Center\s+Atomic\s+Atomic\s+Coordinates\s+\(Angstroms\)\s*\n\s*Number\s+Number\s+Type\s+X\s+Y\s+Z\s*\n\s*-+", 
        content
    ))
    
    if not blocks:
        # Fallback to Input orientation if Standard orientation is absent
        blocks = list(re.finditer(
            r"Input orientation:\s*\n\s*-+\s*\n\s*Center\s+Atomic\s+Atomic\s+Coordinates\s+\(Angstroms\)\s*\n\s*Number\s+Number\s+Type\s+X\s+Y\s+Z\s*\n\s*-+", 
            content
        ))
    
    if not blocks:
        print(f"Error: Could not locate coordinate orientation tables in {log_path}")
        return None

    # Isolate the absolute last geometry configuration block (fully optimized step)
    last_block = blocks[-1]
    start_idx = last_block.end()
    
    atoms = []
    lines = content[start_idx:].split('\n')
    for line in lines:
        if line.strip().startswith('---'):
            break
        parts = line.split()
        if len(parts) == 6:
            # Columns: Center No., Atomic No., Atomic Type, X, Y, Z
            atomic_num = parts[1]
            x, y, z = parts[3], parts[4], parts[5]
            
            # Keep layout formatted cleanly as exact string chunks to preserve float precision
            formatted_line = f"{atomic_num:>3}   {x:>12}  {y:>12}  {z:>12}"
            atoms.append((int(atomic_num), formatted_line))
            
    if not atoms:
        print(f"Error: Standard orientation table was parsed but no atoms were extracted from {log_path}")
        return None

    # Sort array internally from highest atomic number (Z) to lowest descending
    sorted_atoms = sorted(atoms, key=lambda x: x[0], reverse=True)
    return sorted_atoms

def write_xyz_validation_file(xyz_path, sorted_atoms):
    """Writes a standard structural verification .xyz backup file for inspection."""
    atomic_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 35: 'Br'}
    
    with open(xyz_path, 'w') as f:
        f.write(f"{len(sorted_atoms)}\n")
        f.write("Extracted optimized geometry (Sorted by Highest Z Descending)\n")
        for num, atom_line in sorted_atoms:
            parts = atom_line.split()
            sym = atomic_to_symbol.get(num, 'X')
            f.write(f"{sym:<2}  {parts[1]:>12}  {parts[2]:>12}  {parts[3]:>12}\n")


