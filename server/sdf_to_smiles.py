#!/usr/bin/env python3
"""
Convert SDF (Structure Data File) to SMILES (Simplified Molecular Input Line Entry System)
Usage:
    python sdf_to_smiles.py input.sdf output.smi [options]
"""

import argparse
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, MolToSmiles


def convert_sdf_to_smiles(input_sdf: Path, output_smi: Path, 
                         remove_hydrogens: bool = True,
                         canonicalize: bool = True,
                         isomeric_smiles: bool = True) -> None:
    """
    Convert SDF file to SMILES file.
    
    Args:
        input_sdf: Path to input SDF file
        output_smi: Path to output SMILES file
        remove_hydrogens: Whether to remove explicit hydrogens
        canonicalize: Whether to canonicalize SMILES
        isomeric_smiles: Whether to include stereochemistry in SMILES
    """
    if not input_sdf.exists():
        raise FileNotFoundError(f"Input SDF file not found: {input_sdf}")
    
    # Read molecules from SDF
    supplier = SDMolSupplier(str(input_sdf))
    
    smiles_list = []
    valid_count = 0
    error_count = 0
    
    lines = []
    
    for i, mol in enumerate(supplier):
        if mol is None:
            error_count += 1
            print(f"Warning: Failed to read molecule {i+1}", file=sys.stderr)
            continue
            
        try:
            # Process molecule
            if remove_hydrogens:
                mol = Chem.RemoveHs(mol)
            
            # Generate SMILES
            smiles = MolToSmiles(mol, 
                                 canonical=canonicalize,
                                 isomericSmiles=isomeric_smiles)
            
            smiles_list.append(smiles)
            
            line = smiles + ' ' + f'gen{i:06d}'
            lines.append(line)
            valid_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"Warning: Failed to process molecule {i+1}: {e}", file=sys.stderr)
            continue
    
    # Write SMILES to file
    with open(output_smi, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    
    # Print summary
    print(f"Conversion complete:")
    print(f"  Input SDF: {input_sdf}")
    print(f"  Output SMILES: {output_smi}")
    print(f"  Total molecules processed: {valid_count + error_count}")
    print(f"  Successfully converted: {valid_count}")
    print(f"  Failed to process: {error_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert SDF files to SMILES format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sdf_to_smiles.py input.sdf output.smi
  python sdf_to_smiles.py molecules.sdf smiles.txt --no-canonicalize
  python sdf_to_smiles.py test.sdf test.smi --keep-hydrogens --no-isomeric
        """
    )
    
    parser.add_argument('input_sdf', type=Path, help='Input SDF file path')
    parser.add_argument('output_smi', type=Path, help='Output SMILES file path')
    parser.add_argument('--keep-hydrogens', action='store_true',
                       help='Keep explicit hydrogens (default: remove hydrogens)')
    parser.add_argument('--no-canonicalize', action='store_true',
                       help='Do not canonicalize SMILES (default: canonicalize)')
    parser.add_argument('--no-isomeric', action='store_true',
                       help='Do not include stereochemistry (default: include)')
    
    args = parser.parse_args()
    
    try:
        convert_sdf_to_smiles(
            input_sdf=args.input_sdf,
            output_smi=args.output_smi,
            remove_hydrogens=not args.keep_hydrogens,
            canonicalize=not args.no_canonicalize,
            isomeric_smiles=not args.no_isomeric
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
