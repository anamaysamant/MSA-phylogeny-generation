import subprocess
from Bio.PDB import PDBParser, Superimposer

def run_tmalign(pdb1, pdb2):
    """Runs TM-align and returns its output."""
    result = subprocess.run(['TMalign', pdb1, pdb2], capture_output=True, text=True)
    return result.stdout

def parse_tmalign_alignment(output):
    """Parses aligned residue indices from TM-align output."""
    lines = output.splitlines()
    
    # Extract alignment lines (look for '(":" denotes structurally aligned residues)')
    for i, line in enumerate(lines):
        if line.startswith("(") and 'denotes structurally aligned residues' in line:
            seq1 = lines[i + 1].strip()
            aln  = lines[i + 2].strip()
            seq2 = lines[i + 3].strip()
            break

    # Parse aligned residues (non-gap, aligned positions)
    index1, index2 = 0, 0
    matched_indices = []

    for a, b, match in zip(seq1, seq2, aln):
        if a != '-': index1 += 1
        if b != '-': index2 += 1
        if match == ':':
            matched_indices.append((index1, index2))

    return matched_indices

def get_ca_atoms(pdb_file, indices, chain_id='A'):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    chain = structure[0][chain_id]

    # Create an ordered list of residues that have CA atoms
    ordered_residues = [res for res in chain if 'CA' in res]

    atoms = []
    for idx in indices:
        try:
            atom = ordered_residues[idx - 1]['CA']  # 1-based to 0-based
            atoms.append(atom)
        except IndexError:
            print(f"Residue index {idx} out of range")
    return atoms

def align_structures(pdb1, pdb2):
    print("Running TM-align...")
    tmalign_out = run_tmalign(pdb1, pdb2)
    matched_residues = parse_tmalign_alignment(tmalign_out)

    print(f"Matched residues: {len(matched_residues)}")

    res_ids1 = [i for i, _ in matched_residues]
    res_ids2 = [j for _, j in matched_residues]

    atoms1 = get_ca_atoms(pdb1, res_ids1)
    atoms2 = get_ca_atoms(pdb2, res_ids2)

    # Sanity check
    if len(atoms1) != len(atoms2):
        raise ValueError("Mismatched number of aligned atoms!")

    # Superimpose using Biopython
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(atoms2)

    return sup.rms

# Example usage
align_structures("structure1.pdb", "structure2.pdb")