## from https://charmm-gui.org/?doc=lecture&module=scientific&lesson=11

#!/usr/bin/env python
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import copy
import numpy as np
import biotite.structure.io as bsio


def fit_rms(ref_c,c):
    # move geometric center to the origin
    ref_trans = np.average(ref_c, axis=0)
    ref_c = ref_c - ref_trans
    c_trans = np.average(c, axis=0)
    c = c - c_trans

    # covariance matrix
    C = np.dot(c.T, ref_c)

    # Singular Value Decomposition
    (r1, s, r2) = np.linalg.svd(C)

    # compute sign (remove mirroring)
    if np.linalg.det(C) < 0:
        r2[2,:] *= -1.0
    U = np.dot(r1, r2)
    return (c_trans, U, ref_trans)


def set_rmsd(c1, c2):
    rmsd = 0.0
    c_trans, U, ref_trans = fit_rms(c1, c2)
    new_c2 = np.dot(c2 - c_trans, U) + ref_trans
    rmsd = np.sqrt( np.average( np.sum( ( c1 - new_c2 )**2, axis=1 ) ) )
    return rmsd

def get_aligned_coord(self, atoms, name=None):
    new_c2 = copy.deepcopy(atoms)
    for atom in new_c2:
        atom.x, atom.y, atom.z = np.dot(np.array([atom.x, atom.y, atom.z]) - self.c_trans, self.U) + self.ref_trans
    return new_c2


pdbf1 = '1Y3Q.pdb'
struct1 = bsio.load_structure(pdbf1, extra_fields=["b_factor"])

pdbf2 = '1Y3N.pdb'
struct2 = bsio.load_structure(pdbf2, extra_fields=["b_factor"])

atoms1 = []
for atom in struct1:
    if atom.atom_name == "CA":
        atoms1.append(atom.coord)
atoms1 =  np.array(atoms1)

atoms2 = []
for atom in struct2:
    if atom.atom_name == "CA":
        atoms2.append(atom.coord)
atoms2 =  np.array(atoms2)

rmsd = set_rmsd(atoms1, atoms2)


