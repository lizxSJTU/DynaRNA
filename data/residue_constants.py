# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants used in AlphaFold."""

import collections
import functools
import os
from typing import List, Mapping, Tuple

import numpy as np
import tree

# Internal import (35fd).


# Distance from one CA to next CA [trans configuration: omega = 180].
ca_ca = 6.3  #5.157 #3.80209737096

# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ADE and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms = {
    'ADE': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'URA': [],
    'GUA': [],
    'ASP': [],
    'CYT': [],
    'GLN': [],
    'GLU': [],
    'GLY': [],
    'HIS': [],
    'ILE': [],
    'LEU': [],
    'LYS': [],
    'MET': [],
    'PHE': [],
    'PRO': [],
    'SER': [],
    'THR': [],
    'TRP': [],
    'TYR': [],
    'VAL': [],
}

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ADE
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 0.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYT
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 0.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 0.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 0.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0],  # ADE
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 0.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYT
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 0.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 0.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 0.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

# Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
# psi and chi angles:
# 0: 'backbone group',
# 1: 'pre-omega-group', (empty)
# 2: 'phi-group', (currently empty, because it defines only hydrogens)
# 3: 'psi-group',
# 4,5,6,7: 'chi1,2,3,4-group'
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]
rigid_group_atom_positions = {
    'ADE': [
        ["O4'", 0, (1.450, -0.000, 0.000)],
        ["C4'", 0, (0.000, 0.000, 0.000)],
        ["C3'", 0, (-0.378, 1.475, 0.00)],
        ["C5'", 0, (-0.508, -0.803, -1.174)],
        ["O3'", 0, (0.524, 1.321, 0.000)],
    ],
    'URA': [
        ["O4'", 0, (-0.587 ,0.072, -1.330)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-1.086 ,0.532, 0.920)],
        ["C5'", 0, (0.433, -1.446 ,0.250)],  
        ["O3'", 0, (-0.555, 1.005, 2.150)], 
        
    ],
    'GUA': [
        ["O4'", 0, (-0.587,0.072,-1.330)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-1.086 ,0.532 ,0.920)],
        ["C5'", 0, (0.433 ,-1.446 ,0.250)],  
        ["O3'", 0, (-0.555 , 1.005 ,2.150)], 
    ],
    'ASP': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'CYT': [
        ["O4'", 0, (-0.454,0.379,-1.330)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.626,1.035 ,0.920)],
        ["C5'", 0, (-0.416,-1.450,0.250)],  
        ["O3'", 0, (0.077,1.146,2.150)], 
    ],
    'GLN': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'GLU': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'GLY': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'HIS': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'ILE': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'LEU': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'LYS': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'MET': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'PHE': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'PRO': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'SER': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'THR': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'TRP': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'TYR': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
    'VAL': [
        ["O4'", 0, (-0.792 , -0.002 , -1.202)],
        ["C4'", 0, (0.000000, 0.000000, 0.000000)],
        ["C3'", 0, (-0.835, 0.603,1.109)],
        ["C5'", 0, (0.580,-1.345,0.396)],  
        ["O3'", 0, (-0.059,1.309,2.080)], 
    ],
}

# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
residue_atoms = {
    'ADE': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'URA': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'GUA': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'ASP': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'CYT': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'GLU': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'GLN': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'GLY': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'HIS': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'ILE': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'LEU': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'LYS': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'MET': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'PHE': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'PRO': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'SER': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'THR': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'TRP': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'TYR': ["C3'", "C4'", "C5'", "O4'","O3'"],
    'VAL': ["C3'", "C4'", "C5'", "O4'","O3'"],
}

# Naming swaps for ambiguous atom names.
# Due to symmetries in the amino acids the naming of atoms is ambiguous in
# 4 of the 20 amino acids.
# (The LDDT paper lists 7 amino acids as ambiguous, but the naming ambiguities
# in LEU, VAL and ARG can be resolved by using the 3d constellations of
# the 'ambiguous' atoms and their neighbours)
residue_atom_renaming_swaps = {
    'ASP': {"C3'": "C3'"},
    'GLU': {"C3'": "C3'"},
    'PHE': {"C3'": "C3'", "C3'": "C3'"},
    'TYR': {"C3'": "C3'", "C3'": "C3'"},
}

# Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
van_der_waals_radius = {
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'S': 1.8,
}

Bond = collections.namedtuple(
    'Bond', ['atom1_name', 'atom2_name', 'length', 'stddev'])
BondAngle = collections.namedtuple(
    'BondAngle',
    ['atom1_name', 'atom2_name', 'atom3name', 'angle_rad', 'stddev'])


@functools.lru_cache(maxsize=None)
def load_stereo_chemical_props() -> Tuple[Mapping[str, List[Bond]],
                                          Mapping[str, List[Bond]],
                                          Mapping[str, List[BondAngle]]]:
  """Load stereo_chemical_props.txt into a nice structure.

  Load literature values for bond lengths and bond angles and translate
  bond angles into the length of the opposite edge of the triangle
  ("residue_virtual_bonds").

  Returns:
    residue_bonds: Dict that maps resname -> list of Bond tuples.
    residue_virtual_bonds: Dict that maps resname -> list of Bond tuples.
    residue_bond_angles: Dict that maps resname -> list of BondAngle tuples.
  """
  stereo_chemical_props_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), 'stereo_chemical_props.txt'
  )
  with open(stereo_chemical_props_path, 'rt') as f:
    stereo_chemical_props = f.read()
  lines_iter = iter(stereo_chemical_props.splitlines())
  # Load bond lengths.
  residue_bonds = {}
  next(lines_iter)  # Skip header line.
  for line in lines_iter:
    if line.strip() == '-':
      break
    bond, resname, length, stddev = line.split()
    atom1, atom2 = bond.split('-')
    if resname not in residue_bonds:
      residue_bonds[resname] = []
    residue_bonds[resname].append(
        Bond(atom1, atom2, float(length), float(stddev)))
  residue_bonds['UNK'] = []

  # Load bond angles.
  residue_bond_angles = {}
  next(lines_iter)  # Skip empty line.
  next(lines_iter)  # Skip header line.
  for line in lines_iter:
    if line.strip() == '-':
      break
    bond, resname, angle_degree, stddev_degree = line.split()
    atom1, atom2, atom3 = bond.split('-')
    if resname not in residue_bond_angles:
      residue_bond_angles[resname] = []
    residue_bond_angles[resname].append(
        BondAngle(atom1, atom2, atom3,
                  float(angle_degree) / 180. * np.pi,
                  float(stddev_degree) / 180. * np.pi))
  residue_bond_angles['UNK'] = []

  def make_bond_key(atom1_name, atom2_name):
    """Unique key to lookup bonds."""
    return '-'.join(sorted([atom1_name, atom2_name]))

  # Translate bond angles into distances ("virtual bonds").
  residue_virtual_bonds = {}
  for resname, bond_angles in residue_bond_angles.items():
    # Create a fast lookup dict for bond lengths.
    bond_cache = {}
    for b in residue_bonds[resname]:
      bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
    residue_virtual_bonds[resname] = []
    for ba in bond_angles:
      bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
      bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]

      # Compute distance between atom1 and atom3 using the law of cosines
      # c^2 = a^2 + b^2 - 2ab*cos(gamma).
      gamma = ba.angle_rad
      length = np.sqrt(bond1.length**2 + bond2.length**2
                       - 2 * bond1.length * bond2.length * np.cos(gamma))

      # Propagation of uncertainty assuming uncorrelated errors.
      dl_outer = 0.5 / length
      dl_dgamma = (2 * bond1.length * bond2.length * np.sin(gamma)) * dl_outer
      dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
      dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
      stddev = np.sqrt((dl_dgamma * ba.stddev)**2 +
                       (dl_db1 * bond1.stddev)**2 +
                       (dl_db2 * bond2.stddev)**2)
      residue_virtual_bonds[resname].append(
          Bond(ba.atom1_name, ba.atom3name, length, stddev))

  return (residue_bonds,
          residue_virtual_bonds,
          residue_bond_angles)


# Between-residue bond lengths for general bonds (first element) and for Proline
# (second element).
#between_res_bond_length_c_n = [1.329, 1.341]
#between_res_bond_length_stddev_c_n = [0.014, 0.016]

# Between-residue cos_angles.
#between_res_cos_angles_c_n_ca = [-0.5203, 0.0353]  # degrees: 121.352 +- 2.315
#between_res_cos_angles_ca_c_n = [-0.4473, 0.0311]  # degrees: 116.568 +- 1.995

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    "O4'","C4'","C3'","C5'", "O3'", 'C', 'CB', 'O', 'N2', 'N9', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ',"OXT"
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

# A compact atom encodingwith 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_atom14_names = {
    'ADE': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'URA': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'GUA': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ASP': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'CYT': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'GLU': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'GLY': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ILE': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'LEU': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'LYS': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'MET': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'PHE': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'PRO': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'SER': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'TRP': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'TYR': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'VAL': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'UNK': ["O4'", "C4'", "C3'", "O3'","C5'", '',    '',    '',    '',    '',    '',    '',    '',    ''],
}
# pylint: enable=line-too-long
# pylint: enable=bad-whitespace


# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}


def sequence_to_onehot(
    sequence: str,
    mapping: Mapping[str, int],
    map_unknown_to_x: bool = False) -> np.ndarray:
  """Maps the given sequence into a one-hot encoded matrix.

ADErgs:
    sequence: An amino acid sequence.
    mapping: A dictionary mapping amino acids to integers.
    map_unknown_to_x: If True, any amino acid that is not in the mapping will be
      mapped to the unknown amino acid 'X'. If the mapping doesn't contain
      amino acid 'X', an error will be thrown. If False, any amino acid not in
      the mapping will throw an error.

  Returns:
  ADE numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of
    the sequence.

  Raises:
    ValueError: If the mapping doesn't contain values from 0 to
      num_unique_aas - 1 without any gaps.
  """
  num_entries = max(mapping.values()) + 1

  if sorted(set(mapping.values())) != list(range(num_entries)):
    raise ValueError('The mapping must have values from 0 to num_unique_aas-1 '
                     'without any gaps. Got: %s' % sorted(mapping.values()))

  one_hot_arr = np.zeros((len(sequence), num_entries), dtype=int)

  for aa_index, aa_type in enumerate(sequence):
    if map_unknown_to_x:
      if aa_type.isalpha() and aa_type.isupper():
        aa_id = mapping.get(aa_type, mapping['X'])
      else:
        raise ValueError(f'Invalid character in the sequence: {aa_type}')
    else:
      aa_id = mapping[aa_type]
    one_hot_arr[aa_index, aa_id] = 1

  return one_hot_arr


restype_1to3 = {
    'A': 'ADE',
    'R': 'URA',
    'N': 'GUA',
    'D': 'ASP',
    'C': 'CYT',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}


# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Define a restype name for all unknown residues.
unk_restype = 'UNK'

resnames = [restype_1to3[r] for r in restypes] + [unk_restype]
resname_to_idx = {resname: i for i, resname in enumerate(resnames)}


# The mapping here uses hhblits convention, so that B is mapped to D, J and O
# are mapped to X, U is mapped to C, and Z is mapped to E. Other than that the
# remaining 20 amino acids are kept in alphabetical order.
# There are 2 non-amino acid codes, X (representing any amino acid) and
# "-" representing a missing amino acid in an alignment.  The id for these
# codes is put at the end (20 and 21) so that they can easily be ignored if
# desired.
HHBLITS_AA_TO_ID = {
    'A': 0,
    'B': 2,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'J': 20,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 20,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'U': 1,
    'V': 17,
    'W': 18,
    'X': 20,
    'Y': 19,
    'Z': 3,
    '-': 21,
}

# Partial inversion of HHBLITS_AA_TO_ID.
ID_TO_HHBLITS_AA = {
    0: 'A',
    1: 'C',  # Also U.
    2: 'D',  # Also B.
    3: 'E',  # Also Z.
    4: 'F',
    5: 'G',
    6: 'H',
    7: 'I',
    8: 'K',
    9: 'L',
    10: 'M',
    11: 'N',
    12: 'P',
    13: 'Q',
    14: 'R',
    15: 'S',
    16: 'T',
    17: 'V',
    18: 'W',
    19: 'Y',
    20: 'X',  # Includes J and O.
    21: '-',
}

restypes_with_x_and_gap = restypes + ['X', '-']
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = tuple(
    restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[i])
    for i in range(len(restypes_with_x_and_gap)))


def _make_standard_atom_mask() -> np.ndarray:
  """Returns [num_res_types, num_atom_types] mask array."""
  # +1 to account for unknown (all 0s).
  mask = np.zeros([restype_num + 1, atom_type_num], dtype=int)
  for restype, restype_letter in enumerate(restypes):
    restype_name = restype_1to3[restype_letter]
    atom_names = residue_atoms[restype_name]
    for atom_name in atom_names:
      atom_type = atom_order[atom_name]
      mask[restype, atom_type] = 1
  return mask


STANDARD_ATOM_MASK = _make_standard_atom_mask()


# A one hot representation for the first and second atoms defining the axis
# of rotation for each chi-angle in each residue.
def chi_angle_atom(atom_index: int) -> np.ndarray:
  """Define chi-angle rigid groups via one-hot representations."""
  chi_angles_index = {}
  one_hots = []

  for k, v in chi_angles_atoms.items():
    indices = [atom_types.index(s[atom_index]) for s in v]
    indices.extend([-1]*(4-len(indices)))
    chi_angles_index[k] = indices

  for r in restypes:
    res3 = restype_1to3[r]
    one_hot = np.eye(atom_type_num)[chi_angles_index[res3]]
    one_hots.append(one_hot)

  one_hots.append(np.zeros([4, atom_type_num]))  # Add zeros for residue `X`.
  one_hot = np.stack(one_hots, axis=0)
  one_hot = np.transpose(one_hot, [0, 2, 1])

  return one_hot

chi_atom_1_one_hot = chi_angle_atom(1)
chi_atom_2_one_hot = chi_angle_atom(2)

# An array like chi_angles_atoms but using indices rather than names.
chi_angles_atom_indices = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
chi_angles_atom_indices = tree.map_structure(
    lambda atom_name: atom_order[atom_name], chi_angles_atom_indices)
chi_angles_atom_indices = np.array([
    chi_atoms + ([[0, 0, 0, 0]] * (4 - len(chi_atoms)))
    for chi_atoms in chi_angles_atom_indices])

# Mapping from (res_name, atom_name) pairs to the atom's chi group index
# and atom index within that group.
chi_groups_for_atom = collections.defaultdict(list)
for res_name, chi_angle_atoms_for_res in chi_angles_atoms.items():
  for chi_group_i, chi_group in enumerate(chi_angle_atoms_for_res):
    for atom_i, atom in enumerate(chi_group):
      chi_groups_for_atom[(res_name, atom)].append((chi_group_i, atom_i))
chi_groups_for_atom = dict(chi_groups_for_atom)


def _make_rigid_transformation_4x4(ex, ey, translation):
  """Create a rigid 4x4 transformation matrix from two axes and transl."""
  # Normalize ex.
  ex_normalized = ex / np.linalg.norm(ex)

  # make ey perpendicular to ex
  ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
  ey_normalized /= np.linalg.norm(ey_normalized)

  # compute ez as cross product
  eznorm = np.cross(ex_normalized, ey_normalized)
  m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
  m = np.concatenate([m, [[0., 0., 0., 1.]]], axis=0)
  return m


# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group
restype_atom37_to_rigid_group = np.zeros([21, 37], dtype=int)
restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
restype_atom37_rigid_group_positions = np.zeros([21, 37, 3], dtype=np.float32)
restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=int)
restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)


def _make_rigid_group_constants():
  """Fill the arrays above."""
  for restype, restype_letter in enumerate(restypes):
    resname = restype_1to3[restype_letter]
    for atomname, group_idx, atom_position in rigid_group_atom_positions[
        resname]:
      atomtype = atom_order[atomname]
      restype_atom37_to_rigid_group[restype, atomtype] = group_idx
      restype_atom37_mask[restype, atomtype] = 1
      restype_atom37_rigid_group_positions[restype, atomtype, :] = atom_position

      atom14idx = restype_name_to_atom14_names[resname].index(atomname)
      restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
      restype_atom14_mask[restype, atom14idx] = 1
      restype_atom14_rigid_group_positions[restype,
                                           atom14idx, :] = atom_position

  for restype, restype_letter in enumerate(restypes):
    resname = restype_1to3[restype_letter]
    atom_positions = {name: np.array(pos) for name, _, pos
                      in rigid_group_atom_positions[resname]}

    # backbone to backbone is the identity transform
    restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

    # pre-omega-frame to backbone (currently dummy identity matrix)
    restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)

    # phi-frame to backbone
    mat = _make_rigid_transformation_4x4(
        ex=atom_positions["O4'"] - atom_positions["C4'"],
        ey=np.array([1., 0., 0.]),
        translation=atom_positions["O4'"])
    restype_rigid_group_default_frame[restype, 2, :, :] = mat

    # psi-frame to backbone
    mat = _make_rigid_transformation_4x4(
        ex=atom_positions["C3'"] - atom_positions["C4'"],
        ey=atom_positions["C4'"] - atom_positions["O4'"],
        translation=atom_positions["C3'"])
    restype_rigid_group_default_frame[restype, 3, :, :] = mat

    # chi1-frame to backbone
    if chi_angles_mask[restype][0]:
      base_atom_names = chi_angles_atoms[resname][0]
      base_atom_positions = [atom_positions[name] for name in base_atom_names]
      mat = _make_rigid_transformation_4x4(
          ex=base_atom_positions[2] - base_atom_positions[1],
          ey=base_atom_positions[0] - base_atom_positions[1],
          translation=base_atom_positions[2])
      restype_rigid_group_default_frame[restype, 4, :, :] = mat

    # chi2-frame to chi1-frame
    # chi3-frame to chi2-frame
    # chi4-frame to chi3-frame
    # luckily all rotation axes for the next frame start at (0,0,0) of the
    # previous frame
    for chi_idx in range(1, 4):
      if chi_angles_mask[restype][chi_idx]:
        axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]
        axis_end_atom_position = atom_positions[axis_end_atom_name]
        mat = _make_rigid_transformation_4x4(
            ex=axis_end_atom_position,
            ey=np.array([-1., 0., 0.]),
            translation=axis_end_atom_position)
        restype_rigid_group_default_frame[restype, 4 + chi_idx, :, :] = mat


_make_rigid_group_constants()


def make_atom14_dists_bounds(overlap_tolerance=1.5,
                             bond_length_tolerance_factor=15):
  """compute upper and lower bounds for bonds to assess violations."""
  restype_atom14_bond_lower_bound = np.zeros([21, 14, 14], np.float32)
  restype_atom14_bond_upper_bound = np.zeros([21, 14, 14], np.float32)
  restype_atom14_bond_stddev = np.zeros([21, 14, 14], np.float32)
  residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
  for restype, restype_letter in enumerate(restypes):
    resname = restype_1to3[restype_letter]
    atom_list = restype_name_to_atom14_names[resname]

    # create lower and upper bounds for clashes
    for atom1_idx, atom1_name in enumerate(atom_list):
      if not atom1_name:
        continue
      atom1_radius = van_der_waals_radius[atom1_name[0]]
      for atom2_idx, atom2_name in enumerate(atom_list):
        if (not atom2_name) or atom1_idx == atom2_idx:
          continue
        atom2_radius = van_der_waals_radius[atom2_name[0]]
        lower = atom1_radius + atom2_radius - overlap_tolerance
        upper = 1e10
        restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
        restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
        restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
        restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper

    # overwrite lower and upper bounds for bonds and angles
    for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
      atom1_idx = atom_list.index(b.atom1_name)
      atom2_idx = atom_list.index(b.atom2_name)
      lower = b.length - bond_length_tolerance_factor * b.stddev
      upper = b.length + bond_length_tolerance_factor * b.stddev
      restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
      restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
      restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
      restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
      restype_atom14_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
      restype_atom14_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
  return {'lower_bound': restype_atom14_bond_lower_bound,  # shape (21,14,14)
          'upper_bound': restype_atom14_bond_upper_bound,  # shape (21,14,14)
          'stddev': restype_atom14_bond_stddev,  # shape (21,14,14)
         }
