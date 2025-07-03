from ase.build import bulk
import numpy as np

atom_name = "Si"
crystal = "diamond"
a = 5.431
replication = (2, 2, 2)

output_path = "data/diamond_structure_2.xyz"

#結晶構造の作成
atoms = bulk(name=atom_name, crystalstructure=crystal, a=a, cubic=True)
atoms = atoms * replication

#位置を少しずらす (±a/20までの乱数)
random_displacements = np.random.uniform(
    low = - a / 20, 
    high = a / 20, 
    size = (len(atoms), 3)
)

atoms.positions += random_displacements

atoms.write(output_path, format="extxyz")