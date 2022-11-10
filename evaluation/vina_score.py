from vina import Vina
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit import Chem
import numpy as np
import os

for i in range(100):
    path = './' + str(i) + '.sdf'
    if os.path.exists(path):
        print(path)
        v = Vina(sf_name='vina')
        v.set_receptor('.pdbqt')
        v.set_ligand_from_file(str(i)+'.pdbqt')

        # Calculate the docking center
        mol = Chem.MolFromMolFile(path, sanitize=True)
        mol = Chem.AddHs(mol, addCoords=True)
        UFFOptimizeMolecule(mol)
        pos = mol.GetConformer(0).GetPositions()
        center = np.mean(pos, 0)

        v.compute_vina_maps(center=center, box_size=[20, 20, 20])

        # Score the current pose
        energy = v.score()
        print('Score before minimization: %.3f (kcal/mol)' % energy[0])

        # Minimized locally the current pose
        energy_minimized = v.optimize()
        print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
        v.write_pose('ligand_minimized.pdbqt', overwrite=True)

        # Dock the ligand
        v.dock(exhaustiveness=64, n_poses=30)
        v.write_poses('out.pdbqt', n_poses=5, overwrite=True)
