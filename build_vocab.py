import sys
sys.path.append("..")
import rdkit
import rdkit.Chem as Chem
import copy
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
import random
from utils.chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, get_clique_mol_simple
from utils.mol_tree import *
from collections import defaultdict
import os


seed = 2023
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

vocab = {}
cnt = 0
rot = 0

index = torch.load('/n/holyscratch01/mzitnik_lab/zaixizhang/pdbbind_pocket10/index.pt')
for i, pdbid in enumerate(tqdm(index)):
    try:
        path = '/n/holyscratch01/mzitnik_lab/zaixizhang/pdbbind_pocket10/'
        ligand_path = os.path.join(path, os.path.join(pdbid, pdbid+'_ligand.sdf'))
        mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
        moltree = MolTree(mol)
        cnt += 1
        if moltree.num_rotatable_bond > 0:
            rot += 1

    except:
        continue

    for c in moltree.nodes:
        smile_cluster = c.smiles
        if smile_cluster not in vocab:
            vocab[smile_cluster] = 1
        else:
            vocab[smile_cluster] += 1

vocab = dict(sorted(vocab.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
filename = open('./vocab.txt', 'w')
for k, v in vocab.items():
    filename.write(k + ':' + str(v))
    filename.write('\n')
filename.close()

# number of molecules and vocab
print('Size of the motif vocab:', len(vocab))
print('Total number of molecules', cnt)
print('Percent of molecules with rotatable bonds:', rot / cnt)
