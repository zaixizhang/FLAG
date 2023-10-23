import os
import shutil
import argparse
import random
import torch
import numpy as np
import math
from vina import Vina
from openbabel import pybel
import subprocess
import multiprocessing as mp
from functools import partial
from torch_geometric.data import Batch
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch.utils.data import DataLoader
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures, rdMolDescriptors
from rdkit import RDConfig
from rdkit.Chem.Descriptors import MolLogP, qed
from copy import deepcopy
import tempfile
import AutoDockTools
import contextlib
from torch_scatter import scatter_add, scatter_mean
from rdkit.Geometry import Point3D
from meeko import MoleculePreparation
from meeko import obutils
from models.flag import FLAG
from utils.transforms import *
from utils.datasets import get_dataset
from utils.misc import *
from utils.data import *
from utils.mol_tree import *
from utils.chemutils import *
from utils.dihedral_utils import *
from utils.sascorer import compute_sa_score
from rdkit.Chem import AllChem

_fscores = None

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}

STATUS_RUNNING = 'running'
STATUS_FINISHED = 'finished'
STATUS_FAILED = 'failed'


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf':
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')

    def addH(self, polaronly=False, correctforph=True, PH=7):
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None:
            preparator.write_pdbqt_file(lig_pdbqt)
            return
        else:
            return preparator.write_pdbqt_string()


class PrepProt(object):
    def __init__(self, pdb_file):
        self.prot = pdb_file

    def del_water(self, dry_pdb_file):  # optional
        with open(self.prot) as f:
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')]
            dry_lines = [l for l in lines if not 'HOH' in l]

        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file

    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(['pdb2pqr30', '--ff=AMBER', self.prot, self.prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()


def calculate_vina(number, pro_path, lig_path):
    lig_path = os.path.join(lig_path, str(number)+'.sdf')
    size_factor = 1.2
    buffer = 5.
    # openmm_relax(pro_path)
    # relax_sdf(lig_path)
    mol = Chem.MolFromMolFile(lig_path, sanitize=True)
    pos = mol.GetConformer(0).GetPositions()
    center = np.mean(pos, 0)
    ligand_pdbqt = './data/tmp/' + str(number) + '_lig.pdbqt'
    protein_pqr = './data/tmp/' + str(number) + '_pro.pqr'
    protein_pdbqt = './data/tmp/' + str(number) + '_pro.pdbqt'
    lig = PrepLig(lig_path, 'sdf')
    lig.addH()
    lig.get_pdbqt(ligand_pdbqt)

    prot = PrepProt(pro_path)
    prot.addH(protein_pqr)
    prot.get_pdbqt(protein_pdbqt)

    v = Vina(sf_name='vina', seed=0, verbosity=0)
    v.set_receptor(protein_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    x, y, z = (pos.max(0) - pos.min(0)) * size_factor + buffer
    v.compute_vina_maps(center=center, box_size=[x, y, z])
    energy = v.score()
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])
    energy_minimized = v.optimize()
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
    v.dock(exhaustiveness=64, n_poses=32)
    score = v.energies(n_poses=1)[0][0]
    print('Score after docking : %.3f (kcal/mol)' % score)

    return score


def get_feat(mol):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    atomic_numbers = torch.LongTensor([6, 7, 8, 9, 15, 16, 17])  # C N O F P S Cl
    ptable = Chem.GetPeriodicTable()
    Chem.SanitizeMol(mol)
    feat_mat = np.zeros([mol.GetNumAtoms(), len(ATOM_FAMILIES)], dtype=np.int_)
    for feat in factory.GetFeaturesForMol(mol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
    ligand_element = torch.tensor([ptable.GetAtomicNumber(atom.GetSymbol()) for atom in mol.GetAtoms()])
    element = ligand_element.view(-1, 1) == atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
    return torch.cat([element, torch.tensor(feat_mat)], dim=-1).float()


def find_reference(protein_pos, focal_id):
    # Select three reference protein atoms
    d = torch.norm(protein_pos - protein_pos[focal_id], dim=1)
    reference_idx = torch.topk(d, k=4, largest=False)[1]
    reference_pos = protein_pos[reference_idx]
    return reference_pos, reference_idx


def SetAtomNum(mol, atoms):
    for atom in mol.GetAtoms():
        if atom.GetIdx() in atoms:
            atom.SetAtomMapNum(1)
        else:
            atom.SetAtomMapNum(0)
    return mol


def SetMolPos(mol_list, pos_list):
    new_mol_list = []
    for i in range(len(pos_list)):
        mol = mol_list[i]
        conf = mol.GetConformer(0)
        pos = pos_list[i].cpu().double().numpy()
        if mol.GetNumAtoms() == len(pos):
            for node in range(mol.GetNumAtoms()):
                x, y, z = pos[node]
                conf.SetAtomPosition(node, Point3D(x,y,z))
            try:
                AllChem.UFFOptimizeMolecule(mol)
                new_mol_list.append(mol)
            except:
                new_mol_list.append(mol)
    return new_mol_list


def lipinski(mol):
    count = 0
    if qed(mol) <= 5:
        count += 1
    if Chem.Lipinski.NumHDonors(mol) <= 5:
        count += 1
    if Chem.Lipinski.NumHAcceptors(mol) <= 10:
        count += 1
    if Chem.Descriptors.ExactMolWt(mol) <= 500:
        count += 1
    if Chem.Lipinski.NumRotatableBonds(mol) <= 5:
        count += 1
    return count


def refine_pos(ligand_pos, protein_pos, h_ctx_ligand, h_ctx_protein, model, batch, repeats, protein_batch,
               ligand_batch):
    protein_offsets = torch.cumsum(protein_batch.bincount(), dim=0)
    ligand_offsets = torch.cumsum(ligand_batch.bincount(), dim=0)
    protein_offsets, ligand_offsets = torch.cat([torch.tensor([0]), protein_offsets]), torch.cat([torch.tensor([0]), ligand_offsets])

    sr_ligand_idx, sr_protein_idx = [], []
    sr_ligand_idx0, sr_ligand_idx1 = [], []
    for i in range(len(repeats)):
        alpha_index = batch['alpha_carbon_indicator'][protein_batch == i].nonzero().reshape(-1)
        ligand_atom_index = torch.arange(repeats[i])

        p_idx, q_idx = torch.cartesian_prod(ligand_atom_index, torch.arange(len(alpha_index))).chunk(2, dim=-1)
        p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
        sr_ligand_idx.append(ligand_atom_index[p_idx] + ligand_offsets[i])
        sr_protein_idx.append(alpha_index[q_idx] + protein_offsets[i])

        p_idx, q_idx = torch.cartesian_prod(ligand_atom_index, ligand_atom_index).chunk(2, dim=-1)
        p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
        sr_ligand_idx0.append(ligand_atom_index[p_idx] + ligand_offsets[i])
        sr_ligand_idx1.append(ligand_atom_index[q_idx] + ligand_offsets[i])
    sr_ligand_idx, sr_protein_idx = torch.cat(sr_ligand_idx).long(), torch.cat(sr_protein_idx).long()
    sr_ligand_idx0, sr_ligand_idx1 = torch.cat(sr_ligand_idx0).long(), torch.cat(sr_ligand_idx1).long()

    dist_alpha = torch.norm(ligand_pos[sr_ligand_idx] - protein_pos[sr_protein_idx], dim=1)
    dist_intra = torch.norm(ligand_pos[sr_ligand_idx0] - ligand_pos[sr_ligand_idx1], dim=1)
    input_dir_alpha = ligand_pos[sr_ligand_idx] - protein_pos[sr_protein_idx]
    input_dir_intra = ligand_pos[sr_ligand_idx0] - ligand_pos[sr_ligand_idx1]
    distance_emb1 = model.encoder.distance_expansion(torch.norm(input_dir_alpha, dim=1))
    distance_emb2 = model.encoder.distance_expansion(torch.norm(input_dir_intra, dim=1))
    input1 = torch.cat([h_ctx_ligand[sr_ligand_idx], h_ctx_protein[sr_protein_idx], distance_emb1], dim=-1)[dist_alpha <= 10.0]
    input2 = torch.cat([h_ctx_ligand[sr_ligand_idx0], h_ctx_ligand[sr_ligand_idx1], distance_emb2], dim=-1)[dist_intra <= 10.0]
    # distance cut_off
    norm_dir1 = F.normalize(input_dir_alpha, p=2, dim=1)[dist_alpha <= 10.0]
    norm_dir2 = F.normalize(input_dir_intra, p=2, dim=1)[dist_intra <= 10.0]
    force1 = scatter_mean(model.refine_protein(input1) * norm_dir1, dim=0, index=sr_ligand_idx[dist_alpha <= 10.0], dim_size=ligand_pos.size(0))
    force2 = scatter_mean(model.refine_ligand(input2) * norm_dir2, dim=0, index=sr_ligand_idx0[dist_intra <= 10.0], dim_size=ligand_pos.size(0))
    ligand_pos += force1
    ligand_pos += force2

    ligand_pos = [ligand_pos[ligand_batch==k].float() for k in range(len(repeats))]
    return ligand_pos


def ligand_gen(batch, model, vocab, config, center, device, refinement=False):
    pos_list = []
    feat_list = []
    motif_id = [0 for _ in range(config.sample.batch_size)]
    finished = torch.zeros(config.sample.batch_size).bool()
    for i in range(config.sample.max_steps):
        print(i)
        print(finished)
        if torch.sum(finished) == config.sample.batch_size:
            # mol_list = SetMolPos(mol_list, pos_list)
            return mol_list, pos_list
        if i == 0:
            focal_pred, mask_protein, h_ctx = model(protein_pos=batch['protein_pos'],
                                                    protein_atom_feature=batch['protein_atom_feature'].float(),
                                                    ligand_pos=batch['ligand_context_pos'],
                                                    ligand_atom_feature=batch['ligand_context_feature_full'].float(),
                                                    batch_protein=batch['protein_element_batch'],
                                                    batch_ligand=batch['ligand_context_element_batch'])
            protein_atom_feature = batch['protein_atom_feature'].float()
            focal_protein = focal_pred[mask_protein]
            h_ctx_protein = h_ctx[mask_protein]
            focus_score = torch.sigmoid(focal_protein)
            #can_focus = focus_score > 0.5
            slice_idx = torch.cat([torch.tensor([0]).to(device), torch.cumsum(batch['protein_element_batch'].bincount(), dim=0)])
            focal_id = []
            for j in range(len(slice_idx) - 1):
                focus = focus_score[slice_idx[j]:slice_idx[j + 1]]
                focal_id.append(torch.argmax(focus.reshape(-1).float()).item() + slice_idx[j].item())
            focal_id = torch.tensor(focal_id, device=device)

            h_ctx_focal = h_ctx_protein[focal_id]
            current_wid = torch.tensor([vocab.size()] * config.sample.batch_size, device=device)
            next_motif_wid, motif_prob = model.forward_motif(h_ctx_focal, current_wid, torch.arange(config.sample.batch_size, device=device).to(device))
            mol_list = [Chem.MolFromSmiles(vocab.get_smiles(id)) for id in next_motif_wid]

            for j in range(config.sample.batch_size):
                AllChem.EmbedMolecule(mol_list[j])
                AllChem.UFFOptimizeMolecule(mol_list[j])
                ligand_pos, ligand_feat = torch.tensor(mol_list[j].GetConformer().GetPositions(), device=device), get_feat(mol_list[j]).to(device)
                feat_list.append(ligand_feat)
                # set the initial positions with distance matrix
                reference_pos, reference_idx = find_reference(batch['protein_pos'][slice_idx[j]:slice_idx[j + 1]], focal_id[j] - slice_idx[j])

                p_idx, l_idx = torch.cartesian_prod(torch.arange(4), torch.arange(len(ligand_pos))).chunk(2, dim=-1)
                p_idx = p_idx.squeeze(-1).to(device)
                l_idx = l_idx.squeeze(-1).to(device)
                d_m = model.dist_mlp(torch.cat([protein_atom_feature[reference_idx[p_idx]], ligand_feat[l_idx]], dim=-1)).reshape(4,len(ligand_pos))

                d_m = d_m ** 2
                p_d, l_d = self_square_dist(reference_pos), self_square_dist(ligand_pos)
                D = torch.cat([torch.cat([p_d, d_m], dim=1), torch.cat([d_m.permute(1, 0), l_d], dim=1)])
                coordinate = eig_coord_from_dist(D)
                new_pos, _, _ = kabsch_torch(coordinate[:len(reference_pos)], reference_pos,
                                             coordinate[len(reference_pos):])
                # new_pos += (center*0.8+torch.mean(reference_pos, dim=0)*0.2) - torch.mean(new_pos, dim=0)
                new_pos += (center - torch.mean(new_pos, dim=0)) * .8
                pos_list.append(new_pos)

            atom_to_motif = [{} for _ in range(config.sample.batch_size)]
            motif_to_atoms = [{} for _ in range(config.sample.batch_size)]
            motif_wid = [{} for _ in range(config.sample.batch_size)]
            for j in range(config.sample.batch_size):
                for k in range(mol_list[j].GetNumAtoms()):
                    atom_to_motif[j][k] = 0
            for j in range(config.sample.batch_size):
                motif_to_atoms[j][0] = list(np.arange(mol_list[j].GetNumAtoms()))
                motif_wid[j][0] = next_motif_wid[j].item()
        else:
            repeats = torch.tensor([len(pos) for pos in pos_list], device=device)
            ligand_batch = torch.repeat_interleave(torch.arange(config.sample.batch_size, device=device), repeats)
            focal_pred, mask_protein, h_ctx = model(protein_pos=batch['protein_pos'].float(),
                                                    protein_atom_feature=batch['protein_atom_feature'].float(),
                                                    ligand_pos=torch.cat(pos_list, dim=0).float(),
                                                    ligand_atom_feature=torch.cat(feat_list, dim=0).float(),
                                                    batch_protein=batch['protein_element_batch'],
                                                    batch_ligand=ligand_batch)
            # structure refinement
            if refinement:
                pos_list = refine_pos(torch.cat(pos_list, dim=0).float(), batch['protein_pos'].float(),
                                      h_ctx[~mask_protein], h_ctx[mask_protein], model, batch, repeats.tolist(),
                                      batch['protein_element_batch'], ligand_batch)

            focal_ligand = focal_pred[~mask_protein]
            h_ctx_ligand = h_ctx[~mask_protein]
            focus_score = torch.sigmoid(focal_ligand)
            can_focus = focus_score > 0.
            slice_idx = torch.cat([torch.tensor([0], device=device), torch.cumsum(repeats, dim=0)])

            current_atoms_batch, current_atoms = [], []
            for j in range(len(slice_idx) - 1):
                focus = focus_score[slice_idx[j]:slice_idx[j + 1]]
                if torch.sum(can_focus[slice_idx[j]:slice_idx[j + 1]]) > 0 and ~finished[j]:
                    sample_focal_atom = torch.multinomial(focus.reshape(-1).float(), 1)
                    focal_motif = atom_to_motif[j][sample_focal_atom.item()]
                    motif_id[j] = focal_motif
                else:
                    finished[j] = True

                current_atoms.extend((np.array(motif_to_atoms[j][motif_id[j]]) + slice_idx[j].item()).tolist())
                current_atoms_batch.extend([j] * len(motif_to_atoms[j][motif_id[j]]))
                mol_list[j] = SetAtomNum(mol_list[j], motif_to_atoms[j][motif_id[j]])
            # second step: next motif prediction
            current_wid = [motif_wid[j][motif_id[j]] for j in range(len(mol_list))]
            next_motif_wid, motif_prob = model.forward_motif(h_ctx_ligand[torch.tensor(current_atoms)],
                                                 torch.tensor(current_wid).to(device),
                                                 torch.tensor(current_atoms_batch).to(device))

            # assemble
            next_motif_smiles = [vocab.get_smiles(id) for id in next_motif_wid]
            new_mol_list, new_atoms, one_atom_attach, intersection, attach_fail = model.forward_attach(mol_list, next_motif_smiles, device)

            for j in range(len(mol_list)):
                if ~finished[j] and ~attach_fail[j]:
                    # num_new_atoms
                    mol_list[j] = new_mol_list[j]
            rotatable = torch.logical_and(torch.tensor(current_atoms_batch).bincount() == 2, torch.tensor(one_atom_attach))
            rotatable = torch.logical_and(rotatable, ~torch.tensor(attach_fail))
            rotatable = torch.logical_and(rotatable, ~finished).to(device)
            # update motif2atoms and atom2motif
            for j in range(len(mol_list)):
                if attach_fail[j] or finished[j]:
                    continue
                motif_to_atoms[j][i] = new_atoms[j]
                motif_wid[j][i] = next_motif_wid[j]
                for k in new_atoms[j]:
                    atom_to_motif[j][k] = i
                    '''
                    if k in atom_to_motif[j]:
                        continue
                    else:
                        atom_to_motif[j][k] = i'''

            # generate initial positions
            for j in range(len(mol_list)):
                if attach_fail[j] or finished[j]:
                    continue
                mol = mol_list[j]
                anchor = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 1]
                # positions = mol.GetConformer().GetPositions()
                anchor_pos = deepcopy(pos_list[j][anchor]).to(device)
                Chem.SanitizeMol(mol)
                AllChem.EmbedMolecule(mol, useRandomCoords=True)
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except:
                    print('UFF error')
                anchor_pos_new = mol.GetConformer(0).GetPositions()[anchor]
                new_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]
                '''
                R, T = kabsch(np.matrix(anchor_pos), np.matrix(anchor_pos_new))
                new_pos = R * np.matrix(mol.GetConformer().GetPositions()[new_idx]).T + np.tile(T, (1, len(new_idx)))
                new_pos = np.array(new_pos.T)'''
                new_pos = mol.GetConformer().GetPositions()[new_idx]
                new_pos, _, _ = kabsch_torch(torch.tensor(anchor_pos_new, device=device), anchor_pos, torch.tensor(new_pos, device=device))

                conf = mol.GetConformer()
                # update curated parameters
                pos_list[j] = torch.cat([pos_list[j], new_pos])
                feat_list[j] = get_feat(mol_list[j]).to(device)
                for node in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(node, np.array(pos_list[j][node].cpu()))
                assert mol.GetNumAtoms() == len(pos_list[j])

            # predict alpha and rotate (only change the position)
            if torch.sum(rotatable) > 0 and i >= 2:
                repeats = torch.tensor([len(pos) for pos in pos_list])
                ligand_batch = torch.repeat_interleave(torch.arange(len(pos_list)), repeats).to(device)
                slice_idx = torch.cat([torch.tensor([0]), torch.cumsum(repeats, dim=0)])
                xy_index = [(np.array(motif_to_atoms[j][motif_id[j]]) + slice_idx[j].item()).tolist() for j in range(len(slice_idx) - 1) if rotatable[j]]

                alpha = model.forward_alpha(protein_pos=batch['protein_pos'].float(),
                                            protein_atom_feature=batch['protein_atom_feature'].float(),
                                            ligand_pos=torch.cat(pos_list, dim=0).float(),
                                            ligand_atom_feature=torch.cat(feat_list, dim=0).float(),
                                            batch_protein=batch['protein_element_batch'],
                                            batch_ligand=ligand_batch, xy_index=torch.tensor(xy_index, device=device),
                                            rotatable=rotatable)

                rotatable_id = [id for id in range(len(mol_list)) if rotatable[id]]
                xy_index = [motif_to_atoms[j][motif_id[j]] for j in range(len(slice_idx) - 1) if rotatable[j]]
                x_index = [intersection[j] for j in range(len(slice_idx) - 1) if rotatable[j]]
                y_index = [(set(xy_index[k]) - set(x_index[k])).pop() for k in range(len(x_index))]

                for j in range(len(alpha)):
                    mol = mol_list[rotatable_id[j]]
                    new_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]
                    positions = deepcopy(pos_list[rotatable_id[j]])

                    xn_pos = positions[new_idx].float()
                    dir=(positions[x_index[j]] - positions[y_index[j]]).reshape(-1)
                    ref=positions[x_index[j]].reshape(-1)
                    xn_pos = rand_rotate(dir.to(device), ref.to(device), xn_pos.to(device), alpha[j], device=device)
                    if xn_pos.shape[0] > 0:
                        pos_list[rotatable_id[j]][-len(xn_pos):] = xn_pos
                    conf = mol.GetConformer()
                    for node in range(mol.GetNumAtoms()):
                        conf.SetAtomPosition(node, np.array(pos_list[rotatable_id[j]][node].cpu()))
                    assert mol.GetNumAtoms() == len(pos_list[rotatable_id[j]])

    return mol_list, pos_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample.yml')
    parser.add_argument('-i', '--data_id', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--vocab_path', type=str, default='vocab.txt')
    parser.add_argument('--num_workers', type=int, default=64)
    args = parser.parse_args()

    # Load vocab
    vocab = []
    for line in open(args.vocab_path):
        p, _, _ = line.partition(':')
        vocab.append(p)
    vocab = Vocab(vocab)

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.sample.seed)

    # Logging
    log_dir = get_new_log_dir(args.outdir, prefix='%s-%d' % (config_name, args.data_id))
    logger = get_logger('sample', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # Data
    logger.info('Loading data...')
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    masking = LigandMaskAll(vocab)
    transform = Compose([
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        FeaturizeLigandBond(),
        masking,
    ])
    dataset, subsets = get_dataset(
        config=config.dataset,
        transform=transform,
    )
    testset = subsets['test']
    data = testset[args.data_id]
    center = data['ligand_center'].to(args.device)
    test_set = [data for _ in range(config.sample.num_samples)]

    with open(os.path.join(log_dir, 'pocket_info.txt'), 'a') as f:
        f.write(data['protein_filename'] + '\n')

    # Model (Main)
    logger.info('Loading main model...')
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    model = FLAG(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        vocab=vocab,
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    # my code goes here
    sample_loader = DataLoader(test_set, batch_size=config.sample.batch_size,
                               shuffle=False, num_workers=config.sample.num_workers,
                               collate_fn=collate_mols)
    data_list = []
    try:
        with torch.no_grad():
            model.eval()
            number = 0
            number_list = []
            for batch in tqdm(sample_loader):
                for key in batch:
                    batch[key] = batch[key].to(args.device)
                gen_data, pos_list = ligand_gen(batch, model, vocab, config, center, args.device)
                SetMolPos(gen_data, pos_list)
                for mol in gen_data:
                    try:
                        AllChem.UFFOptimizeMolecule(mol)
                    except:
                        print('UFF error')
                data_list.extend(gen_data)
                with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
                    for _, mol in enumerate(gen_data):
                        number+=1
                        if mol.GetNumAtoms() < 12 or MolLogP(mol) < 0.60:
                            continue
                        smiles_f.write(Chem.MolToSmiles(mol) + '\n')
                        writer = Chem.SDWriter(os.path.join(log_dir, '%d.sdf' % number))
                        # writer.SetKekulize(False)
                        writer.write(mol, confId=0)
                        writer.close()
                        number_list.append(number)

                # Calculate metrics
                print([Chem.MolToSmiles(mol) for mol in data_list])
                smiles = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in data_list]
                qed_list = [qed(mol) for mol in smiles if mol.GetNumAtoms() >= 8]
                logp_list = [MolLogP(mol) for mol in smiles]
                sa_list = [compute_sa_score(mol) for mol in smiles]
                Lip_list = [lipinski(mol) for mol in smiles]
                print('QED %.6f | LogP %.6f | SA %.6f | Lipinski %.6f \n' % (np.average(qed_list), np.average(logp_list), np.average(sa_list), np.average(Lip_list)))

    except KeyboardInterrupt:
        logger.info('Terminated. Generated molecules will be saved.')
        with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
            for i, mol in enumerate(data_list):
                if mol.GetNumAtoms() < 12 or MolLogP(mol) < 0.60:
                    continue
                smiles_f.write(Chem.MolToSmiles(mol) + '\n')
                writer = Chem.SDWriter(os.path.join(log_dir, '%d.sdf' % i))
                # writer.SetKekulize(False)
                writer.write(mol, confId=0)
                writer.close()

    pool = mp.Pool(args.num_workers)
    vina_list = []
    pro_path = './data/pdbbind_pocket10/' + os.path.join(data['pdbid'], data['pdbid']+'_pocket.pdb')
    for vina_score in tqdm(pool.imap_unordered(partial(calculate_vina, pro_path=pro_path, lig_path=log_dir), number_list), total=len(number_list)):
        if vina_score != None:
            vina_list.append(vina_score)
    pool.close()
    print('Vina: ', np.average(vina_list))
