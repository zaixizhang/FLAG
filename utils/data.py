import copy
import torch
import numpy as np
from torch_geometric.data import Data, Batch
# from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

FOLLOW_BATCH = ['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']


class ProteinLigandData(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                if key == 'moltree':
                    instance['moltree'] = item
                else:
                    instance['ligand_' + key] = item

        # instance['ligand_nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]}
        return instance


def batch_from_data_list(data_list):
    return Batch.from_data_list(data_list, follow_batch=['ligand_element', 'protein_element'])


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def collate_mols(mol_dicts):
    data_batch = {}
    batch_size = len(mol_dicts)
    for key in ['protein_pos', 'protein_atom_feature', 'ligand_context_pos', 'ligand_context_feature_full',
                'ligand_frontier', 'num_atoms', 'next_wid', 'current_wid', 'current_atoms', 'cand_labels',
                'ligand_pos_torsion', 'ligand_feature_torsion', 'true_sin', 'true_cos', 'true_three_hop',
                'dihedral_mask', 'protein_contact', 'true_dm']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
    # unsqueeze dim0
    for key in ['xn_pos', 'yn_pos', 'ligand_torsion_xy_index', 'y_pos']:
        cat_list = [mol_dict[key].unsqueeze(0) for mol_dict in mol_dicts if len(mol_dict[key]) > 0]
        if len(cat_list) > 0:
            data_batch[key] = torch.cat(cat_list, dim=0)
        else:
            data_batch[key] = torch.tensor([])
    # follow batch
    for key in ['protein_element', 'ligand_context_element', 'current_atoms']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)
    for key in ['ligand_element_torsion']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts if len(mol_dict[key]) > 0])
        if len(repeats) > 0:
            data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)
        else:
            data_batch[key + '_batch'] = torch.tensor([])
    # distance matrix prediction
    p_idx, q_idx = torch.cartesian_prod(torch.arange(4), torch.arange(2)).chunk(2, dim=-1)
    p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
    protein_offsets = torch.cumsum(data_batch['protein_element_batch'].bincount(), dim=0)
    ligand_offsets = torch.cumsum(data_batch['ligand_context_element_batch'].bincount(), dim=0)
    protein_offsets, ligand_offsets = torch.cat([torch.tensor([0]), protein_offsets]), torch.cat([torch.tensor([0]), ligand_offsets])
    ligand_idx, protein_idx = [], []
    for i, mol_dict in enumerate(mol_dicts):
        if len(mol_dict['true_dm']) > 0:
            protein_idx.append(mol_dict['dm_protein_idx'][p_idx] + protein_offsets[i])
            ligand_idx.append(mol_dict['dm_ligand_idx'][q_idx] + ligand_offsets[i])
    if len(ligand_idx) > 0:
        data_batch['dm_ligand_idx'], data_batch['dm_protein_idx'] = torch.cat(ligand_idx), torch.cat(protein_idx)

    # structure refinement (alpha carbon - ligand atom)
    sr_ligand_idx, sr_protein_idx = [], []
    for i, mol_dict in enumerate(mol_dicts):
        if len(mol_dict['true_dm']) > 0:
            ligand_atom_index = torch.arange(len(mol_dict['ligand_context_pos']))
            p_idx, q_idx = torch.cartesian_prod(torch.arange(len(mol_dict['ligand_context_pos'])), torch.arange(len(mol_dict['protein_alpha_carbon_index']))).chunk(2, dim=-1)
            p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
            sr_ligand_idx.append(ligand_atom_index[p_idx] + ligand_offsets[i])
            sr_protein_idx.append(mol_dict['protein_alpha_carbon_index'][q_idx] + protein_offsets[i])
    if len(ligand_idx) > 0:
        data_batch['sr_ligand_idx'], data_batch['sr_protein_idx'] = torch.cat(sr_ligand_idx).long(), torch.cat(sr_protein_idx).long()

    # structure refinement (ligand atom - ligand atom)
    sr_ligand_idx0, sr_ligand_idx1 = [], []
    for i, mol_dict in enumerate(mol_dicts):
        if len(mol_dict['true_dm']) > 0:
            ligand_atom_index = torch.arange(len(mol_dict['ligand_context_pos']))
            p_idx, q_idx = torch.cartesian_prod(torch.arange(len(mol_dict['ligand_context_pos'])), torch.arange(len(mol_dict['ligand_context_pos']))).chunk(2, dim=-1)
            p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
            sr_ligand_idx0.append(ligand_atom_index[p_idx] + ligand_offsets[i])
            sr_ligand_idx1.append(ligand_atom_index[q_idx] + ligand_offsets[i])
    if len(ligand_idx) > 0:
        data_batch['sr_ligand_idx0'], data_batch['sr_ligand_idx1'] = torch.cat(sr_ligand_idx0).long(), torch.cat(sr_ligand_idx1).long()
    # index
    if len(data_batch['y_pos']) > 0:
        repeats = torch.tensor([len(mol_dict['ligand_element_torsion']) for mol_dict in mol_dicts if len(mol_dict['ligand_element_torsion']) > 0])
        offsets = torch.cat([torch.tensor([0]), torch.cumsum(repeats, dim=0)])[:-1]
        data_batch['ligand_torsion_xy_index'] += offsets.unsqueeze(1)

    offsets1 = torch.cat([torch.tensor([0]), torch.cumsum(data_batch['num_atoms'], dim=0)])[:-1]
    data_batch['current_atoms'] += torch.repeat_interleave(offsets1, data_batch['current_atoms_batch'].bincount())
    # cand mols: torch geometric Data
    cand_mol_list = []
    for data in mol_dicts:
        if len(data['cand_labels']) > 0:
            cand_mol_list.extend(data['cand_mols'])
    if len(cand_mol_list) > 0:
        data_batch['cand_mols'] = Batch.from_data_list(cand_mol_list)
    return data_batch

