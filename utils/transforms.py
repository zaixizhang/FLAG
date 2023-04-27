import sys
sys.path.append("..")
import copy
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch_geometric.transforms import Compose
from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

from .data import ProteinLigandData
from .protein_ligand import ATOM_FAMILIES
from .chemutils import enumerate_assemble, list_filter, rand_rotate
from .dihedral_utils import batch_dihedrals

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class RefineData(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # delete H atom of pocket
        protein_element = data.protein_element
        is_H_protein = (protein_element == 1)
        if torch.sum(is_H_protein) > 0:
            not_H_protein = ~is_H_protein
            data.protein_atom_name = list(compress(data.protein_atom_name, not_H_protein))
            data.protein_atom_to_aa_type = data.protein_atom_to_aa_type[not_H_protein]
            data.protein_element = data.protein_element[not_H_protein]
            data.protein_is_backbone = data.protein_is_backbone[not_H_protein]
            data.protein_pos = data.protein_pos[not_H_protein]
        # delete H atom of ligand
        ligand_element = data.ligand_element
        is_H_ligand = (ligand_element == 1)
        if torch.sum(is_H_ligand) > 0:
            not_H_ligand = ~is_H_ligand
            data.ligand_atom_feature = data.ligand_atom_feature[not_H_ligand]
            data.ligand_element = data.ligand_element[not_H_ligand]
            data.ligand_pos = data.ligand_pos[not_H_ligand]
            # nbh
            index_atom_H = torch.nonzero(is_H_ligand)[:, 0]
            index_changer = -np.ones(len(not_H_ligand), dtype=np.int64)
            index_changer[not_H_ligand] = np.arange(torch.sum(not_H_ligand))
            new_nbh_list = [value for ind_this, value in zip(not_H_ligand, data.ligand_nbh_list.values()) if ind_this]
            data.ligand_nbh_list = {i: [index_changer[node] for node in neigh if node not in index_atom_H] for i, neigh
                                    in enumerate(new_nbh_list)}
            # bond
            ind_bond_with_H = np.array([(bond_i in index_atom_H) | (bond_j in index_atom_H) for bond_i, bond_j in
                                        zip(*data.ligand_bond_index)])
            ind_bond_without_H = ~ind_bond_with_H
            old_ligand_bond_index = data.ligand_bond_index[:, ind_bond_without_H]
            data.ligand_bond_index = torch.tensor(index_changer)[old_ligand_bond_index]
            data.ligand_bond_type = data.ligand_bond_type[ind_bond_without_H]

        return data


class FocalBuilder(object):
    def __init__(self, close_threshold=0.8, max_bond_length=2.4):
        self.close_threshold = close_threshold
        self.max_bond_length = max_bond_length
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        # ligand_context_pos = data.ligand_context_pos
        # ligand_pos = data.ligand_pos
        ligand_masked_pos = data.ligand_masked_pos
        protein_pos = data.protein_pos
        context_idx = data.context_idx
        masked_idx = data.masked_idx
        old_bond_index = data.ligand_bond_index
        # old_bond_types = data.ligand_bond_type  # type: 0, 1, 2
        has_unmask_atoms = context_idx.nelement() > 0
        if has_unmask_atoms:
            # # get bridge bond index (mask-context bond)
            ind_edge_index_candidate = [
                (context_node in context_idx) and (mask_node in masked_idx)
                for mask_node, context_node in zip(*old_bond_index)
            ]  # the mask-context order is right
            bridge_bond_index = old_bond_index[:, ind_edge_index_candidate]
            # candidate_bond_types = old_bond_types[idx_edge_index_candidate]
            idx_generated_in_whole_ligand = bridge_bond_index[0]
            idx_focal_in_whole_ligand = bridge_bond_index[1]

            index_changer_masked = torch.zeros(masked_idx.max() + 1, dtype=torch.int64)
            index_changer_masked[masked_idx] = torch.arange(len(masked_idx))
            idx_generated_in_ligand_masked = index_changer_masked[idx_generated_in_whole_ligand]
            pos_generate = ligand_masked_pos[idx_generated_in_ligand_masked]

            data.idx_generated_in_ligand_masked = idx_generated_in_ligand_masked
            data.pos_generate = pos_generate

            index_changer_context = torch.zeros(context_idx.max() + 1, dtype=torch.int64)
            index_changer_context[context_idx] = torch.arange(len(context_idx))
            idx_focal_in_ligand_context = index_changer_context[idx_focal_in_whole_ligand]
            idx_focal_in_compose = idx_focal_in_ligand_context  # if ligand_context was not before protein in the compose, this was not correct
            data.idx_focal_in_compose = idx_focal_in_compose

            data.idx_protein_all_mask = torch.empty(0, dtype=torch.long)  # no use if has context
            data.y_protein_frontier = torch.empty(0, dtype=torch.bool)  # no use if has context

        else:  # # the initial atom. surface atoms between ligand and protein
            assign_index = radius(x=ligand_masked_pos, y=protein_pos, r=4., num_workers=16)
            if assign_index.size(1) == 0:
                dist = torch.norm(data.protein_pos.unsqueeze(1) - data.ligand_masked_pos.unsqueeze(0), p=2, dim=-1)
                assign_index = torch.nonzero(dist <= torch.min(dist) + 1e-5)[0:1].transpose(0, 1)
            idx_focal_in_protein = assign_index[0]
            data.idx_focal_in_compose = idx_focal_in_protein  # no ligand context, so all composes are protein atoms
            data.pos_generate = ligand_masked_pos[assign_index[1]]
            data.idx_generated_in_ligand_masked = torch.unique(assign_index[1])  # for real of the contractive transform

            data.idx_protein_all_mask = data.idx_protein_in_compose  # for input of initial frontier prediction
            y_protein_frontier = torch.zeros_like(data.idx_protein_all_mask,
                                                  dtype=torch.bool)  # for label of initial frontier prediction
            y_protein_frontier[torch.unique(idx_focal_in_protein)] = True
            data.y_protein_frontier = y_protein_frontier

        # generate not positions: around pos_focal ( with `max_bond_length` distance) but not close to true generated within `close_threshold`
        # pos_focal = ligand_context_pos[idx_focal_in_ligand_context]
        # pos_notgenerate = pos_focal + torch.randn_like(pos_focal) * self.max_bond_length  / 2.4
        # dist = torch.norm(pos_generate - pos_notgenerate, p=2, dim=-1)
        # ind_close = (dist < self.close_threshold)
        # while ind_close.any():
        #     new_pos_notgenerate = pos_focal[ind_close] + torch.randn_like(pos_focal[ind_close]) * self.max_bond_length  / 2.3
        #     dist[ind_close] = torch.norm(pos_generate[ind_close] - new_pos_notgenerate, p=2, dim=-1)
        #     pos_notgenerate[ind_close] = new_pos_notgenerate
        #     ind_close = (dist < self.close_threshold)
        # data.pos_notgenerate = pos_notgenerate

        return data


class AtomComposer(object):

    def __init__(self, protein_dim, ligand_dim, knn):
        super().__init__()
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.knn = knn  # knn of compose atoms

    def __call__(self, data: ProteinLigandData):
        # fetch ligand context and protein from data
        ligand_context_pos = data['ligand_context_pos']
        ligand_context_feature_full = data['ligand_context_feature_full']
        protein_pos = data['protein_pos']
        protein_atom_feature = data['protein_atom_feature']
        len_ligand_ctx = len(ligand_context_pos)
        len_protein = len(protein_pos)

        # compose ligand context and protein. save idx of them in compose
        data['compose_pos'] = torch.cat([ligand_context_pos, protein_pos], dim=0)
        len_compose = len_ligand_ctx + len_protein
        ligand_context_feature_full_expand = torch.cat([
            ligand_context_feature_full,
            torch.zeros([len_ligand_ctx, self.protein_dim - self.ligand_dim], dtype=torch.long)
        ], dim=1)
        data['compose_feature'] = torch.cat([ligand_context_feature_full_expand, protein_atom_feature], dim=0)
        data['idx_ligand_ctx_in_compose'] = torch.arange(len_ligand_ctx, dtype=torch.long)  # can be delete
        data['idx_protein_in_compose'] = torch.arange(len_protein, dtype=torch.long) + len_ligand_ctx  # can be delete

        # build knn graph and bond type
        data = self.get_knn_graph(data, self.knn, len_ligand_ctx, len_compose, num_workers=16)
        return data

    @staticmethod
    def get_knn_graph(data: ProteinLigandData, knn, len_ligand_ctx, len_compose, num_workers=1, ):
        data['compose_knn_edge_index'] = knn_graph(data['compose_pos'], knn, flow='target_to_source', num_workers=num_workers)

        id_compose_edge = data['compose_knn_edge_index'][0,
                          :len_ligand_ctx * knn] * len_compose + data['compose_knn_edge_index'][1, :len_ligand_ctx * knn]
        id_ligand_ctx_edge = data['ligand_context_bond_index'][0] * len_compose + data['ligand_context_bond_index'][1]
        idx_edge = [torch.nonzero(id_compose_edge == id_) for id_ in id_ligand_ctx_edge]
        idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
        data['compose_knn_edge_type'] = torch.zeros(len(data['compose_knn_edge_index'][0]),
                                                 dtype=torch.long)  # for encoder edge embedding
        data['compose_knn_edge_type'][idx_edge[idx_edge >= 0]] = data['ligand_context_bond_type'][idx_edge >= 0]
        data['compose_knn_edge_feature'] = torch.cat([
            torch.ones([len(data['compose_knn_edge_index'][0]), 1], dtype=torch.long),
            torch.zeros([len(data['compose_knn_edge_index'][0]), 3], dtype=torch.long),
        ], dim=-1)
        data['compose_knn_edge_feature'][idx_edge[idx_edge >= 0]] = F.one_hot(data['ligand_context_bond_type'][idx_edge >= 0],
                                                                           num_classes=4)  # 0 (1,2,3)-onehot
        return data


class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinLigandData):
        element = data['protein_element'].view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data['protein_atom_to_aa_type'], num_classes=self.max_num_aa)
        is_backbone = data['protein_is_backbone'].view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data['protein_atom_feature'] = x
        return data


class FeaturizeLigandAtom(object):

    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17])  # H C N O F P S Cl
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 9, 15, 16, 17])  # C N O F P S Cl

    @property
    def num_properties(self):
        return len(ATOM_FAMILIES)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + len(ATOM_FAMILIES)

    def __call__(self, data: ProteinLigandData):
        element = data['ligand_element'].view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        x = torch.cat([element, data['ligand_atom_feature']], dim=-1)
        data['ligand_atom_feature_full'] = x
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        data['ligand_bond_feature'] = F.one_hot(data['ligand_bond_type'] - 1, num_classes=3)  # (1,2,3) to (0,1,2)-onehot

        neighbor_dict = {}
        # used in rotation angle prediction
        mol = data['moltree'].mol
        for i, atom in enumerate(mol.GetAtoms()):
            neighbor_dict[i] = [n.GetIdx() for n in atom.GetNeighbors()]
        data['ligand_neighbors'] = neighbor_dict
        return data


class LigandCountNeighbors(object):

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data['ligand_num_neighbors'] = self.count_neighbors(
            data['ligand_bond_index'],
            symmetry=True,
            num_nodes=data['ligand_element'].size(0),
        )
        data['ligand_atom_valence'] = self.count_neighbors(
            data['ligand_bond_index'],
            symmetry=True,
            valence=data['ligand_bond_type'],
            num_nodes=data['ligand_element'].size(0),
        )
        return data


class LigandRandomMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked

    def __call__(self, data: ProteinLigandData):
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)

        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        idx = torch.LongTensor(idx)
        masked_idx = idx[:num_masked]
        context_idx = idx[num_masked:]

        data.ligand_masked_element = data.ligand_element[masked_idx]
        data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]  # For Prediction
        data.ligand_masked_pos = data.ligand_pos[masked_idx]

        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]  # For Input
        data.ligand_context_pos = data.ligand_pos[context_idx]

        data.ligand_context_bond_index, data.ligand_context_bond_feature = subgraph(
            context_idx,
            data.ligand_bond_index,
            edge_attr=data.ligand_bond_feature,
            relabel_nodes=True,
        )
        data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes=context_idx.size(0),
        )

        # print(context_idx)
        # print(data.ligand_context_bond_index)

        # mask = torch.logical_and(
        #     (data.ligand_bond_index[0].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        #     (data.ligand_bond_index[1].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
        # )
        # print(data.ligand_bond_index[:, mask])

        # print(data.ligand_context_num_neighbors)
        # print(data.ligand_num_neighbors[context_idx])

        data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

        data._mask = 'random'

        return data


class LigandBFSMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, vocab=None):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.vocab = vocab
        self.vocab_size = vocab.size()

    @staticmethod
    def get_bfs_perm_motif(moltree, vocab):
        for i, node in enumerate(moltree.nodes):
            node.nid = i
            node.wid = vocab.get_index(node.smiles)
        # num_motifs = len(moltree.nodes)
        bfs_queue = [0]
        bfs_perm = []
        bfs_focal = []
        visited = {bfs_queue[0]}
        while len(bfs_queue) > 0:
            current = bfs_queue.pop(0)
            bfs_perm.append(current)
            next_candid = []
            for motif in moltree.nodes[current].neighbors:
                if motif.nid in visited: continue
                next_candid.append(motif.nid)
                visited.add(motif.nid)
                bfs_focal.append(current)

            random.shuffle(next_candid)
            bfs_queue += next_candid

        return bfs_perm, bfs_focal

    def __call__(self, data):
        bfs_perm, bfs_focal = self.get_bfs_perm_motif(data['moltree'], self.vocab)
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_motifs = len(bfs_perm)
        num_masked = int(num_motifs * ratio)
        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_motifs - num_masked) < self.min_num_unmasked:
            num_masked = num_motifs - self.min_num_unmasked
        num_unmasked = num_motifs - num_masked

        context_motif_ids = bfs_perm[:-num_masked]
        context_idx = set()
        for i in context_motif_ids:
            context_idx = context_idx | set(data['moltree'].nodes[i].clique)
        context_idx = torch.LongTensor(list(context_idx))

        if num_masked == num_motifs:
            data['current_wid'] = torch.tensor([self.vocab_size])
            data['current_atoms'] = torch.tensor([data['protein_contact_idx']])
            data['next_wid'] = torch.tensor([data['moltree'].nodes[bfs_perm[-num_masked]].wid])
        else:
            data['current_wid'] = torch.tensor([data['moltree'].nodes[bfs_focal[-num_masked]].wid])
            data['next_wid'] = torch.tensor([data['moltree'].nodes[bfs_perm[-num_masked]].wid])  # For Prediction
            current_atoms = data['moltree'].nodes[bfs_focal[-num_masked]].clique
            data['current_atoms'] = torch.cat([torch.where(context_idx == i)[0] for i in current_atoms]) + len(data['protein_pos'])

        data['ligand_context_element'] = data['ligand_element'][context_idx]
        data['ligand_context_feature_full'] = data['ligand_atom_feature_full'][context_idx]  # For Input
        data['ligand_context_pos'] = data['ligand_pos'][context_idx]
        data['ligand_center'] = torch.mean(data['ligand_pos'], dim=0)
        data['num_atoms'] = torch.tensor([len(context_idx) + len(data['protein_pos'])])
        # distance matrix prediction
        if len(data['ligand_context_pos']) > 0:
            sample_idx = random.sample(data['moltree'].nodes[bfs_perm[0]].clique, 2)
            data['dm_ligand_idx'] = torch.cat([torch.where(context_idx == i)[0] for i in sample_idx])
            data['dm_protein_idx'] = torch.sort(
                torch.norm(data['protein_pos'] - data['ligand_context_pos'][data['dm_ligand_idx'][0]], dim=-1)).indices[:4]
            data['true_dm'] = torch.norm(data['protein_pos'][data['dm_protein_idx']].unsqueeze(1) - data['ligand_context_pos'][
                data['dm_ligand_idx']].unsqueeze(0), dim=-1).reshape(-1)
        else:
            data['true_dm'] = torch.tensor([])

        if len(data['ligand_context_pos']) > 0:
            data['protein_alpha_carbon_index'] = torch.tensor([i for i, name in enumerate(data['protein_atom_name']) if name =="CA"])
            data['alpha_carbon_indicator'] = torch.tensor([True if name =="CA" else False for name in data['protein_atom_name']])

        # assemble prediction
        data['protein_contact'] = torch.tensor(data['protein_contact'])
        if len(context_motif_ids) > 0:
            cand_labels, cand_mols = enumerate_assemble(data['moltree'].mol, context_idx.tolist(),
                                                        data['moltree'].nodes[bfs_focal[-num_masked]],
                                                        data['moltree'].nodes[bfs_perm[-num_masked]])
            data['cand_labels'] = cand_labels
            data['cand_mols'] = [mol_to_graph_data_obj_simple(mol) for mol in cand_mols]
        else:
            data['cand_labels'], data['cand_mols'] = torch.tensor([]), []

        data['ligand_context_bond_index'], data['ligand_context_bond_feature'] = subgraph(
            context_idx,
            data['ligand_bond_index'],
            edge_attr=data['ligand_bond_feature'],
            relabel_nodes=True,
        )
        data['ligand_context_num_neighbors'] = LigandCountNeighbors.count_neighbors(
            data['ligand_context_bond_index'],
            symmetry=True,
            num_nodes=context_idx.size(0),
        )
        data['ligand_frontier'] = data['ligand_context_num_neighbors'] < data['ligand_num_neighbors'][context_idx]
        data['_mask'] = 'bfs'

        # find a rotatable bond as the current motif
        rotatable_ids = []
        for i, id in enumerate(bfs_focal):
            if data['moltree'].nodes[id].rotatable:
                rotatable_ids.append(i)
        if len(rotatable_ids) == 0:
            # assign empty tensor
            data['ligand_torsion_xy_index'] = torch.tensor([])
            data['dihedral_mask'] = torch.tensor([]).bool()
            data['ligand_element_torsion'] = torch.tensor([])
            data['ligand_pos_torsion'] = torch.tensor([])
            data['ligand_feature_torsion'] = torch.tensor([])
            data['true_sin'], data['true_cos'], data['true_three_hop'] = torch.tensor([]), torch.tensor([]), torch.tensor([])
            data['xn_pos'], data['yn_pos'], data['y_pos'] = torch.tensor([]), torch.tensor([]), torch.tensor([])
        else:
            num_unmasked = random.sample(rotatable_ids, 1)[0]
            current_idx = torch.LongTensor(data['moltree'].nodes[bfs_focal[num_unmasked]].clique)
            next_idx = torch.LongTensor(data['moltree'].nodes[bfs_perm[num_unmasked + 1]].clique)
            current_idx_set = set(data['moltree'].nodes[bfs_focal[num_unmasked]].clique)
            next_idx_set = set(data['moltree'].nodes[bfs_perm[num_unmasked + 1]].clique)
            all_idx = set()
            for i in bfs_perm[:num_unmasked + 2]:
                all_idx = all_idx | set(data['moltree'].nodes[i].clique)
            all_idx = list(all_idx)
            x_id = current_idx_set.intersection(next_idx_set).pop()
            y_id = (current_idx_set - {x_id}).pop()
            data['ligand_torsion_xy_index'] = torch.cat([torch.where(torch.LongTensor(all_idx) == i)[0] for i in [x_id, y_id]])

            x_pos, y_pos = data['ligand_pos'][x_id], data['ligand_pos'][y_id]
            # remove x, y, and non-generated elements
            xn, yn = deepcopy(data['ligand_neighbors'][x_id]), deepcopy(data['ligand_neighbors'][y_id])
            xn.remove(y_id)
            yn.remove(x_id)
            xn, yn = xn[:3], yn[:3]
            # debug
            xn, yn = list_filter(xn, all_idx), list_filter(yn, all_idx)
            xn_pos, yn_pos = torch.zeros(3, 3), torch.zeros(3, 3)
            xn_pos[:len(xn)], yn_pos[:len(yn)] = data['ligand_pos'][xn], data['ligand_pos'][yn]
            xn_idx, yn_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
            xn_idx = xn_idx.squeeze(-1)
            yn_idx = yn_idx.squeeze(-1)
            dihedral_x, dihedral_y = torch.zeros(3), torch.zeros(3)
            dihedral_x[:len(xn)] = 1
            dihedral_y[:len(yn)] = 1
            data['dihedral_mask'] = torch.matmul(dihedral_x.view(3, 1), dihedral_y.view(1, 3)).view(-1).bool()
            data['true_sin'], data['true_cos'] = batch_dihedrals(xn_pos[xn_idx], x_pos.repeat(9, 1), y_pos.repeat(9, 1),
                                                           yn_pos[yn_idx])
            data['true_three_hop'] = torch.linalg.norm(xn_pos[xn_idx] - yn_pos[yn_idx], dim=-1)[data['dihedral_mask']]

            # random rotate to simulate the inference situation
            dir = data['ligand_pos'][current_idx[0]] - data['ligand_pos'][current_idx[1]]
            ref = data['ligand_pos'][current_idx[0]]
            next_motif_pos = data['ligand_pos'][next_idx]
            data['ligand_pos'][next_idx] = rand_rotate(dir, ref, next_motif_pos)

            data['ligand_element_torsion'] = data['ligand_element'][all_idx]
            data['ligand_pos_torsion'] = data['ligand_pos'][all_idx]
            data['ligand_feature_torsion'] = data['ligand_atom_feature_full'][all_idx]

            x_pos = data['ligand_pos'][x_id]
            data['y_pos'] = data['ligand_pos'][y_id] - x_pos
            data['xn_pos'], data['yn_pos'] = torch.zeros(3, 3), torch.zeros(3, 3)
            data['xn_pos'][:len(xn)], data['yn_pos'][:len(yn)] = data['ligand_pos'][xn] - x_pos, data['ligand_pos'][yn] - x_pos

        return data


class LigandMaskAll(LigandBFSMask):

    def __init__(self, vocab):
        super().__init__(min_ratio=1.0, vocab=vocab)


class LigandMixedMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, p_random=0.5, p_bfs=0.25,
                 p_invbfs=0.25):
        super().__init__()

        self.t = [
            LigandRandomMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=False),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=True),
        ]
        self.p = [p_random, p_bfs, p_invbfs]

    def __call__(self, data):
        f = random.choices(self.t, k=1, weights=self.p)[0]
        return f(data)


def get_mask(cfg, vocab):
    if cfg.type == 'bfs':
        return LigandBFSMask(
            min_ratio=cfg.min_ratio,
            max_ratio=cfg.max_ratio,
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
            vocab=vocab
        )
    elif cfg.type == 'random':
        return LigandRandomMask(
            min_ratio=cfg.min_ratio,
            max_ratio=cfg.max_ratio,
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
        )
    elif cfg.type == 'mixed':
        return LigandMixedMask(
            min_ratio=cfg.min_ratio,
            max_ratio=cfg.max_ratio,
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
            p_random=cfg.p_random,
            p_bfs=cfg.p_bfs,
            p_invbfs=cfg.p_invbfs,
        )
    elif cfg.type == 'all':
        return LigandMaskAll()
    else:
        raise NotImplementedError('Unknown mask: %s' % cfg.type)


def kabsch(A, B):
    # Input:
    #     Nominal  A Nx3 matrix of points
    #     Measured B Nx3 matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix (B to A)
    # t = 3x1 translation vector (B to A)
    assert len(A) == len(B)
    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.transpose(BB) * AA
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T
    t = -R * centroid_B.T + centroid_A.T
    return R, t
