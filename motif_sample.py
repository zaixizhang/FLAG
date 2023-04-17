import os
import shutil
import argparse
import random
import torch
import numpy as np
import math
from torch_geometric.data import Batch
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch.utils.data import DataLoader
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures, rdMolDescriptors
from rdkit import RDConfig
from rdkit.Chem.Descriptors import MolLogP, qed

from models.flag import FLAG
from utils.transforms import *
from utils.datasets import get_dataset
from utils.misc import *
from utils.data import *
# from utils.reconstruct import *
# from utils.chem import *
from utils.mol_tree import *
from utils.chemutils import *
from utils.dihedral_utils import *
from utils.sascorer import compute_sa_score

_fscores = None

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}

STATUS_RUNNING = 'running'
STATUS_FINISHED = 'finished'
STATUS_FAILED = 'failed'


def get_feat(mol):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    atomic_numbers = torch.LongTensor([6, 7, 8, 9, 15, 16, 17])  # C N O F P S Cl
    ptable = Chem.GetPeriodicTable()
    Chem.SanitizeMol(mol)
    feat_mat = np.zeros([mol.GetNumAtoms(), len(ATOM_FAMILIES)], dtype=np.long)
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
    for i in range(len(pos_list)):
        mol = mol_list[i]
        conf = mol.GetConformer()
        pos = np.array(pos_list[i])
        if mol.GetNumAtoms() == len(pos):
            for node in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(node, pos[node])
    return mol_list


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
    protein_offsets, ligand_offsets = torch.cat([torch.tensor([0]), protein_offsets]), torch.cat(
        [torch.tensor([0]), ligand_offsets])

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
    force1 = scatter_mean(model.refine_protein(input1) * norm_dir1, dim=0, index=sr_ligand_idx[dist_alpha <= 10.0])
    force2 = scatter_mean(model.refine_ligand(input2) * norm_dir2, dim=0, index=sr_ligand_idx0[dist_intra <= 10.0])
    ligand_pos[:len(force1)] += force1
    ligand_pos[:len(force2)] += force2

    ligand_pos = list(torch.split(ligand_pos, repeats))
    return ligand_pos


def ligand_gen(batch, model, vocab, config, center, refinement=True):
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
            slice_idx = torch.cat(
                [torch.tensor([0]).to(h_ctx.device), torch.cumsum(batch['protein_element_batch'].bincount(), dim=0)])
            focal_id = []
            for j in range(len(slice_idx) - 1):
                focus = focus_score[slice_idx[j]:slice_idx[j + 1]]
                focal_id.append(torch.argmax(focus.reshape(-1).float()).item() + slice_idx[j].item())
            focal_id = torch.tensor(focal_id)

            h_ctx_focal = h_ctx_protein[focal_id]
            current_wid = torch.tensor([vocab.size()] * config.sample.batch_size)
            next_motif_wid, motif_prob = model.forward_motif(h_ctx_focal, current_wid,
                                                 torch.arange(config.sample.batch_size).to(h_ctx_focal.device))
            logp_motif = [[p] for p in np.log(motif_prob.cpu().numpy())] #init logp list

            mol_list = [Chem.MolFromSmiles(vocab.get_smiles(id)) for id in next_motif_wid]
            for j in range(config.sample.batch_size):
                AllChem.EmbedMolecule(mol_list[j])
                AllChem.UFFOptimizeMolecule(mol_list[j])
                ligand_pos, ligand_feat = torch.tensor(mol_list[j].GetConformer().GetPositions()), get_feat(mol_list[j])
                feat_list.append(ligand_feat)
                # set the initial positions with distance matrix
                reference_pos, reference_idx = find_reference(batch['protein_pos'][slice_idx[j]:slice_idx[j + 1]],
                                                              focal_id[j] - slice_idx[j])
                p_idx, l_idx = torch.cartesian_prod(torch.arange(4), torch.arange(len(ligand_pos))).chunk(2, dim=-1)
                p_idx = p_idx.squeeze(-1)
                l_idx = l_idx.squeeze(-1)
                d_m = model.dist_mlp(
                    torch.cat([protein_atom_feature[reference_idx[p_idx]], ligand_feat[l_idx]], dim=-1)).reshape(4,
                                                                                                                 len(ligand_pos))
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
            repeats = torch.tensor([len(pos) for pos in pos_list])
            ligand_batch = torch.repeat_interleave(torch.arange(config.sample.batch_size), repeats)
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
            can_focus = focus_score > 0.5
            slice_idx = torch.cat([torch.tensor([0]), torch.cumsum(repeats, dim=0)])

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
                                                 torch.tensor(current_wid).to(h_ctx_focal.device),
                                                 torch.tensor(current_atoms_batch).to(h_ctx_focal.device), n_samples=5)

            # assemble
            next_motif_smiles = [vocab.get_smiles(id) for id in next_motif_wid]
            new_mol_list, new_atoms, one_atom_attach, intersection, attach_fail = model.forward_attach(mol_list,
                                                                                                       next_motif_smiles)

            for j in range(len(mol_list)):
                if ~finished[j] and ~attach_fail[j]:
                    # num_new_atoms
                    mol_list[j] = new_mol_list[j]
            rotatable = torch.logical_and(torch.tensor(current_atoms_batch).bincount() == 2,
                                          torch.tensor(one_atom_attach))
            rotatable = torch.logical_and(rotatable, ~torch.tensor(attach_fail))
            rotatable = torch.logical_and(rotatable, ~finished)
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
                anchor_pos = pos_list[j][anchor]
                Chem.SanitizeMol(mol)
                AllChem.EmbedMolecule(mol, useRandomCoords=True)
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except:
                    print('UFF error')
                anchor_pos_new = mol.GetConformer().GetPositions()[anchor]
                new_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]
                '''
                R, T = kabsch(np.matrix(anchor_pos), np.matrix(anchor_pos_new))
                new_pos = R * np.matrix(mol.GetConformer().GetPositions()[new_idx]).T + np.tile(T, (1, len(new_idx)))
                new_pos = np.array(new_pos.T)'''
                new_pos = mol.GetConformer().GetPositions()[new_idx]
                new_pos, _, _ = kabsch_torch(torch.tensor(anchor_pos_new), anchor_pos, torch.tensor(new_pos))

                conf = mol.GetConformer()
                # update curated parameters
                pos_list[j] = torch.cat([pos_list[j], new_pos])
                feat_list[j] = get_feat(mol_list[j])
                for node in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(node, np.array(pos_list[j][node]))
                assert mol.GetNumAtoms() == len(pos_list[j])

            # predict alpha and rotate (only change the position)
            if torch.sum(rotatable) > 0 and i >= 2:
                repeats = torch.tensor([len(pos) for pos in pos_list])
                ligand_batch = torch.repeat_interleave(torch.arange(len(pos_list)), repeats)
                slice_idx = torch.cat([torch.tensor([0]), torch.cumsum(repeats, dim=0)])
                xy_index = [(np.array(motif_to_atoms[j][motif_id[j]]) + slice_idx[j].item()).tolist() for j in
                            range(len(slice_idx) - 1) if
                            rotatable[j]]

                alpha = model.forward_alpha(protein_pos=batch['protein_pos'].float(),
                                            protein_atom_feature=batch['protein_atom_feature'].float(),
                                            ligand_pos=torch.cat(pos_list, dim=0).float(),
                                            ligand_atom_feature=torch.cat(feat_list, dim=0).float(),
                                            batch_protein=batch['protein_element_batch'],
                                            batch_ligand=ligand_batch, xy_index=torch.tensor(xy_index),
                                            rotatable=rotatable)

                rotatable_id = [id for id in range(len(mol_list)) if rotatable[id]]
                xy_index = [motif_to_atoms[j][motif_id[j]] for j in range(len(slice_idx) - 1) if rotatable[j]]
                x_index = [intersection[j] for j in range(len(slice_idx) - 1) if rotatable[j]]
                y_index = [(set(xy_index[k]) - set(x_index[k])).pop() for k in range(len(x_index))]

                for j in range(len(alpha)):
                    mol = mol_list[rotatable_id[j]]
                    new_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]
                    positions = pos_list[rotatable_id[j]]

                    xn_pos = positions[new_idx].float()
                    xn_pos = rand_rotate((positions[x_index[j]] - positions[y_index[j]]).reshape(-1),
                                         positions[x_index[j]].reshape(-1),
                                         xn_pos, alpha[j])
                    if xn_pos.shape[0] > 0:
                        pos_list[rotatable_id[j]][-len(xn_pos):] = xn_pos
                    conf = mol.GetConformer()
                    for node in range(mol.GetNumAtoms()):
                        conf.SetAtomPosition(node, np.array(pos_list[rotatable_id[j]][node]))
                    assert mol.GetNumAtoms() == len(pos_list[rotatable_id[j]])

    return mol_list, pos_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample.yml')
    parser.add_argument('-i', '--data_id', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--vocab_path', type=str, default='./utils/vocab.txt')
    args = parser.parse_args()

    # Load vocab
    vocab = []
    num = []
    for line in open(args.vocab_path):
        p1, _, p3 = line.partition(':')
        vocab.append(p1)
        num.append(int(p3))
    vocab = Vocab(vocab)
    num = torch.tensor(num)
    weight = (num[0] / num) ** 0.4

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
    center = data['ligand_center']
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
        weight=weight,
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
            for batch in tqdm(sample_loader):
                '''
                for key in batch:
                    batch[key] = batch[key].to(args.device)'''
                gen_data, pos_list = ligand_gen(batch, model, vocab, config, center)
                data_list.extend(gen_data)
                # Calculate metrics
                print([Chem.MolToSmiles(mol) for mol in data_list])
                smiles = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in data_list]
                qed_list = [qed(mol) for mol in smiles if mol.GetNumAtoms() >= 8]
                logp_list = [MolLogP(mol) for mol in smiles]
                sa_list = [compute_sa_score(mol) for mol in smiles]
                Lip_list = [lipinski(mol) for mol in smiles]
                print('QED %.6f | LogP %.6f | SA %.6f | Lipinski %.6f \n' % (
                    np.average(qed_list), np.average(logp_list), np.average(sa_list), np.average(Lip_list)))
                print(sa_list)
                SetMolPos(data_list, pos_list)

                with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
                    for i, mol in enumerate(data_list):
                        if mol.GetNumAtoms() < 8:
                            continue
                        smiles_f.write(Chem.MolToSmiles(mol) + '\n')
                        writer = Chem.SDWriter(os.path.join(log_dir, '%d.sdf' % i))
                        # writer.SetKekulize(False)
                        writer.write(mol, confId=0)
                        writer.close()

    except KeyboardInterrupt:
        logger.info('Terminated. Generated molecules will be saved.')
        with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
            for i, mol in enumerate(data_list):
                if mol.GetNumAtoms() < 8:
                    continue
                smiles_f.write(Chem.MolToSmiles(mol) + '\n')
                writer = Chem.SDWriter(os.path.join(log_dir, '%d.sdf' % i))
                # writer.SetKekulize(False)
                writer.write(mol, confId=0)
                writer.close()
