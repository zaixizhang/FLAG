import sys

sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data, Batch

from .encoders import get_encoder, GNN_graphpred, MLP
from .fields import get_field
from .common import *
from utils import dihedral_utils, chemutils


class FLAG(Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.protein_atom_emb = Linear(protein_atom_feature_dim, config.hidden_channels)
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, config.hidden_channels)
        self.embedding = nn.Embedding(vocab.size() + 1, config.hidden_channels)
        self.W = nn.Linear(2 * config.hidden_channels, config.hidden_channels)
        self.W_o = nn.Linear(config.hidden_channels, self.vocab.size())
        self.encoder = get_encoder(config.encoder)
        self.comb_head = GNN_graphpred(num_layer=3, emb_dim=config.hidden_channels, num_tasks=1, JK='last',
                                       drop_ratio=0.5, graph_pooling='mean', gnn_type='gin')
        if config.random_alpha:
            self.alpha_mlp = MLP(in_dim=config.hidden_channels * 4, out_dim=1, num_layers=2)
        else:
            self.alpha_mlp = MLP(in_dim=config.hidden_channels * 3, out_dim=1, num_layers=2)
        self.focal_mlp = MLP(in_dim=config.hidden_channels, out_dim=1, num_layers=1)
        #self.c_mlp = MLP(in_dim=config.hidden_channels*4, out_dim=1, num_layers=1)
        self.dist_mlp = MLP(in_dim=protein_atom_feature_dim + ligand_atom_feature_dim, out_dim=1, num_layers=2)
        if config.refinement:
            self.refine_protein = MLP(in_dim=protein_atom_feature_dim + ligand_atom_feature_dim + config.encoder.edge_channels, out_dim=1, num_layers=2)
            self.refine_ligand = MLP(in_dim=ligand_atom_feature_dim*2 + config.encoder.edge_channels, out_dim=1, num_layers=2)

        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction='mean', smoothing=0.1)
        self.pred_loss = nn.CrossEntropyLoss()
        self.comb_loss = nn.BCEWithLogitsLoss()
        self.three_hop_loss = torch.nn.MSELoss()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.dist_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein, batch_ligand):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx, protein_mask = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch_ligand)
        h_ctx = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx)  # (N_p+N_l, H)
        focal_pred = self.focal_mlp(h_ctx)

        return focal_pred, protein_mask, h_ctx

    def forward_motif(self, h_ctx_focal, current_wid, current_atoms_batch):
        node_hiddens = scatter_add(h_ctx_focal, dim=0, index=current_atoms_batch)
        motif_hiddens = self.embedding(current_wid)
        pred_vecs = torch.cat([node_hiddens, motif_hiddens], dim=1)
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        _, preds = torch.max(pred_scores, dim=1)
        # random select in topk
        k = 5
        select_pool = torch.topk(pred_scores, k, dim=1)[1]
        index = torch.randint(k, (select_pool.shape[0],))
        preds = torch.cat([select_pool[i][index[i]].unsqueeze(0) for i in range(len(index))])
        return preds

    def forward_attach(self, mol_list, next_motif_smiles):
        cand_mols, cand_batch, new_atoms, one_atom_attach, intersection, attach_fail = chemutils.assemble(mol_list, next_motif_smiles)
        graph_data = Batch.from_data_list([chemutils.mol_to_graph_data_obj_simple(mol) for mol in cand_mols])
        comb_pred = self.comb_head(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch).reshape(-1)
        slice_idx = torch.cat([torch.tensor([0]), torch.cumsum(cand_batch.bincount(), dim=0)], dim=0)
        select = [(torch.argmax(comb_pred[slice_idx[i]:slice_idx[i + 1]]) + slice_idx[i]).item() for i in
                  range(len(slice_idx) - 1)]
        '''
        select = []
        for k in range(len(slice_idx) - 1):
            id = torch.multinomial(torch.exp(comb_pred[slice_idx[k]:slice_idx[k + 1]]).reshape(-1).float(), 1)
            select.append((id+slice_idx[k]).item())'''

        select_mols = [cand_mols[i] for i in select]
        new_atoms = [new_atoms[i] for i in select]
        one_atom_attach = [one_atom_attach[i] for i in select]
        intersection = [intersection[i] for i in select]
        return select_mols, new_atoms, one_atom_attach, intersection, attach_fail

    def forward_alpha(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein,
                     batch_ligand, xy_index, rotatable):
        # encode again
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx, protein_mask = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch_ligand)
        h_ctx = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx)  # (N_p+N_l, H)
        h_ctx_ligand = h_ctx[~protein_mask]
        hx, hy = h_ctx_ligand[xy_index[:, 0]], h_ctx_ligand[xy_index[:, 1]]
        h_mol = scatter_add(h_ctx_ligand, dim=0, index=batch_ligand)
        h_mol = h_mol[rotatable]
        if self.config.random_alpha:
            rand_dist = torch.distributions.normal.Normal(loc=0, scale=1)
            rand_alpha = rand_dist.sample(hx.shape).to(hx.device)
            alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol, rand_alpha], dim=-1)) + self.alpha_mlp(
                torch.cat([hy, hx, h_mol, rand_alpha], dim=-1))
        else:
            alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol], dim=-1)) + self.alpha_mlp(
                torch.cat([hy, hx, h_mol], dim=-1))
        return alpha

    def get_loss(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, ligand_pos_torsion,
                 ligand_atom_feature_torsion, batch_protein, batch_ligand, batch_ligand_torsion, batch):
        self.device = protein_pos.device
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        loss_list = [0, 0, 0, 0, 0, 0]

        # Encode for motif prediction
        h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch_ligand)
        h_ctx = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx)  # (N_p+N_l, H)
        h_ctx_ligand = h_ctx[~mask_protein]
        h_ctx_protein = h_ctx[mask_protein]
        h_ctx_focal = h_ctx[batch['current_atoms']]

        # Encode for torsion prediction
        if len(batch['y_pos']) > 0:
            h_ligand_torsion = self.ligand_atom_emb(ligand_atom_feature_torsion)
            h_ctx_torison, pos_ctx_torison, batch_ctx_torsion, mask_protein = compose_context_stable(h_protein=h_protein,
                                                                                                     h_ligand=h_ligand_torsion,
                                                                                                     pos_protein=protein_pos,
                                                                                                     pos_ligand=ligand_pos_torsion,
                                                                                                     batch_protein=batch_protein,
                                                                                                     batch_ligand=batch_ligand_torsion)
            h_ctx_torsion = self.encoder(node_attr=h_ctx_torison, pos=pos_ctx_torison, batch=batch_ctx_torsion)  # (N_p+N_l, H)
            h_ctx_ligand_torsion = h_ctx_torsion[~mask_protein]

        # next motif prediction

        node_hiddens = scatter_add(h_ctx_focal, dim=0, index=batch['current_atoms_batch'])
        motif_hiddens = self.embedding(batch['current_wid'])
        pred_vecs = torch.cat([node_hiddens, motif_hiddens], dim=1)
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        pred_loss = self.pred_loss(pred_scores, batch['next_wid'])
        loss_list[0] = pred_loss.item()

        # attachment prediction
        if len(batch['cand_labels']) > 0:
            cand_mols = batch['cand_mols']
            comb_pred = self.comb_head(cand_mols.x, cand_mols.edge_index, cand_mols.edge_attr, cand_mols.batch)
            comb_loss = self.comb_loss(comb_pred, batch['cand_labels'].view(comb_pred.shape).float())
            loss_list[1] = comb_loss.item()
        else:
            comb_loss = 0

        # focal prediction
        focal_ligand_pred, focal_protein_pred = self.focal_mlp(h_ctx_ligand), self.focal_mlp(h_ctx_protein)
        focal_loss = self.focal_loss(focal_ligand_pred.reshape(-1), batch['ligand_frontier'].float()) +\
                     self.focal_loss(focal_protein_pred.reshape(-1), batch['protein_contact'].float())
        loss_list[2] = focal_loss.item()

        # distance matrix prediction
        if len(batch['true_dm']) > 0:
            input = torch.cat(
                [protein_atom_feature[batch['dm_protein_idx']], ligand_atom_feature[batch['dm_ligand_idx']]], dim=-1)
            pred_dist = self.dist_mlp(input)
            dm_loss = self.dist_loss(pred_dist, batch['true_dm'])/10
            loss_list[3] = dm_loss.item()
        else:
            dm_loss = 0

        # structure refinement loss
        if self.config.refinement and len(batch['true_dm']) > 0:
            true_distance_alpha = torch.norm(batch['ligand_context_pos'][batch['sr_ligand_idx']] - batch['protein_pos'][batch['sr_protein_idx']], dim=1)
            true_distance_intra = torch.norm(batch['ligand_context_pos'][batch['sr_ligand_idx0']] - batch['ligand_context_pos'][batch['sr_ligand_idx0']], dim=1)
            input_distance_alpha = ligand_pos[batch['sr_ligand_idx']] - protein_pos[batch['sr_protein_idx']]
            input_distance_intra = ligand_pos[batch['sr_ligand_idx0']] - ligand_pos[batch['sr_ligand_idx1']]
            distance_emb1 = self.encoder.distance_expansion(torch.norm(input_distance_alpha, dim=1))
            distance_emb2 = self.encoder.distance_expansion(torch.norm(input_distance_intra, dim=1))
            input1 = torch.cat([ligand_atom_feature[batch['sr_ligand_idx']], protein_atom_feature[batch['sr_protein_idx']], distance_emb1], dim=-1)
            input2 = torch.cat([ligand_atom_feature[batch['sr_ligand_idx0']], ligand_atom_feature[batch['sr_ligand_idx1']], distance_emb2], dim=-1)
            #distance cut_off
            norm_dir1 = F.normalize(input_distance_alpha, p=2, dim=1)* torch.where(true_distance_alpha>10.0, torch.zeros_like(true_distance_alpha), true_distance_alpha).unsqueeze(1)
            norm_dir2 = F.normalize(input_distance_intra, p=2, dim=1)* torch.where(true_distance_intra>10.0, torch.zeros_like(true_distance_intra), true_distance_intra).unsqueeze(1)
            new_ligand_pos = ligand_pos\
                      + scatter_mean(self.refine_protein(input1)*norm_dir1, dim=0, index=batch['sr_ligand_idx'])\
                      + scatter_mean(self.refine_ligand(input2)*norm_dir2, dim=0, index=batch['sr_ligand_idx0'])
            refine_dist1 = torch.norm(new_ligand_pos[batch['sr_ligand_idx']] - protein_pos[batch['sr_protein_idx']], dim=1)
            refine_dist2 = torch.norm(new_ligand_pos[batch['sr_ligand_idx0']] - new_ligand_pos[batch['sr_ligand_idx1']], dim=1)
            sr_loss = (self.dist_loss(refine_dist1, true_distance_alpha) + self.dist_loss(refine_dist2, true_distance_intra))/10
            loss_list[5] = sr_loss.item()
        else:
            sr_loss = 0

        # torsion prediction
        if len(batch['y_pos']) > 0:
            Hx = dihedral_utils.rotation_matrix_v2(batch['y_pos'])
            xn_pos = torch.matmul(Hx, batch['xn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
            yn_pos = torch.matmul(Hx, batch['yn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
            y_pos = torch.matmul(Hx, batch['y_pos'].unsqueeze(1).permute(0, 2, 1)).squeeze(-1)

            hx, hy = h_ctx_ligand_torsion[batch['ligand_torsion_xy_index'][:, 0]], h_ctx_ligand_torsion[
                batch['ligand_torsion_xy_index'][:, 1]]
            h_mol = scatter_add(h_ctx_ligand_torsion, dim=0, index=batch['ligand_element_torsion_batch'])
            if self.config.random_alpha:
                rand_dist = torch.distributions.normal.Normal(loc=0, scale=1)
                rand_alpha = rand_dist.sample(hx.shape).to(self.device)
                alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol, rand_alpha], dim=-1))
            else:
                alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol], dim=-1))
            # rotate xn
            R_alpha = self.build_alpha_rotation(torch.sin(alpha).squeeze(-1), torch.cos(alpha).squeeze(-1))
            xn_pos = torch.matmul(R_alpha, xn_pos.permute(0, 2, 1)).permute(0, 2, 1)

            p_idx, q_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
            p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
            pred_sin, pred_cos = dihedral_utils.batch_dihedrals(xn_pos[:, p_idx],
                                                 torch.zeros_like(y_pos).unsqueeze(1).repeat(1, 9, 1),
                                                 y_pos.unsqueeze(1).repeat(1, 9, 1),
                                                 yn_pos[:, q_idx])
            dihedral_loss = torch.mean(
                dihedral_utils.von_Mises_loss(batch['true_cos'], pred_cos.reshape(-1), batch['true_sin'], pred_cos.reshape(-1))[batch['dihedral_mask']])
            torsion_loss = -dihedral_loss
            loss_list[4] = torsion_loss.item()
        else:
            torsion_loss = 0

        # dm: distance matrix
        loss = pred_loss + comb_loss + focal_loss + dm_loss + torsion_loss + sr_loss

        return loss, loss_list

    def build_alpha_rotation(self, alpha, alpha_cos=None):
        """
        Builds the alpha rotation matrix

        :param alpha: predicted values of torsion parameter alpha (n_dihedral_pairs)
        :return: alpha rotation matrix (n_dihedral_pairs, 3, 3)
        """
        H_alpha = torch.FloatTensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(alpha.shape[0], 1, 1).to(self.device)

        if torch.is_tensor(alpha_cos):
            H_alpha[:, 1, 1] = alpha_cos
            H_alpha[:, 1, 2] = -alpha
            H_alpha[:, 2, 1] = alpha
            H_alpha[:, 2, 2] = alpha_cos
        else:
            H_alpha[:, 1, 1] = torch.cos(alpha)
            H_alpha[:, 1, 2] = -torch.sin(alpha)
            H_alpha[:, 2, 1] = torch.sin(alpha)
            H_alpha[:, 2, 2] = torch.cos(alpha)

        return H_alpha

