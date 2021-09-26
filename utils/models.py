import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pdb
from torch.nn import functional as F
from dgl.ops import edge_softmax
import dgl.function as fn
from torch.distributions.normal import Normal
import math
import numpy as np
import yaml

'''MLP model'''
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class HGTLayer(nn.Module):
    def __init__(self,in_dim, out_dim, dropout=0.2,use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = 4
        self.node_dict = {'ped': 0,'skater': 1,'biker': 2,'car': 3,'lane': 4,'sidewalk': 5,'lawn': 6,'Obstacle': 7,'Door': 8}
        self.edge_dict = {'adj': 0, 'on': 1, '_on': 2}
        self.num_types = len(self.node_dict)
        self.num_relations = len(self.edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types

        self.d_k = self.out_dim // self.n_heads  # 64/4 = 16
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm
        print('use_norm = {}'.format(use_norm))

        for t in range(self.num_types):      #每个type的节点有一个编码器
            self.k_linears.append(nn.Linear(self.in_dim, self.out_dim))
            self.q_linears.append(nn.Linear(self.in_dim, self.out_dim))
            self.v_linears.append(nn.Linear(self.in_dim, self.out_dim))
            self.a_linears.append(nn.Linear(self.out_dim, self.out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(self.out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, self.n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, self.n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]
                a = k_linear(h[srctype])
                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            # G.multi_update_all({etype: (fn.u_mul_e('v', 't', 'm'), fn.sum('m', 't')) \
            #                     for etype in edge_dict}, cross_reducer='mean')
            G.multi_update_all({(srctype, etype, dsttype): (fn.u_mul_e('v', 't', 'm'), fn.sum('m', 't')) \
                                for (srctype, etype, dsttype) in G.canonical_etypes}, cross_reducer='mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

class CNNMapEncoder(nn.Module):
    def __init__(self, map_channels, hidden_channels, output_size, masks, strides):
        super(CNNMapEncoder, self).__init__()
        self.convs = nn.ModuleList()
        patch_size_x = 1500
        patch_size_y = 1400
        input_size = (map_channels, patch_size_x, patch_size_y)
        x_dummy = torch.ones(input_size).unsqueeze(0) * torch.tensor(float('nan'))

        for i, hidden_size in enumerate(hidden_channels):
            self.convs.append(nn.Conv2d(map_channels if i == 0 else hidden_channels[i-1],
                                        hidden_channels[i], masks[i],
                                        stride=strides[i]))
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), output_size)

    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# class CNNMapEncoder(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(CNNMapEncoder, self).__init__()
#         self.conv1 = nn.Conv2d(3, 128, 5, stride=2)
#         self.conv2 = nn.Conv2d(128, 256, 5, stride=3)
#         self.conv3 = nn.Conv2d(256, 64, 5, stride=2)
#         self.fc = nn.Linear(7 * 7 * 64, 512)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, 7 * 7 * 64)
#         x = F.relu(self.fc(x))
#         return x

class PECNet(nn.Module):

    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, predictor_size, non_local_theta_size, non_local_phi_size, non_local_g_size, fdim, zdim, nonlocal_pools, non_local_dim, sigma, past_length, future_length, verbose):
        '''
        Args:
            size parameters: Dimension sizes
            nonlocal_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(PECNet, self).__init__()
        self.t_human = {'ped': 0, 'skater': 1, 'biker': 2, 'car': 3}
        self.t_scene = {0: 'lane', 1: 'sidewalk', 2: 'lawn', 3: 'Obstacle', 4: 'Door'}

        self.zdim = zdim
        self.nonlocal_pools = nonlocal_pools
        self.sigma = sigma
        self.n_layers = 2

        fdim = 16
        ddim = 16
        enc_past_size = [512,256]
        enc_dest_size = [8,16]
        zdim = 16
        sdim = 256
        scene_hidden_size = [1024, 512, 256]
        non_local_dim = 128
        non_local_theta_size = [256,128,64]
        non_local_phi_size = [256,128,64]
        predictor_size = [1024,512,256]
        dec_size = [1024,512,1024]
        # takes in the past
        self.encoder_past = nn.ModuleList()
        self.encoder_delta = nn.ModuleList()
        # self.encoder_past_lstm = nn.LSTM(input_size=2, hidden_size=fdim, num_layers=1, batch_first=True)

        self.encoder_dest = nn.ModuleList()

        self.encoder_latent = nn.ModuleList()

        self.decoder = nn.ModuleList()

        self.predictor = nn.ModuleList()

        self.gcs = nn.ModuleList()

        for i in self.t_human.keys():
            self.encoder_past.append(MLP(input_dim = past_length*2, output_dim = fdim, hidden_size=enc_past_size))

            self.encoder_delta.append(MLP(input_dim = (past_length-1)*2, output_dim = fdim, hidden_size=enc_past_size))
            # self.encoder_past_lstm = nn.LSTM(input_size=2, hidden_size=fdim, num_layers=1, batch_first=True)

            self.encoder_dest.append(MLP(input_dim = 2, output_dim = ddim, hidden_size=enc_dest_size))

            self.encoder_latent.append(MLP(input_dim = fdim + fdim + ddim, output_dim = 2*zdim, hidden_size=enc_latent_size))

            self.decoder.append(MLP(input_dim =  fdim + fdim + zdim, output_dim = 2, hidden_size=dec_size))

            self.predictor.append(MLP(input_dim= fdim + fdim + ddim + 2, output_dim=2 * (future_length - 1), hidden_size=predictor_size))

        self.encoder_scene = MLP(input_dim=1000 * 2, output_dim=2 * fdim, hidden_size=scene_hidden_size)

        hidden_channels = [5,5,5,3]
        strides = [2,2,1,1]
        masks= [5, 5, 5, 5]
        output_size = 2 * fdim

        self.encoder_scene_picture = CNNMapEncoder(3, hidden_channels, output_size, masks, strides)

        for _ in range(self.n_layers):
            self.gcs.append(
                HGTLayer(in_dim=2 * fdim, out_dim=2 * fdim , dropout=0.1, use_norm=True))

        # self.social_theta = MLP(input_dim = 2*fdim + 2, output_dim = non_local_dim, hidden_size=non_local_theta_size)#  128   [256 128 64]
        # self.social_phi = MLP(input_dim = 2*fdim + 2, output_dim = non_local_dim, hidden_size=non_local_phi_size)
        # self.social_g = MLP(input_dim = 2*fdim + 2, output_dim = 2*fdim + 2, hidden_size=non_local_g_size)
        #
        # self.scene_theta = MLP(input_dim=2 * fdim + 2, output_dim=non_local_dim, hidden_size=non_local_theta_size)  # 128   [256 128 64]
        # self.scene_phi = MLP(input_dim=sdim, output_dim=non_local_dim, hidden_size=non_local_phi_size)
        # self.scene_g = MLP(input_dim=sdim, output_dim=2 * fdim + 2, hidden_size=non_local_g_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
#             print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
            print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest[1])))
            print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent[1])))
            print("Scene Encoder architecture : {}".format(architecture(self.encoder_scene)))
            print("Decoder architecture : {}".format(architecture(self.decoder[1])))
            print("Predictor architecture : {}".format(architecture(self.predictor[1])))

            # print("Non Local Theta architecture : {}".format(architecture(self.social_theta)))
            # print("Non Local Phi architecture : {}".format(architecture(self.social_phi)))
            # print("Non Local g architecture : {}".format(architecture(self.social_g)))

    # def forward(self, x, initial_pos,  dest = None, mask_human =  None, mask_scene = None, scene = None, device=torch.device('cpu')):
    def forward(self, g, pic_b, pic_dict_b,scene_name, type_human, type_scene ,device=torch.device('cpu')):

        # provide destination iff training
        # assert model.training
        # assert self.training ^ (g.nodes['ped'].data['dest'] is None)
        # assert self.training ^ (mask_human is None)

        ftraj, fdelta = {}, {}
        # encode
        for human in type_human:
            ftraj[human] = self.encoder_past[self.t_human[human]](g.nodes[human].data['x'])
            fdelta[human] = self.encoder_delta[self.t_human[human]](g.nodes[human].data['delta'])

        # pic_feature = self.encoder_scene_picture(pic_b.permute(0,3,1,2))
        # pic_fea_all = {}

        # for i, scene in enumerate(scene_name):
        #     for human in type_human:
        #         # pic_fea_all[human] = torch.zeros((g.num_nodes(human), 64), dtype=torch.float64).to(device)
        #         if human not in pic_dict_b[scene].keys():
        #             continue
        #         human_len = pic_dict_b[scene][human]
                # pic_fea_all[human][pic_dict_b[scene][human]] = pic_feature[i].unsqueeze(0).repeat(len(human_len),1).to(device)

        z = {}
        if not self.training:
            for human in type_human:
                z[human] = torch.Tensor(g.nodes[human].data['x'].size(0), self.zdim)
                z[human].normal_(0, self.sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            dest_features, features, latent, mu, logvar, var, eps = {},{},{},{},{},{},{}
            for human in type_human:
                dest_features[human] = self.encoder_dest[self.t_human[human]](g.nodes[human].data['dest'])
                features[human] = torch.cat((ftraj[human], fdelta[human], dest_features[human] ), dim=1)
                latent[human] =  self.encoder_latent[self.t_human[human]](features[human])

                mu[human] = latent[human][:, 0:self.zdim] # 2-d array
                logvar[human] = latent[human][:, self.zdim:] # 2-d array

                var[human] = logvar[human].mul(0.5).exp_()
                eps[human] = torch.DoubleTensor(var[human].size()).normal_()
                eps[human] = eps[human].to(device)
                z[human] = eps[human].mul(var[human]).add_(mu[human])

        decoder_input, generated_dest = {}, {}
        for human in type_human:
            z[human] = z[human].double().to(device)
            # decoder_input[human] = torch.cat((ftraj[human], z[human],pic_fea_all[human]), dim = 1)
            decoder_input[human] = torch.cat((ftraj[human],fdelta[human], z[human]), dim = 1)
            generated_dest[human] = self.decoder[self.t_human[human]](decoder_input[human])

        generated_dest_features, prediction_features,  pred_future= {}, {}, {}
        if self.training:
            # prediction in training, no best selection
            for human in type_human:
                generated_dest_features[human] = self.encoder_dest[self.t_human[human]](generated_dest[human])

                prediction_features[human] = torch.cat((ftraj[human], fdelta[human], generated_dest_features[human], g.nodes[human].data['initial_pos']), dim = 1)

            # for scene in type_scene:
            #     prediction_features[scene] = self.encoder_scene(g.nodes[scene].data['data'])
                # prediction_features[scene] = self.encoder_scene_picture(g.nodes[scene].data['picture'].unsqueeze(1))

            # X = prediction_features

            # for i in range(self.n_layers):
            #     X = self.gcs[i](g, X)

            for human in type_human:
                # pred_future[human] = self.predictor[self.t_human[human]](X[human]+prediction_features[human])
                pred_future[human] = self.predictor[self.t_human[human]](prediction_features[human])
            return generated_dest, mu, logvar, pred_future

        return generated_dest

    # separated for forward to let choose the best destination
    def predict(self, g, generated_dest, type_human , type_scene):

        ftraj, fdelta, generated_dest_features, prediction_features, interpolated_future = {}, {}, {}, {}, {}
        for human in type_human:
            ftraj[human] = self.encoder_past[self.t_human[human]](g.nodes[human].data['x'])
            fdelta[human] = self.encoder_delta[self.t_human[human]](g.nodes[human].data['delta'])

        # out_encoder, (final_hidden_state, final_cell_state) = self.encoder_past_lstm(past)  # out_encoder, (final_hidden_state, final_cell_state)
        # ftraj = out_encoder[:, -1:].squeeze()

            generated_dest_features[human] = self.encoder_dest[self.t_human[human]](generated_dest[human])
            prediction_features[human] = torch.cat((ftraj[human], fdelta[human], generated_dest_features[human], g.nodes[human].data['initial_pos']), dim = 1)


        # for scene in type_scene:
        #     prediction_features[scene] = self.encoder_scene(g.nodes[scene].data['data'])
            # prediction_features[scene] = self.encoder_scene_picture(g.nodes[scene].data['picture'].unsqueeze(1))

        # X = prediction_features

        # for i in range(self.n_layers):
        #     X = self.gcs[i](g, X)

        for human in type_human:
            # interpolated_future[human] = self.predictor[self.t_human[human]](X[human]+prediction_features[human])
            interpolated_future[human] = self.predictor[self.t_human[human]](prediction_features[human])


        return interpolated_future
