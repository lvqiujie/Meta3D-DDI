from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from models.meta_model import *


class MetaSchNet(torch.nn.Module):
    def __init__(self, energy_and_force=False, cutoff=10.0, num_layers=6, hidden_channels=128, num_filters=128,
                 num_gaussians=50,out_channels = 1):
        super(MetaSchNet, self).__init__()

        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        # self.init_v = Embedding(100, hidden_channels)
        self.init_v_weight = nn.Parameter(torch.Tensor(100, hidden_channels))


        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([update_e(hidden_channels, num_filters, num_gaussians, cutoff) for _ in range(num_layers)])

        self.update_u = update_u(hidden_channels,out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.init_v_weight)
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, pos, z, batch, params=None):
        param_dict = dict()

        if params is not None:
            # params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        # z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)


        if param_dict["init_v_weight"] is not None:
            embedding_weight = param_dict["init_v_weight"]
        else:
            embedding_weight = self.init_v_weight
        # v = self.init_v(z)
        v = F.embedding(z, embedding_weight)


        for i,(update_e, update_v)  in enumerate(zip(self.update_es, self.update_vs)):
            e = update_e(v, dist, dist_emb, edge_index, param_dict["update_es"+str(i)])
            v = update_v(v, e, edge_index, param_dict["update_vs"+str(i)])
        u = self.update_u(v, batch, param_dict["update_u"])

        return u


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(update_e, self).__init__()
        self.cutoff = cutoff

        self.lin = MetaLinearLayer(hidden_channels, num_filters, use_bias=False)
        self.mlp_lin1 = MetaLinearLayer(num_gaussians, num_filters)
        self.mlp_softplus = ShiftedSoftplus()
        self.mlp_lin2 = MetaLinearLayer(num_filters, num_filters)


        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weights)
        torch.nn.init.xavier_uniform_(self.mlp_lin1.weights)
        self.mlp_lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp_lin2.weights)
        self.mlp_lin2.bias.data.fill_(0)


    def forward(self, v, dist, dist_emb, edge_index, params=None):
        lin_params = None
        mlp_lin1_params = None
        mlp_lin2_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            lin_params = params['lin']
            mlp_lin1_params = params['mlp_lin1']
            mlp_lin2_params = params['mlp_lin2']

        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        lin1_out = self.mlp_lin1(dist_emb, mlp_lin1_params)
        softplus_out = self.mlp_softplus(lin1_out)
        W = self.mlp_lin2(softplus_out, mlp_lin2_params) * C.view(-1, 1)
        v = self.lin(v, lin_params)
        e = v[j] * W
        return e


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(update_v, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1 = MetaLinearLayer(num_filters, hidden_channels)
        self.lin2 = MetaLinearLayer(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weights)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weights)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, e, edge_index, params=None):
        lin1_params = None
        lin2_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            lin1_params = params['lin1']
            lin2_params = params['lin2']

        _, i = edge_index
        out = scatter(e, i, dim=0)
        out = self.lin1(out, lin1_params)
        out = self.act(out)
        out = self.lin2(out, lin2_params)
        return v + out


class update_u(torch.nn.Module):
    def __init__(self, hidden_channels,out_channels = 1):
        super(update_u, self).__init__()
        self.lin1 = MetaLinearLayer(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = MetaLinearLayer(hidden_channels // 2, out_channels)

        # self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weights)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weights)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch, params=None):
        lin1_params = None
        lin2_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            lin1_params = params['lin1']
            lin2_params = params['lin2']

        v = self.lin1(v, lin1_params)
        v = self.act(v)
        v = self.lin2(v, lin2_params)
        u = scatter(v, batch, dim=0)
        return u


class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

