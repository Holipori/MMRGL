import torch
import torch.nn as nn


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A = A.reshape(A.shape[0], A.shape[1], -1)
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        # A_out = A[:, :, self.n_node*self.n_edge_types:]
        A_t = A.transpose(1, 2)
        A_1 = A_t[:, :A_in.shape[1], :]
        A_2 = A_t[:, A_in.shape[1]:2*A_in.shape[1], :]
        A_3 = A_t[:, -A_in.shape[1]:, :]
        A_out = torch.cat((A_1, A_2, A_3), 2)

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output

class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()

        assert (opt.state_dim >= opt.annotation_dim,  \
                'state_dim must be no less than annotation_dim')

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps
        self.num_ans_candidates = opt.num_ans_candidates
        self.KG_dim = opt.KG_dim
        self.importance_net = nn.Linear(self.state_dim + self.KG_dim, 1)

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, self.state_dim)
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A, KG_feat, topk = 2):
        # annotation = annotation.unsqueeze(2)
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)

            expand_prop_state = prop_state.unsqueeze(2).expand(prop_state.shape[0], self.n_node, KG_feat.shape[0], self.state_dim)
            expand_KG_feat = KG_feat.unsqueeze(0).unsqueeze(1).expand(prop_state.shape[0], self.n_node, KG_feat.shape[0], self.KG_dim)
            concat_state = torch.cat((expand_prop_state, expand_KG_feat), 3)
            importance = self.importance_net(concat_state.reshape(-1, self.state_dim + self.KG_dim)).reshape(prop_state.shape[0], self.n_node, KG_feat.shape[0])
            top_node_idx = torch.topk(importance, topk, dim=2)[1]

        # annotation one hot. 64x52 -> 64x52x56
        annotation_onehot = torch.zeros(annotation.shape[0], annotation.shape[1], self.annotation_dim)
        annotation_onehot = annotation_onehot.to(prop_state.device)
        annotation_onehot.scatter_(2, annotation.unsqueeze(2), 1)


        join_state = torch.cat((prop_state, annotation_onehot), 2)

        output = self.out(join_state)
        output = output.sum(1).squeeze(1)

        return output