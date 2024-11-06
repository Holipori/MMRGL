"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import torch
import torch.nn as nn
from model.fusion import BAN, BUTD, MuTAN
from model.language_model import WordEmbedding, QuestionEmbedding,\
                                 QuestionSelfAttention
from model.relation_encoder import ImplicitRelationEncoder,\
                                   ExplicitRelationEncoder
from model.classifier import SimpleClassifier
from model.graph_att import GAttNet as GAT
import pandas as pd
import pickle
import numpy as np
import json
from tqdm import tqdm
import math
from torch.autograd import Function
from model.ggnn import GGNN
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, input):
        rms = torch.sqrt(torch.mean(input ** 2, dim=-1, keepdim=True))
        output = input / (rms + self.eps) * self.weight
        return output
class Self_attention(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, 2048)
        self.linear2 = nn.Linear(2048, d_model)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.num_layers = num_layers

    def forward(self, x, k ,v):
        for i in range(self.num_layers):
            residual = x
            x, _ = self.self_attn(x, k, v)
            x = self.norm1(x + residual)
            residual = x
            x = F.relu(self.linear1(x))
            x = self.dropout1(x)
            x = self.linear2(x)
            x = self.dropout2(x)
            x = self.norm2(x + residual)
        return x
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, out_size=2):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, out_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    # self.apply(init_weights)
    # self.iter_num = 0
    # self.alpha = 10
    # self.low = 0.0
    # self.high = 1.0
    # self.max_iter = 10000.0

    def forward(self, feat, alpha=1):
        # x = x * 1.0
        x = GradReverse.apply(feat, 1)
        # x.register_hook(grl_hook(alpha))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

class ReGAT(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, q_att, v_relation,
                 joint_embedding, classifier, glimpse, fusion, relation_type, args):
        super(ReGAT, self).__init__()
        self.name = "ReGAT_%s_%s" % (relation_type, fusion)
        self.relation_type = relation_type
        self.fusion = fusion
        self.dataset = dataset
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att
        self.v_relation = v_relation
        self.joint_embedding = joint_embedding
        self.classifier = classifier
        self.args = args

        if args.use_graph_classification:
            if args.use_pos_emb:
                pos_emb_dim = args.imp_pos_emb_dim
            else:
                pos_emb_dim = 0
            self.graph_classification_net = GAT(args.dir_num, 1, dataset.features.shape[-1], args.num_hid,
                                     nongt_dim=args.nongt_dim,
                                     label_bias=args.label_bias,
                                     num_heads=args.num_heads,
                                     pos_emb_dim=pos_emb_dim)
            if args.pure_classification:
                dim = dataset.num_ans_candidates
            else:
                dim = dataset.num_ans_candidates -1
            self.graph_classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                  dim, 0.5)
    def get_emb(self, tokens):
        w_emb = self.w_emb(tokens)
        q_emb_seq = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_emb_self_att = self.q_att(q_emb_seq)
        return q_emb_self_att

    def forward(self, v, b, q, implicit_pos_emb, sem_adj_matrix,
                spa_adj_matrix, labels, node_label, _, bbx_label):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        pos: [batch_size, num_objs, nongt_dim, emb_dim]
        sem_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]
        spa_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]

        return: logits, not probs
        """
        ## question embedding

        w_emb = self.w_emb(q)
        assert(not q.isnan().any())
        try:
            assert(not w_emb.isnan().any())
        except:
            print('a')
        q_emb_seq = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        assert(not q_emb_seq.isnan().any())
        q_emb_self_att = self.q_att(q_emb_seq)
        # q_emb_self_att = q_emb_seq[:,0,:]
        assert(not q_emb_self_att.isnan().any())
        # q_emb_self_att = q_emb_seq


        assert(not v.isnan().any())

        ## answer embedding
        if self.args.use_contrastive:
            answer_logits = self.get_emb(labels)
        else:
            answer_logits = torch.zeros(1).to(v.device)

        if self.args.use_graph_classification:
            v_graph = v.clone()
        # [batch_size, num_rois, out_dim]
        if self.relation_type == "semantic":
            v_emb, aff = self.v_relation.forward(v, sem_adj_matrix, q_emb_self_att)
        elif self.relation_type == "spatial":
            v_emb, aff = self.v_relation.forward(v, spa_adj_matrix, q_emb_self_att)
        else:  # implicit
            v_emb, aff = self.v_relation.forward(v, implicit_pos_emb, q_emb_self_att)
        assert( not v_emb.isnan().any())


        # classification here
        if self.args.use_graph_classification:
            imp_adj_mat = torch.autograd.Variable(
                torch.ones(
                    v.size(0), v.size(1), v.size(1), 1)).to(v.device)
            if self.args.use_pos_emb:
                out,_ = self.graph_classification_net(v_graph, imp_adj_mat, implicit_pos_emb)
            else:
                out,_ = self.graph_classification_net(v_graph, imp_adj_mat, None)
            out = torch.mean(out, 1)
            logits_graph = self.graph_classifier(out)
        else:
            logits_graph = torch.zeros(1).to(v.device)



        if self.fusion == "ban":
            joint_emb, att = self.joint_embedding(v_emb, q_emb_seq, b)
        elif self.fusion == "butd":
            q_emb = self.q_emb(w_emb)  # [batch, q_dim]
            joint_emb, att = self.joint_embedding(v_emb, q_emb)
        else:  # mutan
            joint_emb, att = self.joint_embedding(v_emb, q_emb_self_att)
        if self.classifier:
            logits = self.classifier(joint_emb)
        else:
            logits = joint_emb
        # logits = logits.softmax(dim=-1)

        # logits = answer_logits @ joint_emb.T
        assert( not logits.isnan().any())
        return logits, answer_logits, att, logits_graph

class my_Attention(nn.Module):
    def __init__(self, input_size):
        super(my_Attention, self).__init__()
        self.linear = nn.Linear(input_size, input_size // 2)
        self.projection = nn.Linear(input_size // 2, 1)

    def forward(self, input):
        x = self.linear(input)
        x = torch.tanh(x)
        x = self.projection(x)
        x = torch.softmax(x, dim=1)
        x = torch.sum(input * x, dim=1, keepdim=True)

        return x

class DualWeighting(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, q_att, v_relation_imp, v_relation_sem,
                 joint_embedding, classifier, glimpse, fusion, relation_type, args):
        super(DualWeighting, self).__init__()
        self.name = "DualWeighting_%s" % ( fusion)
        self.relation_type = relation_type
        self.fusion = fusion
        self.dataset = dataset
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att
        self.v_relation_imp = v_relation_imp
        self.v_relation_sem = v_relation_sem
        self.joint_embedding = joint_embedding
        self.classifier = classifier
        self.args = args
        self.att = my_Attention(1024)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)

        self.self_attention = Self_attention(1024, 8, 2)

        self.trans_fc = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.final_fc1 = nn.Linear(1024, 1024)
        self.final_fc2 = nn.Linear(1024, 1024)
        self.final_fc3 = nn.Linear(1024, 1024)
        self.final_fc4 = nn.Linear(1024, 1024)
        self.final_fc5 = nn.Linear(1024, 1024)
        self.final_fc6 = nn.Linear(1024, 1024)
        # self.final_fc7 = nn.Linear(1024*108, 1024*56)
        # self.final_fc7 = nn.Linear(1024*108, 1024*56)
        self.fc_q = nn.Linear(1024, 1024)
        self.fc_k = nn.Linear(1024, 1024)
        self.fc_v = nn.Linear(1024, 1024)
        self.softmax = nn.Softmax(dim=-1)

        self.v_fc = nn.Linear(1024, 1024)
        self.new_feat_matric_fc = nn.Linear(600, 1024)
        self.dropout = nn.Dropout(0.5)
        self.RMSnorm = RMSNorm(1024)

        self.ggnn = GGNN(args)

        # inital weight matrix
        # self.Wb = nn.Parameter(torch.randn(1024, 1024))
        # self.Wq = nn.Parameter(torch.randn(1024, 1024))
        # self.Wv = nn.Parameter(torch.randn(1024, 1024))
        # self.w_hv = nn.Parameter(torch.randn(1, 1024))
        # self.w_hq = nn.Parameter(torch.randn(1, 1024))

        self.loss_fn = nn.CrossEntropyLoss()
        self.net_D = AdversarialNetwork(1024, 1024, 2)

        self.disease_size = len(self.node_names()) - len(self.get_kg_ana_only())
        self.ana_size = len(self.get_kg_ana_only())
        self.fc = nn.Linear(1024 * self.args.n_node, 1024)
        self.fc_fc = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, self.disease_size)
        self.KG_matrix_feat = self.get_init_KG_matrix()
        self.semantic_kg = self.get_semantic_adj()
        self.bn = nn.BatchNorm1d(args.relation_dim)




        if args.use_graph_classification:
            if args.use_pos_emb:
                pos_emb_dim = args.imp_pos_emb_dim
            else:
                pos_emb_dim = 0
            self.graph_classification_net = GAT(args.dir_num, 1, dataset.features.shape[-1], args.num_hid,
                                     nongt_dim=args.nongt_dim,
                                     label_bias=args.label_bias,
                                     num_heads=args.num_heads,
                                     pos_emb_dim=pos_emb_dim)
            if args.pure_classification:
                dim = dataset.num_ans_candidates
            else:
                dim = dataset.num_ans_candidates -1
            self.graph_classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                  dim, 0.5)

    def get_kg_ana_only(self):
        # more simpler names
        kg_dict = {}
        # anatomical part
        kg_dict['right lung'] = 'Lung'
        kg_dict['right upper lung zone'] = 'Lung'
        kg_dict['right mid lung zone'] = 'Lung'
        kg_dict['right lower lung zone'] = 'Lung'
        kg_dict['right hilar structures'] = 'Lung'
        kg_dict['right apical zone'] = 'Lung'
        kg_dict['right costophrenic angle'] = 'Pleural'
        kg_dict['right hemidiaphragm'] = 'Pleural'  # probably
        kg_dict['left lung'] = 'Lung'
        kg_dict['left upper lung zone'] = 'Lung'
        kg_dict['left mid lung zone'] = 'Lung'
        kg_dict['left lower lung zone'] = 'Lung'
        kg_dict['left hilar structures'] = 'Lung'
        kg_dict['left apical zone'] = 'Lung'
        kg_dict['left costophrenic angle'] = 'Pleural'
        kg_dict['left hemidiaphragm'] = 'Pleural'  # probably

        kg_dict['trachea'] = 'Lung'
        kg_dict['right clavicle'] = 'Bone'
        kg_dict['left clavicle'] = 'Bone'
        kg_dict['aortic arch'] = 'Heart'
        kg_dict['upper mediastinum'] = 'Mediastinum'
        kg_dict['svc'] = 'Heart'
        kg_dict['cardiac silhouette'] = 'Heart'
        kg_dict['cavoatrial junction'] = 'Heart'
        kg_dict['right atrium'] = 'Heart'
        kg_dict['carina'] = 'Lung'

        return kg_dict
    def node_names(self):
        disease_lib_path = 'dataset_construction/lib/disease_lib_llm_full.csv'
        disease_lib = pd.read_csv(disease_lib_path)
        disease_names = disease_lib['official_name'].tolist()

        ana_names = list(self.get_kg_ana_only().keys())

        return disease_names+ana_names

    @torch.no_grad()
    def get_init_KG_matrix(self):
        node_names = self.node_names()

        # convert node_names into tokens using dictionary
        node_tokens = []
        n = self.dataset.dictionary.ntoken
        for i, name in enumerate(node_names):
            if name not in self.dataset.dictionary.word2idx:
                self.dataset.dictionary.add_word(name)
            node_tokens.append(self.dataset.dictionary.word2idx[name])

        self.w_emb = WordEmbedding(self.dataset.dictionary.ntoken, 300, .0, self.args.op)
        node_tokens = torch.tensor(node_tokens)
        w_emb = self.w_emb(node_tokens.unsqueeze(0))
        w_emb = w_emb.squeeze(0)
        return w_emb

    def get_kg(self):
        kg_dict = {}

        # anatomical part
        kg_dict = self.get_kg_ana_only()

        # disease part
        disease_lib_path = 'dataset_construction/lib/disease_lib_llm_full.csv'
        disease_lib = pd.read_csv(disease_lib_path)

        for i in range(len(disease_lib)):
            kg_dict[disease_lib['official_name'][i]] = disease_lib['location'][i]

        return kg_dict

    def get_directed_kg(self):

        path = '/home/xinyue/chatgpt/output/all_diseases_standardized4.json'
        diseases = json.load(open(path, 'r'))

        disease_lib_path = 'dataset_construction/lib/disease_lib_llm_full.csv'
        disease_lib = pd.read_csv(disease_lib_path)

        extracted_diseases_from_report = disease_lib['official_name'].values
        report_dis_dict = {dis: i for i, dis in enumerate(extracted_diseases_from_report)}

        kg_adj = np.zeros((len(extracted_diseases_from_report), len(extracted_diseases_from_report)))
        # for disease in tqdm(diseases):
        #     for ent in disease['entity']:
        #         if disease['entity'][ent]['infer'] != []:
        #             for target in disease['entity'][ent]['infer']:
        #                 kg_adj[report_dis_dict[ent]][report_dis_dict[target]] += 1

        # save kg_adj
        kg_adj_path = 'data/mimic/kg_infer_adj.npy'
        np.save(kg_adj_path, kg_adj)

        for i in range(len(kg_adj)):
            for j in range(len(kg_adj)):
                if kg_adj[i][j] > 100:
                    print(extracted_diseases_from_report[i], 'to', extracted_diseases_from_report[j], kg_adj[i][j])

        return kg_adj, report_dis_dict

    @torch.no_grad()
    def get_semantic_adj(self):
        '''

        '''
        use_infer = False

        node_labels = self.node_names()
        kg_ana = self.get_kg()
        # get small_adj
        with open('/home/xinyue/VQA_ReGat/data/mimic/GT_counting_adj.pkl', "rb") as tf:
            small_counting_adj = pickle.load(tf)
            for i in range(len(small_counting_adj)):
                small_counting_adj[i] = small_counting_adj[i] / small_counting_adj[i][i]
            small_adj = np.where(small_counting_adj > 0.18, 2, 0)  # set threshold to 0.2, label = 2.
        # get small_name2index
        path = 'data/mimic/mimic-cxr-2.0.0-chexpert.csv.gz'
        df = pd.read_csv(path)
        mimic_list = df.columns[2:16].values
        small_name2index = {key.lower(): i for i, key in enumerate(mimic_list)}
        # get kg_infer
        kg_infer, kg_infer_word2idx = self.get_directed_kg()
        kg_infer = np.where(kg_infer > 100, 3, 0)

        adj_matrix = torch.zeros([self.disease_size + self.ana_size, self.disease_size + self.ana_size])
        for i in range(len(node_labels)):
            for j in range(i, len(node_labels)):
                name1 = node_labels[i]
                name2 = node_labels[j]
                # semantic garph 1
                ana1 = kg_ana[name1]
                ana2 = kg_ana[name2]
                if isinstance(ana1, str):
                    ana1 = ana1.lower()
                if isinstance(ana2, str):
                    ana2 = ana2.lower()
                if i== j or (ana1 == ana2 and isinstance(ana1, str) and isinstance(ana2, str)):
                    # adj_matrix[i,j] = kg_idx[kg_dict[thing_classes[pred_classes[i]]]]
                    # adj_matrix[j, i] = adj_matrix[i,j]
                    if i < self.disease_size and j >= self.disease_size or i >= self.disease_size and j < self.disease_size or i == j:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1

                # semantic garph 2

                if type(node_labels[i]) != str and math.isnan(node_labels[i]):
                    pass
                elif type(node_labels[j]) != str and math.isnan(node_labels[j]):
                    pass
                elif node_labels[i].lower() in small_name2index and node_labels[j].lower() in small_name2index and i != j:
                    # small_adj is the 14x14 co-occurrence matrix
                    value = max(small_adj[
                                    small_name2index[node_labels[i].lower()], small_name2index[node_labels[j].lower()]], adj_matrix[i, j])
                    adj_matrix[i, j] = value
                    adj_matrix[j, i] = value


                # semantic garph 3
                if use_infer:
                    if name1 in kg_infer_word2idx and name2 in kg_infer_word2idx and i != j:
                        adj_matrix[i, j] = kg_infer[kg_infer_word2idx[name1]][kg_infer_word2idx[name2]]

        # transform into onehot
        adj_matrix = torch.nn.functional.one_hot(adj_matrix.long(), num_classes=4).float()[:,:,1:]

        return adj_matrix

    def get_emb(self, tokens):
        w_emb = self.w_emb(tokens)
        q_emb_seq = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_emb_self_att = self.q_att(q_emb_seq)
        return q_emb_self_att

    def forward(self, v, b, q, implicit_pos_emb, sem_adj_matrix,
                spa_adj_matrix, labels, node_label, sem_region_feats, bbx_label):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        pos: [batch_size, num_objs, nongt_dim, emb_dim]
        sem_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]
        spa_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]

        return: logits, not probs
        """
        ## question embedding

        w_emb = self.w_emb(q)
        assert(not q.isnan().any())
        assert(not w_emb.isnan().any())
        q_emb_seq = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        assert(not q_emb_seq.isnan().any())
        q_emb_self_att = self.q_att(q_emb_seq)
        # q_emb_self_att = q_emb_seq[:,0,:]
        assert(not q_emb_self_att.isnan().any())
        # q_emb_self_att = q_emb_seq


        assert(not v.isnan().any())


        if self.args.ggnn:
            pass
            # 1, input feature x, annatations ana, adj matrix
            out = self.ggnn(v, bbx_label, sem_adj_matrix, self.KG_matrix_feat.to(v.device))
            logits = self.classifier(out)


            loss = torch.tensor(0).to(logits.device)
            return logits, loss, loss, loss
        else:
            ## calculate graph node vector
            if not self.args.sem_region_feature:
                # 1, calculate the final feature vector for the image
                # v_emb, aff = self.v_relation_imp.forward(v, implicit_pos_emb, q_emb_self_att)
                feat_matrix = self.v_fc(v)

                vec = self.fc(v.view(-1, 1024*self.args.n_node))

                # 2, predict the disease label, and get the graph node vector (N x 1, N = 30 + 26). (26 are all assinged to 1.)
                vec = nn.functional.relu(vec)
                node_logits = self.fc2(vec)
                node_logits = node_logits.softmax(dim=-1)
                addition = torch.ones(node_logits.size(0), self.ana_size).to(node_logits.device)
                logits = torch.cat((node_logits, addition), dim=-1)

                node_pred = node_logits

                # # 3, get NxD matrix, N is the number of nodes, D is the dimension of the feature vector
                # feat_matrix = torch.matmul(pred.unsqueeze(2), vec.unsqueeze(1)) # pred is the Feature Vector in the TMI paper.
            else:
                # for each node, get the feature vector
                # 1, for the first 30 disease nodes, get the cadgram of predicting the disease label
                v_disease = sem_region_feats[0]
                # v_disease = v[:,26:,:]
                v_disease = v_disease * sem_region_feats[1].unsqueeze(-1)
                # 2, for the last 26 anatomy nodes, use the region feature
                v_ana = v[:,:26,:]
                feat_matrix = torch.cat((v_disease, v_ana), dim=1)
                node_pred = torch.zeros(1).to(v.device)

                disease_logits = sem_region_feats[1]
                ana_logits = torch.ones(disease_logits.shape[0], 26).to(v_emb.device)
                logits = torch.cat((disease_logits, ana_logits), dim=-1)

            ## calculate the KG embedding
            # 0, setting a KG embedding matrix (N x K) in self.init()

            # 1, GCN calculatation, get the KG embedding matrix (N x D)
            v_emb, aff = self.v_relation_sem.forward(self.KG_matrix_feat.unsqueeze(0).to(feat_matrix.device), self.semantic_kg.unsqueeze(0).to(feat_matrix.device))
            # add batch normailization
            v_emb = self.bn(v_emb.squeeze(0)).unsqueeze(0)
            # combine these two
            v_emb = v_emb.clone().expand(feat_matrix.size(0), v_emb.shape[1], -1)


            # v_emb = v_emb * logits.unsqueeze(-1)
            # 1, get the combine feature maps (N x D)
            loss = torch.tensor(0).to(v_emb.device)
            if self.args.cross_attention:
                raise NotImplementedError
                if self.args.testing_code == 1:
                    feat_matrix = torch.cat((feat_matrix, v_emb), dim=-1)
                    feat_matrix = self.transformer(feat_matrix)
                    feat_matrix = self.final_fc1(feat_matrix[:, :, :1024]) + self.final_fc2(feat_matrix[:, :, 1024:])

            else:
                if self.args.testing_code == 1:
                    feat_matrix = self.final_fc1(feat_matrix) + self.final_fc2(v_emb)
                elif self.args.testing_code == 2:
                    loss = torch.mean(torch.abs(feat_matrix - v_emb))

                    feat_matrix = self.final_fc1(feat_matrix) + self.final_fc2(v_emb)
                elif self.args.testing_code == 3:
                    pred1 = self.net_D(feat_matrix)
                    pred2 = self.net_D(v_emb)
                    preds = torch.cat((pred1, pred2), dim=0)
                    targets = torch.cat((torch.ones(pred1.size(0), 1).to(pred1.device), torch.zeros(pred2.size(0), 1).to(pred2.device)), dim=0)
                    targets = torch.zeros(preds.size(0), 2).to(preds.device).scatter_(1, targets.long(), 1).long()
                    loss = self.loss_fn(preds, targets)

                    feat_matrix = self.final_fc1(feat_matrix) + self.final_fc2(v_emb)
                elif self.args.testing_code == 4:
                    v_visual = self.fc(v.view(-1, 1024 * self.args.n_node))
                    v_visual = self.bn1(v_visual)
                    v_visual = nn.functional.relu(v_visual)
                    v_visual = self.dropout(v_visual)
                    v_visual = v_visual.unsqueeze(1).expand(-1,v_emb.shape[1],1024)
                    v_visual = v_visual * logits.unsqueeze(-1)
                    new_v = torch.cat((v_visual, v_emb), dim=1)
                    new_v = self.transformer(new_v.transpose(0, 1)).transpose(0, 1)
                    new_v = self.final_fc3(new_v[:, :56,:]) + self.final_fc4(new_v[:, 56:, :])
                    new_v = self.bn2(new_v.transpose(1,2)).transpose(1,2)
                    new_v = nn.functional.relu(new_v)
                    v_visual = self.dropout(new_v)

                    feat_matrix = torch.cat((feat_matrix, new_v), dim=1)
                    feat_matrix = torch.mean(feat_matrix, dim=1).unsqueeze(1).expand(-1, v_emb.shape[1], -1)
                elif self.args.testing_code == 42:
                    v_visual = self.fc(v.view(-1, 1024 * self.args.n_node))
                    v_visual = self.bn1(v_visual)
                    v_visual = nn.functional.relu(v_visual)
                    v_visual = self.dropout(v_visual)
                    v_visual = v_visual.unsqueeze(1).expand(-1, v_emb.shape[1], 1024)
                    v_visual = v_visual * logits.unsqueeze(-1)
                    new_v = torch.cat((v_visual, v_emb), dim=1)
                    # new_v = self.transformer(new_v.transpose(0, 1)).transpose(0, 1)
                    new_v = self.final_fc3(new_v[:, :56, :]) + self.final_fc4(new_v[:, 56:, :])
                    new_v = self.bn2(new_v.transpose(1,2)).transpose(1,2)
                    new_v = nn.functional.relu(new_v)
                    new_v = self.dropout(new_v)

                    feat_matrix = torch.cat((feat_matrix, new_v), dim=1)
                    feat_matrix = torch.mean(feat_matrix, dim=1).unsqueeze(1)

                elif self.args.testing_code == 5:
                    v_visual = self.fc(v.view(-1, 1024 * self.args.n_node))
                    v_visual = self.bn1(v_visual)
                    v_visual = nn.functional.relu(v_visual)
                    v_visual = self.dropout(v_visual)
                    v_visual = v_visual.unsqueeze(1).expand(-1, v_emb.shape[1], 1024)
                    v_visual = v_visual * logits.unsqueeze(-1)


                    feat_matrix = self.self_attention(v_visual, v_emb, v_emb)
                elif self.args.testing_code == 6:
                    v_visual = self.fc(v.view(-1, 1024 * self.args.n_node))
                    v_visual = self.bn1(v_visual)
                    v_visual = nn.functional.relu(v_visual)
                    v_visual = self.dropout(v_visual)
                    v_visual = v_visual.unsqueeze(1).expand(-1, v_emb.shape[1], 1024)
                    v_visual = v_visual * logits.unsqueeze(-1)

                    feat_matrix = self.self_attention(v_emb, v_visual, v_visual)
                elif self.args.testing_code == 7:
                    kg = self.KG_matrix_feat.unsqueeze(0).to(feat_matrix.device).expand(feat_matrix.size(0), -1, -1)
                    new_feat_matrix = kg * logits.unsqueeze(-1)
                    new_feat_matrix = self.new_feat_matric_fc(new_feat_matrix)
                    new_feat_matrix = nn.functional.relu(new_feat_matrix)
                    new_feat_matrix = self.dropout(new_feat_matrix)

                    word_feat_matrix = torch.cat((feat_matrix, v_emb, new_feat_matrix), dim=1)
                    word_feat_matrix = self.RMSnorm(word_feat_matrix)
                    feat_matrix = self.transformer(word_feat_matrix.transpose(0,1)).transpose(0,1)

                    feat_matrix = torch.mean(feat_matrix, dim=1).unsqueeze(1)
                elif self.args.testing_code == 8:
                    kg = self.KG_matrix_feat.unsqueeze(0).to(feat_matrix.device).expand(feat_matrix.size(0), -1, -1)
                    new_feat_matrix = kg * logits.unsqueeze(-1)
                    new_feat_matrix = self.new_feat_matric_fc(new_feat_matrix)

                    feat_matrix = self.self_attention(new_feat_matrix, v_emb, v_emb)

                    feat_matrix = torch.mean(feat_matrix, dim=1).unsqueeze(1)
                elif self.args.testing_code == 9:
                    kg = self.KG_matrix_feat.unsqueeze(0).to(feat_matrix.device).expand(feat_matrix.size(0), -1, -1)
                    new_feat_matrix = kg * logits.unsqueeze(-1)
                    new_feat_matrix = self.new_feat_matric_fc(new_feat_matrix)
                    new_feat_matrix = nn.functional.relu(new_feat_matrix)
                    new_feat_matrix = self.dropout(new_feat_matrix)

                    feat_matrix = self.self_attention(new_feat_matrix, torch.cat((feat_matrix, v_emb), dim=1), torch.cat((feat_matrix, v_emb), dim=1))

                    feat_matrix = torch.mean(feat_matrix, dim=1).unsqueeze(1)
                elif self.args.testing_code == 10:
                    kg = self.KG_matrix_feat.unsqueeze(0).to(feat_matrix.device).expand(feat_matrix.size(0), -1, -1)
                    new_feat_matrix = kg * logits.unsqueeze(-1)
                    new_feat_matrix = self.new_feat_matric_fc(new_feat_matrix)

                    word_feat_matrix = torch.cat((v_emb, new_feat_matrix), dim=1)
                    feat_matrix = self.transformer(word_feat_matrix.transpose(0,1)).transpose(0,1)

                    feat_matrix = torch.mean(feat_matrix, dim=1).unsqueeze(1)




            # 2, add question embedding (N x (D+D))
            if self.fusion == "ban":
                feat_matrix, att = self.joint_embedding(feat_matrix, q_emb_seq, b)
            elif self.fusion == "butd":
                q_emb = self.q_emb(w_emb)  # [batch, q_dim]
                feat_matrix, att = self.joint_embedding(feat_matrix, q_emb)
            else:  # mutan
                feat_matrix, att = self.joint_embedding(feat_matrix, q_emb_self_att)

            # feat_matrix = torch.cat((feat_matrix, q_emb_self_att.unsqueeze(1).expand(-1, self.disease_size + self.ana_size, -1)), dim=-1)

            # 3, attention, then get the final feature vector (1 x D)
            # feat_matrix = self.att(feat_matrix)


            # 4, predict the final answer (1 x 143)
            logits = self.classifier(feat_matrix)





            assert( not logits.isnan().any())
            return logits, loss, att, node_pred


def build_regat(dataset, args):
    print("Building ReGAT model with %s relation and %s fusion method" %
          (args.relation_type, args.fusion))
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600,
                              args.num_hid, 1, False, .0)
    q_att = QuestionSelfAttention(args.num_hid, .2)

    if args.relation_type == "semantic":
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.sem_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    elif args.relation_type == "my_semantic":
        v_relation_imp = ImplicitRelationEncoder(
            dataset.v_dim, args.num_hid, args.relation_dim,
            args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
            num_heads=args.num_heads, num_steps=args.num_steps,
            residual_connection=args.residual_connection,
            label_bias=args.label_bias,
            use_q=args.use_q)
        v_relation_sem = ExplicitRelationEncoder(
                        600, 0, args.relation_dim,
                        args.dir_num, args.sem_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    elif args.relation_type == "spatial":
        v_relation = ExplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.spa_label_num,
                        num_heads=args.num_heads,
                        num_steps=args.num_steps, nongt_dim=args.nongt_dim,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)
    elif args.relation_type == "implicit":
        v_relation = ImplicitRelationEncoder(
                        dataset.v_dim, args.num_hid, args.relation_dim,
                        args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
                        num_heads=args.num_heads, num_steps=args.num_steps,
                        residual_connection=args.residual_connection,
                        label_bias=args.label_bias)

    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                  dataset.num_ans_candidates, 0.5)
    gamma = 0
    if args.fusion == "ban":
        joint_embedding = BAN(args.relation_dim, args.num_hid, args.ban_gamma)
        gamma = args.ban_gamma
    elif args.fusion == "butd":
        joint_embedding = BUTD(args.relation_dim, args.num_hid, args.num_hid)
    else:
        joint_embedding = MuTAN(args.relation_dim, args.num_hid,
                                dataset.num_ans_candidates, args.mutan_gamma)
        gamma = args.mutan_gamma
        classifier = None
    if args.relation_type == 'my_semantic':
        return DualWeighting(dataset, w_emb, q_emb, q_att, v_relation_imp, v_relation_sem, joint_embedding,
                 classifier, gamma, args.fusion, args.relation_type, args)
    return ReGAT(dataset, w_emb, q_emb, q_att, v_relation, joint_embedding,
                 classifier, gamma, args.fusion, args.relation_type, args)

