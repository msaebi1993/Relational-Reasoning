"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Fact scoring networks.
 Code adapted from https://github.com/TimDettmers/ConvE/blob/master/model.py
"""

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import collections
import numpy as np

class TripleE(nn.Module):
    def __init__(self, args, num_entities):
        super(TripleE, self).__init__()
        conve_args = copy.deepcopy(args)
        conve_args.model = 'conve'
        self.conve_nn = ConvE(conve_args, num_entities)
        conve_state_dict = torch.load(args.conve_state_dict_path)
        conve_nn_state_dict = get_conve_nn_state_dict(conve_state_dict)
        self.conve_nn.load_state_dict(conve_nn_state_dict)

        complex_args = copy.deepcopy(args)
        complex_args.model = 'complex'
        self.complex_nn = ComplEx(complex_args)

        distmult_args = copy.deepcopy(args)
        distmult_args.model = 'distmult'
        self.distmult_nn = DistMult(distmult_args)

    def forward(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        distmult_kg = secondary_kgs[1]
        return (self.conve_nn.forward(e1, r, conve_kg)
                + self.complex_nn.forward(e1, r, complex_kg)
                + self.distmult_nn.forward(e1, r, distmult_kg)) / 3

    def forward_fact(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        distmult_kg = secondary_kgs[1]
        return (self.conve_nn.forward_fact(e1, r, conve_kg)
                + self.complex_nn.forward_fact(e1, r, complex_kg)
                + self.distmult_nn.forward_fact(e1, r, distmult_kg)) / 3

class HyperE(nn.Module):
    def __init__(self, args, num_entities):
        super(HyperE, self).__init__()
        self.conve_nn = ConvE(args, num_entities)
        conve_state_dict = torch.load(args.conve_state_dict_path)
        conve_nn_state_dict = get_conve_nn_state_dict(conve_state_dict)
        self.conve_nn.load_state_dict(conve_nn_state_dict)

        complex_args = copy.deepcopy(args)
        complex_args.model = 'complex'
        self.complex_nn = ComplEx(complex_args)

    def forward(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        return (self.conve_nn.forward(e1, r, conve_kg)
                + self.complex_nn.forward(e1, r, complex_kg)) / 2

    def forward_fact(self, e1, r, e2, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        return (self.conve_nn.forward_fact(e1, r, e2, conve_kg)
                + self.complex_nn.forward_fact(e1, r, e2, complex_kg)) / 2

class ComplEx(nn.Module):
    def __init__(self, args):
        self.entity_img_type_embeddings= None
        self.entity_img_type_embeddings= None
        super(ComplEx, self).__init__()

    def forward(self, e1, r, kg):
        def dist_mult(E1, R, E2):
            return torch.mm(E1 * R, E2.transpose(1, 0))

        E1_real = kg.get_entity_embeddings(e1)
        R_real = kg.get_relation_embeddings(r)
        E2_real = kg.get_all_entity_embeddings()
        E1_img = kg.get_entity_img_embeddings(e1)
        R_img = kg.get_relation_img_embeddings(r)
        E2_img = kg.get_all_entity_img_embeddings()

        rrr = dist_mult(R_real, E1_real, E2_real)
        rii = dist_mult(R_real, E1_img, E2_img)
        iri = dist_mult(R_img, E1_real, E2_img)
        iir = dist_mult(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir
        S = F.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2, kg):
        def dist_mult_fact(E1, R, E2):
            return torch.sum(E1 * R * E2, dim=1, keepdim=True)
        
        E1_real = kg.get_entity_embeddings(e1)
        R_real = kg.get_relation_embeddings(r)
        E2_real = kg.get_entity_embeddings(e2)
        E1_img = kg.get_entity_img_embeddings(e1)
        R_img = kg.get_relation_img_embeddings(r)
        E2_img = kg.get_entity_img_embeddings(e2)


        rrr = dist_mult_fact(R_real, E1_real, E2_real)
        rii = dist_mult_fact(R_real, E1_img, E2_img)
        iri = dist_mult_fact(R_img, E1_real, E2_img)
        iir = dist_mult_fact(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir
        S = F.sigmoid(S)
        return S

class ConvE(nn.Module):
    def __init__(self, args, num_entities):
        super(ConvE, self).__init__()
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        assert(args.emb_2D_d1 * args.emb_2D_d2 == args.entity_dim)
        assert(args.emb_2D_d1 * args.emb_2D_d2 == args.relation_dim)
        self.emb_2D_d1 = args.emb_2D_d1
        self.emb_2D_d2 = args.emb_2D_d2
        self.num_out_channels = args.num_out_channels
        self.w_d = args.kernel_size
        self.HiddenDropout = nn.Dropout(args.hidden_dropout_rate)
        self.FeatureDropout = nn.Dropout(args.feat_dropout_rate)

        # stride = 1, padding = 0, dilation = 1, groups = 1
        self.conv1 = nn.Conv2d(1, self.num_out_channels, (self.w_d, self.w_d), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_out_channels)
        self.bn2 = nn.BatchNorm1d(self.entity_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        h_out = 2 * self.emb_2D_d1 - self.w_d + 1
        w_out = self.emb_2D_d2 - self.w_d + 1
        self.feat_dim = self.num_out_channels * h_out * w_out
        self.fc = nn.Linear(self.feat_dim, self.entity_dim)

    def forward(self, e1, r, kg):
        E1 = kg.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = kg.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = kg.get_all_entity_embeddings()
        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.mm(X, E2.transpose(1, 0))
        X += self.b.expand_as(X)

        S = F.sigmoid(X)
        return S

    def forward_fact(self, e1, r, e2, kg):
        """
        Compute network scores of the given facts.
        :param e1: [batch_size]
        :param r:  [batch_size]
        :param e2: [batch_size]
        :param kg:
        """

        E1 = kg.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = kg.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = kg.get_entity_embeddings(e2)

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.matmul(X.unsqueeze(1), E2.unsqueeze(2)).squeeze(2)
        X += self.b[e2].unsqueeze(1)

        S = F.sigmoid(X)
        return S

class DistMult(nn.Module):
    def __init__(self, args):
        super(DistMult, self).__init__()

    def forward(self, e1, r, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_all_entity_embeddings()
        S = torch.mm(E1 * R, E2.transpose(1, 0))
        S = F.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)
        S = torch.sum(E1 * R * E2, dim=1, keepdim=True)
        S = F.sigmoid(S)
        return S

def get_conve_nn_state_dict(state_dict):
    conve_nn_state_dict = {}
    for param_name in ['mdl.b', 'mdl.conv1.weight', 'mdl.conv1.bias', 'mdl.bn0.weight', 'mdl.bn0.bias',
                       'mdl.bn0.running_mean', 'mdl.bn0.running_var', 'mdl.bn1.weight', 'mdl.bn1.bias',
                       'mdl.bn1.running_mean', 'mdl.bn1.running_var', 'mdl.bn2.weight', 'mdl.bn2.bias',
                       'mdl.bn2.running_mean', 'mdl.bn2.running_var', 'mdl.fc.weight', 'mdl.fc.bias']:
        conve_nn_state_dict[param_name.split('.', 1)[1]] = state_dict['state_dict'][param_name]

    return conve_nn_state_dict

def get_conve_kg_state_dict(state_dict,data_dir,use_type,agg_method):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kg_state_dict = dict()
    dim=100
    lin=nn.Linear(dim*2, dim).to(device)

    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight']:
        key=param_name.split('.', 1)[1]
        kg_state_dict[key] = state_dict['state_dict'][param_name]

    if use_type=='type':
        entity_type_embeddings,_ = get_type_embeddings(agg_method,data_dir,'conve',dim)
        if True:
        #with torch.no_grad():
            key1='entity_embeddings.weight'
            for e in range(len(kg_state_dict[key1])-1):
                tmp=kg_state_dict[key1][e]#[:dim]
                concat_emb=torch.cat((tmp,entity_type_embeddings[e]),dim=-1)
                concat_emb=concat_emb.to(device)
                kg_state_dict[key1][e]= lin(concat_emb)


    return kg_state_dict

def get_type_embeddings(fcn,data_dir,emb_method,dim):
    #load type_embeddings:
    if emb_method=='complex':
        with open(os.path.join(data_dir, 'complex_'+str(dim)+'_entity_type_embs_real_'+fcn+'.pkl'), 'rb') as f:
            tmp_type_dict= pickle.load(f)
            od = collections.OrderedDict(sorted(tmp_type_dict.items()))
            entity_type_embeddings= torch.cuda.FloatTensor(np.array(list(od.values())).astype('float64'))

        with open(os.path.join(data_dir, 'complex_'+str(dim)+'_entity_type_embs_img_'+fcn+'.pkl'), 'rb') as f:
            tmp_type_dict= pickle.load(f)
            od = collections.OrderedDict(sorted(tmp_type_dict.items()))
            entity_img_type_embeddings= torch.cuda.FloatTensor(np.array(list(od.values())).astype('float64'))
        return entity_type_embeddings,entity_img_type_embeddings

    elif emb_method=='distmult':
        with open(os.path.join(data_dir, 'distmult_'+str(dim)+'_entity_type_embs_'+fcn+'.pkl'), 'rb') as f:
            tmp_type_dict= pickle.load(f)
            od = collections.OrderedDict(sorted(tmp_type_dict.items()))
            entity_type_embeddings= torch.cuda.FloatTensor(np.array(list(od.values())).astype('float64'))
        return entity_type_embeddings,0

    elif emb_method=='conve':
        with open(os.path.join(data_dir, 'conve_'+str(dim)+'_entity_type_embs_'+fcn+'.pkl'), 'rb') as f:
            tmp_type_dict= pickle.load(f)
            od = collections.OrderedDict(sorted(tmp_type_dict.items()))
            entity_type_embeddings= torch.cuda.FloatTensor(np.array(list(od.values())).astype('float64'))
    return entity_type_embeddings,0

def get_complex_kg_state_dict(args,state_dict,data_dir,use_type,agg_method,lin_cat):
    parallel=args.parallel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kg_state_dict = dict()
    dim=100
    lin=nn.Linear(dim*2, dim).to(device)


    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight',
                       'kg.entity_img_embeddings.weight', 'kg.relation_img_embeddings.weight']:
        key=param_name.split('.', 1)[1]
        if lin_cat=='lin':
            kg_state_dict[key]=state_dict['state_dict'][param_name]
        else:
            kg_state_dict[key]= F.pad(input=state_dict['state_dict'][param_name], pad=(0, dim), mode='constant', value=0)

    if use_type=='type':
        entity_type_embeddings,entity_img_type_embeddings= get_type_embeddings(agg_method,data_dir,'complex',dim)
        key1='entity_embeddings.weight'
        for e in range(len(kg_state_dict[key1])-1):
            tmp=kg_state_dict[key1][e][:dim]
            concat_emb=torch.cat((tmp,entity_type_embeddings[e]),dim=-1)
            if lin_cat=='lin':
                concat_emb=concat_emb.to(device)
                kg_state_dict[key1][e]=lin(concat_emb)
            else:
                kg_state_dict[key1][e]= concat_emb

        key2='entity_img_embeddings.weight'
        for e in range(len(kg_state_dict[key2])-1):
            tmp=kg_state_dict[key2][e][:dim]
            concat_emb=torch.cat((tmp,entity_img_type_embeddings[e]),dim=-1)
            if lin_cat=='lin':
                concat_emb=concat_emb.to(device)
                kg_state_dict[key2][e]=lin(concat_emb)
            else:
                kg_state_dict[key1][e]= concat_emb

    return kg_state_dict

def get_distmult_kg_state_dict(args,state_dict,data_dir,use_type,agg_method,lin_cat):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel=args.parallel
    kg_state_dict = dict()
    dim=100
    lin=nn.Linear(dim*2, dim).to(device)

    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight']:
        key0=param_name.split('.', 1)[1]
        if parallel==1:
            a=key0.split('.')
            key=a[0]+'.module.'+a[1]
        else:
            key=key0
        if lin_cat=='lin':
            kg_state_dict[key] = state_dict['state_dict'][param_name]
        else:
            kg_state_dict[key]= F.pad(input=state_dict['state_dict'][param_name], pad=(0, dim), mode='constant', value=0)

    if use_type=='type':
        entity_type_embeddings,_ = get_type_embeddings(agg_method,data_dir,'distmult',dim)
        if parallel==1:
            key1='entity_embeddings.module.weight'
        else:
            key1='entity_embeddings.weight'

        for e in range(len(kg_state_dict[key1])-1):
            tmp=kg_state_dict[key1][e][:dim]
            concat_emb= torch.cat((tmp,entity_type_embeddings[e]),dim=-1)
            kg_state_dict[key1][e]= concat_emb
            if lin_cat=='lin':
                concat_emb=concat_emb.to(device)
                kg_state_dict[key1][e]= lin(concat_emb)
            else:
                kg_state_dict[key1][e]= concat_emb

    return kg_state_dict



