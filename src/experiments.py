#!/usr/bin/env python3

"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Experiment Portal.
"""

import copy
import itertools
import numpy as np
import os, sys
import random

import torch

from src.parse_args import parser
from src.parse_args import args
import src.data_utils as data_utils
import src.eval
from src.hyperparameter_range import hp_range
from src.knowledge_graph import KnowledgeGraph
from src.emb.fact_network import ComplEx, ConvE, DistMult
from src.emb.fact_network import get_conve_kg_state_dict, get_complex_kg_state_dict, get_distmult_kg_state_dict
from src.emb.emb import EmbeddingBasedMethod
from src.rl.graph_search.pn import GraphSearchPolicy
from src.rl.graph_search.pg import PolicyGradient
from src.rl.graph_search.rs_pg import RewardShapingPolicyGradient
from src.utils.ops import flatten

parallel=args.parallel
print(parallel)
#torch.cuda.set_device(args.gpu)
if parallel==1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
else:
    torch.cuda.set_device(args.gpu)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def process_data():
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, args.test, args.add_reverse_relations)

def initialize_model_directory(args, random_seed=None):
    # add model parameter info to model directory
    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    reverse_edge_tag = '-RV' if args.add_reversed_training_edges else ''
    entire_graph_tag = '-EG' if args.train_entire_graph else ''
    if args.xavier_initialization:
        initialization_tag = '-xavier'
    elif args.uniform_entity_initialization:
        initialization_tag = '-uniform'
    else:
        initialization_tag = ''

    # Hyperparameter signature
    if args.model in ['rule']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.baseline,
            args.entity_dim,
            args.relation_dim,
            args.history_num_layers,
            args.learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.bandwidth,
            args.beta
        )
    elif args.model.startswith('point'):
        if args.baseline == 'avg_reward':
            print('* Policy Gradient Baseline: average reward')
        elif args.baseline == 'avg_reward_normalized':
            print('* Policy Gradient Baseline: average reward baseline plus normalization')
        else:
            print('* Policy Gradient Baseline: None')
        if args.action_dropout_anneal_interval < 1000:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.action_dropout_anneal_factor,
                args.action_dropout_anneal_interval,
                args.bandwidth,
                args.beta
            )
            if args.mu != 1.0:
                hyperparam_sig += '-{}'.format(args.mu)
        else:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.bandwidth,
                args.beta
            )
        if args.reward_shaping_threshold > 0:
            hyperparam_sig += '-{}'.format(args.reward_shaping_threshold)
    elif args.model == 'distmult':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model == 'complex':
        hyperparam_sig = '{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.emb_dropout_rate,
            args.label_smoothing_epsilon
        )
    elif args.model in ['conve', 'hypere', 'triplee']:
        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.entity_dim,
            args.relation_dim,
            args.learning_rate,
            args.num_out_channels,
            args.kernel_size,
            args.emb_dropout_rate,
            args.hidden_dropout_rate,
            args.feat_dropout_rate,
            args.label_smoothing_epsilon
        )
    else:
        raise NotImplementedError

    model_sub_dir = '{}-{}{}{}{}-{}'.format(
        dataset,
        args.model,
        reverse_edge_tag,
        entire_graph_tag,
        initialization_tag,
        hyperparam_sig
    )
    if args.model == 'set':
        model_sub_dir += '-{}'.format(args.beam_size)
        model_sub_dir += '-{}'.format(args.num_paths_per_entity)
    if args.relation_only:
        model_sub_dir += '-ro'
    elif args.relation_only_in_path:
        model_sub_dir += '-rpo'
    elif args.type_only:
        model_sub_dir += '-to'

    if args.use_type=='type':
        model_sub_dir += '-type-concat'
    else:
        model_sub_dir += '-notype'

    if args.agg_method:
        model_sub_dir += '-{}'.format(args.agg_method)

    if args.prune_method:
        model_sub_dir += '-{}'.format(args.prune_method)
    else:
        model_sub_dir += '-page_rank'

    if args.neigh_encoder:
        model_sub_dir += '-{}'.format(args.neigh_encoder)
    else:
        model_sub_dir += '-nEnc0'

    if args.e_r_trainable=='False':
        model_sub_dir += '-noTrain'

    if args.test:
        model_sub_dir += '-test'

    if random_seed:
        model_sub_dir += '.{}'.format(random_seed)

    model_dir = os.path.join(model_root_dir, model_sub_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('Model directory created: {}'.format(model_dir))
    else:
        print('Model directory exists: {}'.format(model_dir))

    args.model_dir = model_dir

def construct_model(args):
    """
    Construct NN graph.
    """
    kg = KnowledgeGraph(args)
    if args.model.endswith('.gc'):
        kg.load_fuzzy_facts()

    if args.model in ['point', 'point.gc']:
        pn = GraphSearchPolicy(args)
        lf = PolicyGradient(args, kg, pn)
    elif args.model.startswith('point.rs'):
        pn = GraphSearchPolicy(args)
        fn_model = args.model.split('.')[2]
        fn_args = copy.deepcopy(args)
        fn_args.model = fn_model
        fn_args.relation_only = False
        if fn_model == 'complex':
            fn = ComplEx(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'distmult':
            fn = DistMult(fn_args)
            fn_kg = KnowledgeGraph(fn_args)
        elif fn_model == 'conve':
            fn = ConvE(fn_args, kg.num_entities)
            fn_kg = KnowledgeGraph(fn_args)
        lf = RewardShapingPolicyGradient(args, kg, pn, fn_kg, fn)
    elif args.model == 'complex':
        fn = ComplEx(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'distmult':
        fn = DistMult(args)
        lf = EmbeddingBasedMethod(args, kg, fn)
    elif args.model == 'conve':
        fn = ConvE(args, kg.num_entities)
        lf = EmbeddingBasedMethod(args, kg, fn)
    else:
        raise NotImplementedError
    return lf

def train(lf):
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges)
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    lf.run_train(train_data, dev_data)

def inference(lf):
    lf.batch_size = args.dev_batch_size
    lf.eval()
    if args.model == 'hypere':
        conve_kg_state_dict = get_conve_kg_state_dict(torch.load(args.conve_state_dict_path))
        lf.kg.load_state_dict(conve_kg_state_dict)
        secondary_kg_state_dict = get_complex_kg_state_dict(torch.load(args.complex_state_dict_path))
        lf.secondary_kg.load_state_dict(secondary_kg_state_dict)
    elif args.model == 'triplee':
        conve_kg_state_dict = get_conve_kg_state_dict(torch.load(args.conve_state_dict_path))
        lf.kg.load_state_dict(conve_kg_state_dict)
        complex_kg_state_dict = get_complex_kg_state_dict(torch.load(args.complex_state_dict_path))
        lf.secondary_kg.load_state_dict(complex_kg_state_dict)
        distmult_kg_state_dict = get_distmult_kg_state_dict(torch.load(args.distmult_state_dict_path))
        lf.tertiary_kg.load_state_dict(distmult_kg_state_dict)
    else:
        lf.load_checkpoint(get_checkpoint_path(args))
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    if 'NELL' in args.data_dir:
        adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
        seen_entities = data_utils.load_seen_entities(adj_list_path, entity_index_path)
    else:
        seen_entities = set()

    eval_metrics = {
        'dev': {},
        'test': {}
    }

    if args.compute_map:
        relation_sets = [
            'concept:athletehomestadium',
            'concept:athleteplaysforteam',
            'concept:athleteplaysinleague',
            'concept:athleteplayssport',
            'concept:organizationheadquarteredincity',
            'concept:organizationhiredperson',
            'concept:personborninlocation',
            'concept:teamplayssport',
            'concept:worksfor'
        ]
        mps = []
        for r in relation_sets:
            print('* relation: {}'.format(r))
            test_path = os.path.join(args.data_dir, 'tasks', r, 'test.pairs')
            test_data, labels = data_utils.load_triples_with_label(
                test_path, r, entity_index_path, relation_index_path, seen_entities=seen_entities)
            pred_scores = lf.forward(test_data, verbose=False)
            mp = src.eval.link_MAP(test_data, pred_scores, labels, lf.kg.all_objects, verbose=True)
            mps.append(mp)
        map_ = np.mean(mps)
        print('Overall MAP = {}'.format(map_))
        eval_metrics['test']['avg_map'] = map
    elif args.eval_by_relation_type:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
        pred_scores = lf.forward(dev_data, verbose=False)
        to_m_rels, to_1_rels, _ = data_utils.get_relations_by_type(args.data_dir, relation_index_path)
        relation_by_types = (to_m_rels, to_1_rels)
        print('Dev set evaluation by relation type (partial graph)')
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.dev_objects, relation_by_types, verbose=True)
        print('Dev set evaluation by relation type (full graph)')
        src.eval.hits_and_ranks_by_relation_type(
            dev_data, pred_scores, lf.kg.all_objects, relation_by_types, verbose=True)
    elif args.eval_by_seen_queries:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
        pred_scores = lf.forward(dev_data, verbose=False)
        seen_queries = data_utils.get_seen_queries(args.data_dir, entity_index_path, relation_index_path)
        print('Dev set evaluation by seen queries (partial graph)')
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.dev_objects, seen_queries, verbose=True)
        print('Dev set evaluation by seen queries (full graph)')
        src.eval.hits_and_ranks_by_seen_queries(
            dev_data, pred_scores, lf.kg.all_objects, seen_queries, verbose=True)
    else:
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        test_path = os.path.join(args.data_dir, 'test.triples')
        dev_data = data_utils.load_triples(
            dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        test_data = data_utils.load_triples(
            test_path, entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
        print('Dev set performance:')
        pred_scores = lf.forward(dev_data, verbose=args.save_beam_search_paths)
        dev_metrics = src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
        eval_metrics['dev'] = {}
        eval_metrics['dev']['hits_at_1'] = dev_metrics[0]
        eval_metrics['dev']['hits_at_3'] = dev_metrics[1]
        eval_metrics['dev']['hits_at_5'] = dev_metrics[2]
        eval_metrics['dev']['hits_at_10'] = dev_metrics[3]
        eval_metrics['dev']['mrr'] = dev_metrics[4]
        src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.all_objects, verbose=True)
        print('Test set performance:')
        pred_scores = lf.forward(test_data, verbose=False)
        test_metrics = src.eval.hits_and_ranks(test_data, pred_scores, lf.kg.all_objects, verbose=True)
        eval_metrics['test']['hits_at_1'] = test_metrics[0]
        eval_metrics['test']['hits_at_3'] = test_metrics[1]
        eval_metrics['test']['hits_at_5'] = test_metrics[2]
        eval_metrics['test']['hits_at_10'] = test_metrics[3]
        eval_metrics['test']['mrr'] = test_metrics[4]

    return eval_metrics


def export_to_embedding_projector(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_to_embedding_projector()

def export_reward_shaping_parameters(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_reward_shaping_parameters()

def export_fuzzy_facts(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_fuzzy_facts()

def export_error_cases(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.batch_size = args.dev_batch_size
    lf.eval()
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    lf.load_checkpoint(get_checkpoint_path(args))
    print('Dev set performance:')
    pred_scores = lf.forward(dev_data, verbose=False)
    src.eval.hits_and_ranks(dev_data, pred_scores, lf.kg.dev_objects, verbose=True)
    src.eval.export_error_cases(dev_data, pred_scores, lf.kg.dev_objects, os.path.join(lf.model_dir, 'error_cases.pkl'))

def compute_fact_scores(lf):
    data_dir = args.data_dir
    train_path = os.path.join(data_dir, 'train.triples')
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')

    train_data = data_utils.load_triples(train_path, entity_index_path, relation_index_path)
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path)
    test_data = data_utils.load_triples(test_path, entity_index_path, relation_index_path)
    lf.eval()
    lf.load_checkpoint(get_checkpoint_path(args))
    train_scores = lf.forward_fact(train_data)
    dev_scores = lf.forward_fact(dev_data)
    test_scores = lf.forward_fact(test_data)

    print('Train set average fact score: {}'.format(float(train_scores.mean())))
    print('Dev set average fact score: {}'.format(float(dev_scores.mean())))
    print('Test set average fact score: {}'.format(float(test_scores.mean())))

def get_checkpoint_path(args):
    if not args.checkpoint_path:
        return os.path.join(args.model_dir, 'model_best.tar')
    else:
        return args.checkpoint_path

def load_configs(config_path):
    with open(config_path) as f:
        print('loading configuration file {}'.format(config_path))
        for line in f:
            if not '=' in line:
                continue
            arg_name, arg_value = line.strip().split('=')
            if arg_value.startswith('"') and arg_value.endswith('"'):
                arg_value = arg_value[1:-1]
            if hasattr(args, arg_name):
                print('{} = {}'.format(arg_name, arg_value))
                arg_value2 = getattr(args, arg_name)
                if type(arg_value2) is str:
                    setattr(args, arg_name, arg_value)
                elif type(arg_value2) is bool:
                    if arg_value == 'True':
                        setattr(args, arg_name, True)
                    elif arg_value == 'False':
                        setattr(args, arg_name, False)
                    else:
                        raise ValueError('Unrecognized boolean value description: {}'.format(arg_value))
                elif type(arg_value2) is int:
                    setattr(args, arg_name, int(arg_value))
                elif type(arg_value2) is float:
                    setattr(args, arg_name, float(arg_value))
                else:
                    raise ValueError('Unrecognized attribute type: {}: {}'.format(arg_name, type(arg_value2)))
            else:
                raise ValueError('Unrecognized argument: {}'.format(arg_name))
    return args

def run_experiment(args):

    if args.test:
        if 'NELL' in args.data_dir:
            dataset = os.path.basename(args.data_dir)
            args.distmult_state_dict_path = data_utils.change_to_test_model_path(dataset, args.distmult_state_dict_path)
            args.complex_state_dict_path = data_utils.change_to_test_model_path(dataset, args.complex_state_dict_path)
            args.conve_state_dict_path = data_utils.change_to_test_model_path(dataset, args.conve_state_dict_path)
        args.data_dir += '.test'

    if args.process_data:

        # Process knowledge graph data

        process_data()
    else:
        with torch.set_grad_enabled(args.train or args.search_random_seed or args.grid_search):
            if args.search_random_seed:

                # Search for best random seed

                # search log file
                task = os.path.basename(os.path.normpath(args.data_dir))
                out_log = '{}.{}.rss'.format(task, args.model)
                o_f = open(out_log, 'w')

                print('** Search Random Seed **')
                o_f.write('** Search Random Seed **\n')
                o_f.close()
                num_runs = 5

                hits_at_1s = {}
                hits_at_10s = {}
                mrrs = {}
                mrrs_search = {}
                for i in range(num_runs):

                    o_f = open(out_log, 'a')

                    random_seed = random.randint(0, 1e16)
                    print("\nRandom seed = {}\n".format(random_seed))
                    o_f.write("\nRandom seed = {}\n\n".format(random_seed))
                    torch.manual_seed(random_seed)
                    torch.cuda.manual_seed_all(args, random_seed)
                    initialize_model_directory(args, random_seed)
                    lf = construct_model(args)
                    lf.cuda()
                    train(lf)
                    metrics = inference(lf)
                    hits_at_1s[random_seed] = metrics['test']['hits_at_1']
                    hits_at_10s[random_seed] = metrics['test']['hits_at_10']
                    mrrs[random_seed] = metrics['test']['mrr']
                    mrrs_search[random_seed] = metrics['dev']['mrr']
                    # print the results of the hyperparameter combinations searched so far
                    print('------------------------------------------')
                    print('Random Seed\t@1\t@10\tMRR')
                    for key in hits_at_1s:
                        print('{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    print('------------------------------------------')
                    o_f.write('------------------------------------------\n')
                    o_f.write('Random Seed\t@1\t@10\tMRR\n')
                    for key in hits_at_1s:
                        o_f.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    o_f.write('------------------------------------------\n')

                    # compute result variance
                    import numpy as np
                    hits_at_1s_ = list(hits_at_1s.values())
                    hits_at_10s_ = list(hits_at_10s.values())
                    mrrs_ = list(mrrs.values())
                    print('Hits@1 mean: {:.3f}\tstd: {:.6f}'.format(np.mean(hits_at_1s_), np.std(hits_at_1s_)))
                    print('Hits@10 mean: {:.3f}\tstd: {:.6f}'.format(np.mean(hits_at_10s_), np.std(hits_at_10s_)))
                    print('MRR mean: {:.3f}\tstd: {:.6f}'.format(np.mean(mrrs_), np.std(mrrs_)))
                    o_f.write('Hits@1 mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(hits_at_1s_), np.std(hits_at_1s_)))
                    o_f.write('Hits@10 mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(hits_at_10s_), np.std(hits_at_10s_)))
                    o_f.write('MRR mean: {:.3f}\tstd: {:.6f}\n'.format(np.mean(mrrs_), np.std(mrrs_)))
                    o_f.close()

                # find best random seed
                best_random_seed, best_mrr = sorted(mrrs_search.items(), key=lambda x: x[1], reverse=True)[0]
                print('* Best Random Seed = {}'.format(best_random_seed))
                print('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}'.format(
                    hits_at_1s[best_random_seed],
                    hits_at_10s[best_random_seed],
                    mrrs[best_random_seed]))
                with open(out_log, 'a'):
                    o_f.write('* Best Random Seed = {}\n'.format(best_random_seed))
                    o_f.write('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}\n'.format(
                        hits_at_1s[best_random_seed],
                        hits_at_10s[best_random_seed],
                        mrrs[best_random_seed])
                    )
                    o_f.close()

            elif args.grid_search:

                # Grid search

                # search log file
                task = os.path.basename(os.path.normpath(args.data_dir))
                out_log = '{}.{}.gs'.format(task, args.model)
                o_f = open(out_log, 'w')

                print("** Grid Search **")
                o_f.write("** Grid Search **\n")
                hyperparameters = args.tune.split(',')

                if args.tune == '' or len(hyperparameters) < 1:
                    print("No hyperparameter specified.")
                    sys.exit(0)

                grid = hp_range[hyperparameters[0]]
                for hp in hyperparameters[1:]:
                    grid = itertools.product(grid, hp_range[hp])

                hits_at_1s = {}
                hits_at_10s = {}
                mrrs = {}
                grid = list(grid)
                print('* {} hyperparameter combinations to try'.format(len(grid)))
                o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
                o_f.close()

                for i, grid_entry in enumerate(list(grid)):

                    o_f = open(out_log, 'a')

                    if not (type(grid_entry) is list or type(grid_entry) is list):
                        grid_entry = [grid_entry]
                    grid_entry = flatten(grid_entry)
                    print('* Hyperparameter Set {}:'.format(i))
                    o_f.write('* Hyperparameter Set {}:\n'.format(i))
                    signature = ''
                    for j in range(len(grid_entry)):
                        hp = hyperparameters[j]
                        value = grid_entry[j]
                        if hp == 'bandwidth':
                            setattr(args, hp, int(value))
                        else:
                            setattr(args, hp, float(value))
                        signature += ':{}'.format(value)
                        print('* {}: {}'.format(hp, value))
                    initialize_model_directory(args)
                    lf = construct_model(args)
                    lf.cuda()
                    train(lf)
                    metrics = inference(lf)
                    hits_at_1s[signature] = metrics['dev']['hits_at_1']
                    hits_at_10s[signature] = metrics['dev']['hits_at_10']
                    mrrs[signature] = metrics['dev']['mrr']
                    # print the results of the hyperparameter combinations searched so far
                    print('------------------------------------------')
                    print('Signature\t@1\t@10\tMRR')
                    for key in hits_at_1s:
                        print('{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    print('------------------------------------------\n')
                    o_f.write('------------------------------------------\n')
                    o_f.write('Signature\t@1\t@10\tMRR\n')
                    for key in hits_at_1s:
                        o_f.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                            key, hits_at_1s[key], hits_at_10s[key], mrrs[key]))
                    o_f.write('------------------------------------------\n')
                    # find best hyperparameter set
                    best_signature, best_mrr = sorted(mrrs.items(), key=lambda x:x[1], reverse=True)[0]
                    print('* best hyperparameter set')
                    o_f.write('* best hyperparameter set\n')
                    best_hp_values = best_signature.split(':')[1:]
                    for i, value in enumerate(best_hp_values):
                        hp_name = hyperparameters[i]
                        hp_value = best_hp_values[i]
                        print('* {}: {}'.format(hp_name, hp_value))
                    print('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}'.format(
                        hits_at_1s[best_signature],
                        hits_at_10s[best_signature],
                        mrrs[best_signature]
                    ))
                    o_f.write('* @1: {:.3f}\t@10: {:.3f}\tMRR: {:.3f}\ns'.format(
                        hits_at_1s[best_signature],
                        hits_at_10s[best_signature],
                        mrrs[best_signature]
                    ))

                    o_f.close()

            elif args.run_ablation_studies:
                None
            else:
                initialize_model_directory(args)
                lf = construct_model(args)
                lf.cuda()

                if args.train:
                    train(lf)
                elif args.inference:
                    inference(lf)
                elif args.eval_by_relation_type:
                    inference(lf)
                elif args.eval_by_seen_queries:
                    inference(lf)
                elif args.export_to_embedding_projector:
                    export_to_embedding_projector(lf)
                elif args.export_reward_shaping_parameters:
                    export_reward_shaping_parameters(lf)
                elif args.compute_fact_scores:
                    compute_fact_scores(lf)
                elif args.export_fuzzy_facts:
                    export_fuzzy_facts(lf)
                elif args.export_error_cases:
                    export_error_cases(lf)

if __name__ == '__main__':
    run_experiment(args)
