'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
'''

import os
from os.path import join, exists
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
import random
import json


from dataset_mimic_full import mimicfull_VQAFeatureDataset, Dictionary
from model.regat import build_regat
from config.parser import parse_with_config
from train import train, evaluate
from utils import utils
from utils.utils import trim_collate
import wandb

# os.environ['CUDA_LAUNCH_BLOCKING']='1'
# os.environ['CUDA_VISIBLE_DEVICES']='1'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
def parse_args():
    parser = argparse.ArgumentParser()
    '''
    For training logistics
    '''
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--base_lr', type=float, default=0.001) # 7e-5
    parser.add_argument('--lr_decay_start', type=int, default=15)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_step', type=int, default=2)
    parser.add_argument('--lr_decay_based_on_val', action='store_true',
                        help='Learning rate decay when val score descreases')
    parser.add_argument('--grad_accu_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=0.25)
    parser.add_argument('--weight_decay', type=float, default=5e-4) # 0
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--save_optim', action='store_true',
                        help='save optimizer')
    parser.add_argument('--log_interval', type=int, default=-1,
                        help='Print log for certain steps')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')

    '''
    loading trained models
    '''
    parser.add_argument('--checkpoint', type=str, default="")

    '''
    For dataset
    '''
    parser.add_argument('--dataset', type=str, default="mimic-vqa",
                        choices=["mimic-vqa",'vqarad','vqamed'])
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--use_both', action='store_true',
                        help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', action='store_true',
                        help='use visual genome dataset to train?')
    parser.add_argument('--adaptive', action='store_true',
                        help='adaptive or fixed number of regions')
    '''
    Model
    '''
    parser.add_argument('--relation_type', type=str, default='implicit',
                        choices=["spatial", "semantic", "implicit", 'my_semantic']) # this argument doesn't work here. please go to mimic_vqa.json
    parser.add_argument('--fusion', type=str, default='butd', choices=["ban", "butd", "mutan"])
    parser.add_argument('--tfidf', action='store_true',
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help="op used in tfidf word embedding")
    parser.add_argument('--num_hid', type=int, default=1024)
    '''
    Fusion Hyperparamters
    '''
    parser.add_argument('--ban_gamma', type=int, default=1, help='glimpse')
    parser.add_argument('--mutan_gamma', type=int, default=2, help='glimpse')
    '''
    Hyper-params for relations
    '''
    # hyper-parameters for implicit relation
    parser.add_argument('--imp_pos_emb_dim', type=int, default=64,
                        help='geometric embedding feature dim')

    # hyper-parameters for explicit relation
    parser.add_argument('--spa_label_num', type=int, default=11,
                        help='number of edge labels in spatial relation graph')
    parser.add_argument('--sem_label_num', type=int, default=15,
                        help='number of edge labels in \
                              semantic relation graph')

    # shared hyper-parameters
    parser.add_argument('--dir_num', type=int, default=2,
                        help='number of directions in relation graph')
    parser.add_argument('--relation_dim', type=int, default=1024,
                        help='relation feature dim')
    parser.add_argument('--nongt_dim', type=int, default=20,
                        help='number of objects consider relations per image')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='number of attention heads \
                              for multi-head attention')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='number of graph propagation steps')
    parser.add_argument('--residual_connection', action='store_true',
                        help='Enable residual connection in relation encoder')
    parser.add_argument('--label_bias', action='store_true',
                        help='Enable bias term for relation labels \
                              in relation encoder')


    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_graph_classification', type=bool, default=False)
    parser.add_argument('--use_pos_emb', type=bool, default=False, help='only for graph classification net')
    parser.add_argument('--dryrun', type=str2bool, default=True, help='wandb') # manually assign it here
    parser.add_argument('--use_contrastive', type=bool, default=False, help='contrastive loss')
    parser.add_argument('--pure_classification', type=bool, default=True, help='contrastive loss')
    parser.add_argument('--use_q', type=bool, default=False, help='if using q in the feat matrix before the concat, when using my_semantic following TMI paper. This won\'t affect normal implicit/spatial/semantic relations')
    parser.add_argument('--sem_region_feature', type=bool, default=False, help= 'the modified semantic calculation following TMI paper')
    parser.add_argument('--test_spa_adj_thr', type=float, default=0, help='testing spatial adjaceny matrix threshold')
    parser.add_argument('--cross_attention', type=bool, default=False, help='cross attention')
    parser.add_argument('--testing_code', type=int, default=0, help='use spatial relation')

    parser.add_argument('--ggnn', type=bool, default=False, help='use ggnn')
    parser.add_argument('--state_dim', type=int, default=1024, help='use ggnn')
    parser.add_argument('--annotation_dim', type=int, default=60, help='use ggnn')
    parser.add_argument('--n_edge_types', type=int, default=3, help='use ggnn')
    parser.add_argument('--n_steps', type=int, default=2, help='use ggnn')
    parser.add_argument('--n_node', type=int, default=52, help='use ggnn')
    parser.add_argument('--KG_dim', type=int, default=600, help='use ggnn')
    parser.add_argument('--num_ans_candidates', type=int, default=3, help='use ggnn')
    # can use config files

    parser.add_argument('--config', default='config/mimic_vqa.json', help='JSON config files')

    args = parse_with_config(parser)
    return args


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available," +
                         "this code currently only support GPU.")
    n_device = torch.cuda.device_count()
    print("Found %d GPU cards for training" % (n_device))
    device = torch.device("cuda")
    # device = torch.device("cpu")
    batch_size = args.batch_size



    torch.backends.cudnn.benchmark = True

    if args.seed != -1:
        print("Predefined randam seed %d" % args.seed)
    else:
        # fix seed
        args.seed = random.randint(1, 10000)
        print("Choose random seed %d" % args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if "ban" == args.fusion:
        fusion_methods = args.fusion+"_"+str(args.ban_gamma)
    else:
        fusion_methods = args.fusion

    if args.dataset == 'mimic-full':
        dictionary = Dictionary.load_from_file(
            join(args.data_folder, 'mimic/mimic_dictionary_full.pkl'))
    elif args.dataset == 'mimic-vqa':
        dictionary = Dictionary.load_from_file(
            join(args.data_folder, 'mimic_vqa/mimic_dictionary.pkl'))
        node_names = dictionary.node_names()
        for name in node_names:
            dictionary.add_word(name)
    elif args.dataset == 'vqamed':
        dictionary = Dictionary.load_from_file(
            join(args.data_folder, 'vqamed/vqamed_dictionary.pkl'))
    elif args.dataset == 'vqarad':
        dictionary = Dictionary.load_from_file(
            join(args.data_folder, 'vqarad/vqarad_dictionary.pkl'))
    else:
        dictionary = Dictionary.load_from_file(
                        join(args.data_folder, 'glove/dictionary.pkl'))

    val_dset = mimicfull_VQAFeatureDataset(
        'val', args.dataset, dictionary, args.relation_type, adaptive=args.adaptive,
        pos_emb_dim=args.imp_pos_emb_dim, dataroot=args.data_folder, args=args)
    train_dset = mimicfull_VQAFeatureDataset(
        'train', args.dataset,dictionary, args.relation_type,
        adaptive=args.adaptive, pos_emb_dim=args.imp_pos_emb_dim,
        dataroot=args.data_folder, args=args)
    test_dset = mimicfull_VQAFeatureDataset(
        'test', args.dataset, dictionary, args.relation_type, adaptive=args.adaptive,
        pos_emb_dim=args.imp_pos_emb_dim, dataroot=args.data_folder, args=args)
    args.num_ans_candidates = train_dset.num_ans_candidates

    model = build_regat(val_dset, args).to(device)
    model = nn.DataParallel(model).cuda()


    if args.checkpoint != "":
        print("Loading weights from %s" % (args.checkpoint))
        if not os.path.exists(args.checkpoint):
            raise ValueError("No such checkpoint exists!")
        checkpoint = torch.load(args.checkpoint)
        state_dict = checkpoint.get('model_state', checkpoint)
        matched_state_dict = {}
        unexpected_keys = set()
        missing_keys = set()
        for name, param in model.named_parameters():
            missing_keys.add(name)
        for key, data in state_dict.items():
            if key in missing_keys:
                # if key == 'w_emb.emb.weight' or key == 'w_emb.emb_.weight':
                #     data = torch.cat((data,data[:58-35]))
                # elif key == 'classifier.main.3.bias' or key == 'classifier.main.3.weight_v':
                #     data = torch.cat((data,data[:1]))
                matched_state_dict[key] = data
                missing_keys.remove(key)
            else:
                unexpected_keys.add(key)
        print("Unexpected_keys:", list(unexpected_keys))
        print("Missing_keys:", list(missing_keys))
        model.load_state_dict(matched_state_dict, strict=False)

    # use train & val splits to optimize, only available for vqa, not vqa_cp

    train_loader = DataLoader(train_dset, batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=trim_collate)
    eval_loader = DataLoader(test_dset, batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=trim_collate)
    test_loader = DataLoader(test_dset, batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=trim_collate)


    output_meta_folder = join(args.output, "regat_%s" % args.relation_type)
    if not args.use_q:
        output_meta_folder += "_noQ"
    utils.create_dir(output_meta_folder)
    args.output = output_meta_folder+"/%s_%s_%s_%d" % (
                fusion_methods, args.relation_type,
                args.dataset, args.seed)
    if exists(args.output) and os.listdir(args.output):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output))
    utils.create_dir(args.output)
    with open(join(args.output, 'hps.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    logger = utils.Logger(join(args.output, 'log.txt'))
    if not args.dryrun:
        extra = '_2questions'
        relation = args.relation_type + "_noQ" if not args.use_q else args.relation_type
        name = 'ReGat_on_' + args.dataset + '_' + relation + '_' + str(args.nongt_dim)+extra
        if args.test_spa_adj_thr:
            name += '_adj-thr' + str(args.test_spa_adj_thr)
        if args.relation_type == 'my_semantic' and args.cross_attention:
            name += '_cross-attention'
        if args.relation_type == 'my_semantic' and args.testing_code != 0:
            name += '_testing-code' + str(args.testing_code)
        wandb.init(project='mimic-VQA', config=args, name = name, allow_val_change=True)
        # wandb.config.update(args, )
        args = wandb.config
    if not args.dryrun:
        wandb.watch(model)
    # train(model, train_loader, eval_loader, args, train_dset, wandb, device)

    eval_score, bound, entropy, eval_score2, _ = evaluate(
        model, test_loader, device, args, val_dset, wandb, best=100)
