"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import pickle
import numpy as np
from utils import utils
from model.position_emb import prepare_graph_variables
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import  GradScaler


tqdm.monitor_interval = 0


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    assert(not logits.isnan().any() )
    assert(not labels.isnan().any())
    loss = F.binary_cross_entropy_with_logits(
                                logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels, device, questions, idx2ques, question_score):
    # argmax
    logits = torch.max(logits, 1)[1].data
    logits = logits.view(-1, 1)
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits, 1)
    scores = (one_hots * labels)

    for idx, question in enumerate(questions):
        question = get_raw_question(question, idx2ques)
        if question not in question_score:
            question_score[question] = int(scores.sum(1)[idx])
            question_score[question+'_t'] = 1
        else:
            question_score[question] += int(scores.sum(1)[idx])
            question_score[question+'_t'] += 1
    return scores

def get_raw_question(q, idx2ques):
    raw = ''
    for w in q:
        try:
            word = idx2ques[int(w.cpu())]
        except:
            return raw
        raw = raw + ' ' + word
    return raw

def print_question_score(question_score, question_score2, name = ''):
    print(name, 'VQA questions score1')
    for score in question_score:
        try:
            print(score, question_score[score] ,'/', question_score[score+'_t'], ';', question_score[score]/question_score[score+'_t'])
        except:
            pass
    print('\n')
    print(name, 'graph classification questions score2')
    for score in question_score2:
        try:
            print(score, question_score[score], '/', question_score[score + '_t'], ';',
                  question_score[score] / question_score[score + '_t'])
        except:
            pass

def get_contrastive_loss(logits, answer_logits, cel_loss, device):
    logits = logits @ answer_logits.T
    labels = torch.arange(len(logits)).to(device)
    loss_i = cel_loss(logits, labels)
    loss_t = cel_loss(logits.T, labels)
    loss = (loss_i + loss_t) / 2
    return loss

def enrich_answer(anss):
    if anss[0] =="yes" or anss[0] == 'no':
        return anss[0]
    else:
        if len(anss) == 1:
            return 'an x-ray image contains ' + anss[0]
        else:
            ans = 'an x-ray image contains '
            for i in range(len(anss)):
                if i == len(anss) -1:
                    ans += 'and ' + anss[i]
                else:
                    ans += anss[i]+', '
        return ans

def compute_t_emb(tokens,model):
    with torch.no_grad():
        emb = model.get_emb(tokens)
    return emb

def compute_score(logits, target_ori, dataset, model,device):
    anss_tokens = []
    for i in range(7):
        ans_t = enrich_answer([dataset.label2ans[i]])
        anss_tokens.append(torch.tensor(dataset.sub_tokenize(ans_t)).to(device).unsqueeze(0))
    anss_tokens = torch.cat(anss_tokens)
    anss_emb = compute_t_emb(anss_tokens, model)
    logits /= logits.norm(dim=-1, keepdim=True)
    anss_emb /= anss_emb.norm(dim=-1, keepdim=True)
    similarity = (100.0 * logits @ anss_emb.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    pred = torch.max(similarity, 1)[1].data
    pred = pred.view(-1, 1)
    one_hots = torch.zeros(*target_ori.size()).to(device)
    one_hots.scatter_(1, pred, 1)
    scores = (one_hots * target_ori)
    return scores

def collect_preds_tgts(length, logits, target_ori, preds, targets, dataset, model, device):
    anss_tokens = []
    for i in range(length):
        ans_t = enrich_answer([dataset.label2ans[i]])
        anss_tokens.append(torch.tensor(dataset.sub_tokenize(ans_t)).to(device).unsqueeze(0))
    anss_tokens = torch.cat(anss_tokens)
    anss_emb = compute_t_emb(anss_tokens, model)
    logits /= logits.norm(dim=-1, keepdim=True)
    anss_emb /= anss_emb.norm(dim=-1, keepdim=True)
    pred = (100.0 * logits @ anss_emb.T).softmax(dim=-1) # similarity scores
    preds.append(pred.cpu().detach().numpy())
    targets.append(target_ori.cpu().detach().numpy())

    return preds, targets

def check_cls(q, cls_set):
    for cls in cls_set:
        if (np.array(q.cpu()) == cls).all():
            return True
    return False

def get_q_type_index(q, cls_set, what_idx_set, yesno_idx_set):
    cls_idx = []
    what_idx = []
    yesno_idx = []
    for i in range(len(q)):
        # if check_cls(q[i], cls_set):
        #     cls_idx.append(i)
        if int(q[i][0]) in what_idx_set:
            what_idx.append(i)
        elif int(q[i][0]) in yesno_idx_set:
        # if int(q[i][0]) in yesno_idx_set:
            yesno_idx.append(i)
    return cls_idx, what_idx, yesno_idx

def get_q_sets(dataset, pure_classification):
    cls = ['what is the primary abnormality in this image?', 'what abnormality is seen in the image?',
           'what is most alarming about this x-ray?', 'what is abnormal in the x-ray?', ]
    cls_set = []
    for c in cls:
        cls_set.append(np.array(dataset.sub_tokenize(c)))
    what_idx_set = set([dataset.dictionary.word2idx['what']])
    yesno_idx_set = set()
    yesno_start = ['is', 'whether', 'does']
    if pure_classification:
        yesno_start = ['is']
    for item in yesno_start:
        yesno_idx_set.add(dataset.dictionary.word2idx[item])
    return cls_set, what_idx_set, yesno_idx_set

def compute_yesno_score(logits, labels, device):
    logits = np.argmax(logits,1)
    # logits = logits.view(-1, 1)
    one_hots = np.eye(labels.shape[-1])[logits]
    # one_hots.scatter_(1, logits, 1)
    scores = (one_hots * labels).sum()/len(logits)
    return scores

def compute_acc(targets, preds, topk=[1, 3, 5, 10, 15]):
    '''

    Args:
        targets: shape= [n x c] (multiple targets)
        preds: shape = [n x c]
        topk: list

    Returns:

    '''

    maxk = max(topk)
    batch_size = targets.shape[0]
    _, pred = preds.topk(maxk, 1, True, True)
    ret = []
    for k in topk:
        correct = (targets * torch.zeros_like(targets).scatter(1, pred[:, :k], 1)).float()
        ret.append(float(correct.sum() / targets.sum()))
    return ret

def train(model, train_loader, eval_loader, args, dataset, wandb, device=torch.device("cuda")):
    N = len(train_loader.dataset)
    lr_default = args.base_lr
    num_epochs = args.epochs
    lr_decay_epochs = range(args.lr_decay_start, num_epochs,
                            args.lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,
                            1.5 * lr_default, 2.0 * lr_default]

    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=lr_default, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=args.weight_decay)
    scaler = GradScaler()

    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.6f, decay_step=%d, decay_rate=%.2f,'
                 % (lr_default, args.lr_decay_step,
                    args.lr_decay_rate) + 'grad_clip=%.2f' % args.grad_clip)
    logger.write('LR decay epochs: '+','.join(
                                        [str(i) for i in lr_decay_epochs]))
    last_eval_score, eval_score = 0, 0
    relation_type = train_loader.dataset.relation_type
    cel_loss = nn.CrossEntropyLoss()
    with open('data/mimic/mimic_idx2words_full.pkl', 'rb') as f:
        idx2ques = pickle.load(f)

    cls_set, what_idx_set, yesno_idx_set = get_q_sets(dataset, args.pure_classification)
    best = 0
    best_ori = 0
    for epoch in range(0, num_epochs):
        pbar = tqdm(total=len(train_loader))
        total_norm, count_norm = 0, 0
        total_loss, train_score = 0, 0
        count, average_loss, att_entropy = 0, 0, 0
        if args.use_graph_classification:
            train_score2 = 0
        t = time.time()
        if epoch < len(gradual_warmup_steps):
            for i in range(len(optim.param_groups)):
                optim.param_groups[i]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.6f' %
                         optim.param_groups[-1]['lr'])
        elif (epoch in lr_decay_epochs or
              eval_score < last_eval_score and args.lr_decay_based_on_val):
            for i in range(len(optim.param_groups)):
                optim.param_groups[i]['lr'] *= args.lr_decay_rate
            logger.write('decreased lr: %.6f' % optim.param_groups[-1]['lr'])
        else:
            logger.write('lr: %.6f' % optim.param_groups[-1]['lr'])
        last_eval_score = eval_score

        mini_batch_count = 0
        batch_multiplier = args.grad_accu_steps
        question_score = {}
        question_score2 = {}
        preds, targets = [], []
        cls_preds, cls_targets = [], []
        what_preds, what_targets = [], []
        yesno_preds, yesno_targets = [], []
        for i, (v, norm_bb, q, target, _, _, bb, spa_adj_matrix, sem_adj_matrix, node_label, img, sem_region_feats, bbx_label) in enumerate(train_loader):
            batch_size = v.size(0)
            num_objects = v.size(1)

            v = Variable(v).to(device)
            norm_bb = Variable(norm_bb).to(device)
            q = Variable(q).to(device)
            if args.dataset == 'mimic' or args.dataset == 'mimic-full' or args.dataset == 'mimic-vqa' or args.dataset == 'vqamed' or args.dataset == 'vqarad':
                target2 = target[1]
                target2 = target2.to(device)
                target_ori = target[2].to(device)
                target = target[0]
                if isinstance(sem_region_feats, list):
                    for i in range(len(sem_region_feats)):
                        sem_region_feats[i] = sem_region_feats[i].to(device)
                else:
                    sem_region_feats = sem_region_feats.to(device)
                bbx_label = bbx_label.to(device)
            target = Variable(target).to(device)
            pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
                relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
                args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
                args.sem_label_num, device)

            # with torch.cuda.amp.autocast() as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable :
            logits, ad_loss, att, node_pred = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                              spa_adj_matrix, target, node_label, sem_region_feats, bbx_label)

            if args.use_contrastive:
                loss = get_contrastive_loss(logits, answer_logits, cel_loss,device)
            else:
                loss = instance_bce_with_logits(logits, target_ori)
            if args.relation_type == 'my_semantic':
                loss0 = instance_bce_with_logits(node_pred, node_label.to(node_pred.device))
                loss = loss + loss0
            if ad_loss.shape != torch.tensor(0.5).shape:
                ad_loss = sum(ad_loss)
            if ad_loss != 0:
                loss = loss + ad_loss
            if not args.dryrun:
                wandb.log({'vqa_loss': loss})

            if args.use_graph_classification:
                loss2 = instance_bce_with_logits(pred2, target2)
                if not args.dryrun:
                    wandb.log({'graph_cls_loss': loss2})
                loss = loss + loss2
                batch_score2 = compute_score_with_logits(pred2, target2, device, q, idx2ques, question_score2).sum()
                train_score2 += batch_score2
            # else:
            #     pred, att = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
            #                       spa_adj_matrix, target)
            #     loss = instance_bce_with_logits(pred, target)

            loss /= batch_multiplier
            # try:
            scaler.scale(loss).backward()

            mini_batch_count += 1
            # if mini_batch_count == batch_multiplier:
            # optim.step()
            # optim.zero_grad()
            mini_batch_count = 0

            eps = 1e-6
            for name, param in model.module.named_parameters():
                if "weight_v" in name:  # Only operate on the weight_v parameter, weight_g is not needed.
                    param.grad.data.div_(param.norm() + eps)  # Divide the gradient by its norm and add eps.

            scaler.step(optim)
            scaler.update()

            if args.use_graph_classification:
                loss *= batch_multiplier
                loss -= loss2
                loss /= batch_multiplier
            total_norm += nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip)
            count_norm += 1
            # batch_score = compute_score(logits, target_ori, dataset, model,device).sum()
            length = len(dataset.label2ans)
            if args.use_contrastive:
                preds, targets = collect_preds_tgts(length, logits, target_ori, preds, targets, dataset, model, device)
            else:
                preds.append(logits.cpu().detach().numpy())
                targets.append(target_ori.cpu().detach().numpy())

                cls_idx, what_idx, yesno_idx = get_q_type_index(q, cls_set, what_idx_set, yesno_idx_set)
                # cls_idx = torch.tensor(cls_idx)
                what_idx = torch.tensor(what_idx)
                yesno_idx = torch.tensor(yesno_idx)
                #
                # cls_preds.append(torch.index_select(logits.cpu().detach(), 0, cls_idx))
                # cls_targets.append(torch.index_select(target_ori.cpu().detach(), 0, cls_idx))
                # what_preds.append(torch.index_select(logits.cpu().detach(), 0, what_idx))
                # what_targets.append(torch.index_select(target_ori.cpu().detach(), 0, what_idx))

                if len(yesno_idx) >0:
                    yesno_pred = torch.index_select(logits.cpu().detach(), 0, yesno_idx)
                    yesno_target = torch.index_select(target_ori.cpu().detach(), 0, yesno_idx)
                    yesno_preds.append(yesno_pred)
                    yesno_targets.append(yesno_target)





            # batch_score = compute_score_with_logits(pred, target, device, q, idx2ques, question_score).sum()
            # train_score += batch_score
            total_loss = total_loss + loss.data.item() * batch_multiplier * v.size(0)

            pbar.update(1)

            if args.log_interval > 0:
                average_loss = average_loss + loss.data.item() * batch_multiplier
                if model.fusion == "ban":
                    current_att_entropy = torch.sum(calc_entropy(att.data))
                    att_entropy += current_att_entropy / batch_size / att.size(1)
                count += 1
                if i % args.log_interval == 0:
                    att_entropy /= count
                    average_loss /= count
                    print("step {} / {} (epoch {}), ave_loss {:.3f},".format(
                            i, len(train_loader), epoch,
                            average_loss),
                          "att_entropy(att/aff) {:.3f}".format(att_entropy))
                    # print_question_score(question_score, question_score2)
                    average_loss = 0
                    count = 0
                    att_entropy = 0

        total_loss /= N
        # train_score = 100 * train_score / N
        targets = np.vstack(targets)
        preds = np.vstack(preds)
        # cls_targets = np.vstack(cls_targets)
        # cls_preds = np.vstack(cls_preds)
        # what_targets = np.vstack(what_targets)
        # what_preds = np.vstack(what_preds)
        if len(yesno_targets)>0:
            yesno_targets = np.vstack(yesno_targets)
            yesno_preds = np.vstack(yesno_preds)

        acc_top = compute_acc(torch.from_numpy(targets), torch.from_numpy(preds))
        # if args.pure_classification:
        #     what_targets = np.delete(what_targets, [dataset.ans2label['no']], axis=1)
        #     what_preds = np.delete(what_preds, [dataset.ans2label['no']], axis=1)
        # else:
        #     cls_targets = np.delete(cls_targets,[dataset.ans2label['yes'] , dataset.ans2label['no']], axis=1)
        #     cls_preds = np.delete(cls_preds, [dataset.ans2label['yes'], dataset.ans2label['no']], axis=1)
        # what_targets = np.delete(what_targets, [dataset.ans2label['yes'], dataset.ans2label['no']], axis=1)
        # what_preds = np.delete(what_preds, [dataset.ans2label['yes'], dataset.ans2label['no']], axis=1)
        micro_roc_auc = roc_auc_score(targets, preds, average="micro")
        try:
            macro_roc_auc = roc_auc_score(targets, preds, average="macro")
        except:
            pass






        # cls_micro_roc_auc = roc_auc_score(cls_targets, cls_preds, average="micro")
        # cls_macro_roc_auc = roc_auc_score(cls_targets, cls_preds, average="macro")
        # what_micro_roc_auc = roc_auc_score(what_targets, what_preds, average="micro")
        # what_macro_roc_auc = roc_auc_score(what_targets, what_preds, average="macro")
        if len(yesno_targets) > 0:
            yesno_score = compute_yesno_score(yesno_preds, yesno_targets, device)
        else:
            yesno_score = 0


        if not args.dryrun:
            wandb.log({'train_micro_auc':micro_roc_auc,
                       # 'train_macro_auc':macro_roc_auc,
                       # 'train_cls_micro_auc': cls_micro_roc_auc,
                       # 'train_cls_macro_auc': cls_macro_roc_auc,
                       # 'train_what_micro_auc': what_micro_roc_auc,
                       # 'train_what_macro_auc': what_macro_roc_auc,
                       'train_yesno_score': yesno_score,
                       'train_acc-1': acc_top[0],
                        'train_acc-3': acc_top[1],
                        'train_acc-5': acc_top[2],
                        'train_acc-10': acc_top[3],
                        'train_acc-25': acc_top[4],
                       'lr': optim.param_groups[-1]['lr']
                       })
            try:
                wandb.log({'train_macro_auc': macro_roc_auc})
            except:
                pass


        if eval_loader is not None:
            eval_score, bound, entropy, eval_score2, best = evaluate(model, eval_loader, device, args, dataset, wandb, best)
            if best > best_ori:
                best_ori = best
                logger.write("saving current model weights to folder")
                model_path = os.path.join(args.output, 'model_%d.pth' % epoch)
                opt = optim if args.save_optim else None
                utils.save_model(model_path, model, epoch, opt)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        if args.use_graph_classification:
            train_score2 = 100 * train_score2 / N
            logger.write('\ttrain_loss: %.2f, norm: %.4f, AUC(micro): %.2f,  graph_cls_score: %.2f'
                         % (total_loss, total_norm / count_norm, 100 *micro_roc_auc, train_score2))
        else:
            AUC = 'AUC(micro): %.2f' % (100 *micro_roc_auc)
            try:
                AUC += 'AUC(macro): %.2f' % (100 * macro_roc_auc)
            except:
                pass
            logger.write('\ttrain_loss: %.2f, norm: %.4f, %s,  yesno_score: %.2f, '
                         % (total_loss, total_norm / count_norm, AUC, 100* yesno_score,
                            ))
            # logger.write('\ttrain_loss: %.2f, norm: %.4f, AUC(micro): %.2f, AUC(macro): %.2f, cls_AUC(micro): %.2f, cls_AUC(macro): %.2f, what_AUC(micro): %.2f, what_AUC(macro): %.2f, yesno_score: %.2f'
            #             % (total_loss, total_norm / count_norm, 100 * micro_roc_auc, 100 * macro_roc_auc, 100 * cls_micro_roc_auc,
            #             100 * cls_macro_roc_auc, 100 * what_micro_roc_auc, 100 * what_macro_roc_auc, 100 * yesno_score))

            # logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f'
            #              % (total_loss, total_norm / count_norm, train_score))
        if eval_loader is not None:
            if args.use_graph_classification:
                logger.write('\teval AUC(micro): %.2f,  eval_graph_cls_score: %.2f'
                             % (100 * eval_score[0], 100 * eval_score2))
            else:
                logger.write('\teval AUC(micro): %.2f, eval yesno_score: %.2f'
                             'eval what_AUC(micro): %.2f, eval what_AUC(macro): %.2f,'
                             % (100 * eval_score[0], 100 * eval_score[2],
                                100 * eval_score[3], 100 * eval_score[4]))
                if args.dataset == 'vqamed' or args.dataset == 'vqarad':
                    logger.write('\teval acc: %.2f'
                                 % (100 * eval_score[1]))
                # logger.write('\teval score: %.2f (%.2f)'
                #              % (100 * eval_score, 100 * bound))

            if entropy is not None:
                info = ''
                for i in range(entropy.size(0)):
                    info = info + ' %.2f' % entropy[i]
                logger.write('\tentropy: ' + info)
        # if (eval_loader is not None)\
        #    or (eval_loader is None and epoch >= args.saving_epoch):
        #     logger.write("saving current model weights to folder")
        #     model_path = os.path.join(args.output, 'model_%d.pth' % epoch)
        #     opt = optim if args.save_optim else None
        #     utils.save_model(model_path, model, epoch, opt)


@torch.no_grad()
def evaluate(model, dataloader, device, args, dataset, wandb, best):
    model.eval()
    relation_type = dataloader.dataset.relation_type
    score = 0
    upper_bound = 0
    num_data = 0
    score2 = 0
    N = len(dataloader.dataset)
    entropy = None
    if model.module.fusion == "ban":
        entropy = torch.Tensor(model.glimpse).zero_().to(device)
    pbar = tqdm(total=len(dataloader))

    question_score2 = {}
    question_score = {}
    with open('data/mimic/mimic_idx2words_full.pkl', 'rb') as f:
        idx2ques = pickle.load(f)
    preds, targets = [], []
    cls_preds, cls_targets = [], []
    what_preds, what_targets = [], []
    yesno_preds, yesno_targets = [], []
    cls_set, what_idx_set, yesno_idx_set = get_q_sets(dataset,args.pure_classification)
    for i, (v, norm_bb, q, target, _, _, bb, spa_adj_matrix,
            sem_adj_matrix, node_label, img, sem_region_feats, bbx_label) in enumerate(dataloader):
        batch_size = v.size(0)
        num_objects = v.size(1)
        v = Variable(v).to(device)
        norm_bb = Variable(norm_bb).to(device)
        q = Variable(q).to(device)
        if args.dataset == 'mimic' or args.dataset == 'mimic-full' or args.dataset == 'mimic-vqa' or args.dataset == 'vqamed' or args.dataset == 'vqarad':
            target2 = target[1]
            target2 = target2.to(device)
            target_ori = target[2].to(device)
            target = target[0]
            if isinstance(sem_region_feats, list):
                for i in range(len(sem_region_feats)):
                    sem_region_feats[i] = sem_region_feats[i].to(device)
            else:
                sem_region_feats = sem_region_feats.to(device)
            bbx_label = bbx_label.to(device)
        target = Variable(target).to(device)

        pos_emb, sem_adj_matrix, spa_adj_matrix = prepare_graph_variables(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
            args.sem_label_num, device)
        logits, ad_loss, att, node_logits = model(v, norm_bb, q, pos_emb, sem_adj_matrix,
                          spa_adj_matrix, target, node_label, sem_region_feats, bbx_label)

        val_loss = instance_bce_with_logits(logits, target_ori)
        if not args.dryrun:
            wandb.log({'val_loss': val_loss.item()})

        if args.use_graph_classification:
            batch_score2 = compute_score_with_logits(
                            pred2, target2, device, q, idx2ques, question_score2).sum()
            score2 += batch_score2
        # batch_score = compute_score(logits, target_ori, dataset, model,device).sum()
        length = len(dataset.label2ans)
        if args.use_contrastive:
            preds, targets = collect_preds_tgts(length, logits, target_ori, preds, targets, dataset, model, device)
        else:
            preds.append(logits.cpu().detach().numpy())
            targets.append(target_ori.cpu().detach().numpy())

            cls_idx, what_idx, yesno_idx = get_q_type_index(q, cls_set, what_idx_set, yesno_idx_set)
            # cls_idx = torch.tensor(cls_idx)
            what_idx = torch.tensor(what_idx)
            yesno_idx = torch.tensor(yesno_idx)
            #
            # cls_preds.append(torch.index_select(logits.cpu().detach(), 0, cls_idx))
            # cls_targets.append(torch.index_select(target_ori.cpu().detach(), 0, cls_idx))

            # what_preds.append(torch.index_select(logits.cpu().detach(), 0, what_idx))
            # what_targets.append(torch.index_select(target_ori.cpu().detach(), 0, what_idx))

            if len(yesno_idx)>0:
                yesno_pred = torch.index_select(logits.cpu().detach(), 0, yesno_idx)
                yesno_target = torch.index_select(target_ori.cpu().detach(), 0, yesno_idx)
                yesno_preds.append(yesno_pred)
                yesno_targets.append(yesno_target)

        # score += batch_score
        upper_bound += (target.max(1)[0]).sum()
        num_data += logits.size(0)
        if att is not None and 0 < model.module.glimpse\
                and entropy is not None:
            entropy += calc_entropy(att.data)[:model.module.glimpse]
        pbar.update(1)

    # score = score / len(dataloader.dataset)

    targets = np.vstack(targets)
    preds = np.vstack(preds)
    # cls_targets = np.vstack(cls_targets)
    # cls_preds = np.vstack(cls_preds)
    # what_targets = np.vstack(what_targets)
    # what_preds = np.vstack(what_preds)
    if len(yesno_targets)>0:
        yesno_targets = np.vstack(yesno_targets)
        yesno_preds = np.vstack(yesno_preds)

    if args.dataset == 'mimic-full':
        datasetname = 'mimic'
    elif args.dataset == 'mimic-vqa':
        datasetname = 'mimic_vqa'
    elif args.dataset == 'vqamed':
        datasetname = 'vqamed'
    elif args.dataset == 'vqarad':
        datasetname = 'vqarad'


    micro_roc_auc = roc_auc_score(targets, preds, average="micro")
    try:
        macro_roc_auc = roc_auc_score(targets, preds, average="macro")
    except:
        pass
    acc_top = compute_acc(torch.from_numpy(targets), torch.from_numpy(preds))

    if acc_top[0] > best:
        with open('data/' + datasetname + '/saved/targets_' + args.relation_type + '.pkl', 'wb') as f:
            pickle.dump(targets, f)
            print('saved')
        with open('data/' + datasetname + '/saved/preds_' + args.relation_type + '.pkl', 'wb') as f:
            pickle.dump(preds, f)
            print('saved')
        best = acc_top[0]

    acc = 0
    if args.dataset == 'vqamed' or args.dataset == 'vqarad':
        acc = (preds.argmax(1) == targets.argmax(1)).mean()
        print('acc: ', acc)

    what_micro_roc_auc = 0
    what_macro_roc_auc = 0
    if len(yesno_preds)>0:
        yesno_score = compute_yesno_score(yesno_preds, yesno_targets, device)
    else:
        yesno_score = 0
    if not args.dryrun:
        wandb.log({'val_micro_auc':micro_roc_auc,
                   # 'val_macro_auc':macro_roc_auc,
                   'val_yesno_score': yesno_score,
                   'val_acc-1': acc_top[0],
                   'val_acc-3': acc_top[1],
                    'val_acc-5': acc_top[2],
                     'val_acc-10': acc_top[3],
                    'val_acc-15': acc_top[4],
                   })
        try:
            wandb.log({'val_macro_auc':macro_roc_auc})
        except:
            pass
        if args.dataset == 'vqamed' or args.dataset == 'vqarad':
            wandb.log({'val_acc': acc})
    print(
        '\teval AUC(micro): %.2f,  eval yesno_score: %.2f'
        % (100 * micro_roc_auc, 100 * yesno_score))
    print('\teval acc-1: %.2f, acc-3: %.2f, acc-5: %.2f, acc-10: %.2f, acc-15: %.2f'
          % (100 * acc_top[0], 100 * acc_top[1], 100 * acc_top[2], 100 * acc_top[3], 100 * acc_top[4]))
    print('\tbest acc: %.2f' % (100 * best))
    try:
        print('\teval AUC(macro): %.2f' % (100 * macro_roc_auc))
    except:
        pass

    if args.use_graph_classification:
        score2 = score2 / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    model.train()
    # print_question_score(question_score, question_score2, name='eval')
    return (micro_roc_auc, acc, yesno_score, what_micro_roc_auc, what_macro_roc_auc), upper_bound, entropy, score2, best


def calc_entropy(att):
    # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p + eps).log()).sum(2).sum(0)  # g
