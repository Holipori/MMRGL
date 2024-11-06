"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
from __future__ import print_function
import os
import json
import pickle
import numpy as np
from utils import utils
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from utils.compute_spa_adj import get_adj_matrix

COUNTING_ONLY = False


# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word): # seems not used
        sentence = sentence.lower()
        sentence = sentence.replace(',', '')\
            .replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK
                # for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open('data/dictionary.pkl', 'rb')) # this is for make sure the first 34 order is the same as old cls questions

        word2idx2, idx2word2 = pickle.load(open(path, 'rb'))
        for w2i in word2idx2:
            if w2i not in word2idx:
                word2idx[w2i] = word2idx2[w2i]
                idx2word.append(w2i)

        d = cls(word2idx, idx2word)
        return d

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
        disease_lib_path = '/home/xinyue/SRDRL/data/question_gen/lib/disease_lib.csv'
        disease_lib = pd.read_csv(disease_lib_path)
        disease_names = disease_lib['official_name'].tolist()

        ana_names = list(self.get_kg_ana_only().keys())

        return disease_names+ana_names

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if answer is not None:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'Questions/v2_OpenEnded_mscoco_%s_questions.json' %
        (name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    # train, val
    if 'test' != name[:4]:
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY \
               or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, answer))
    # test2015
    else:
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY \
               or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id],
                                             question, None))

    return entries



def _find_coco_id(vgv, vgv_id):
    for v in vgv:
        if v['image_id'] == vgv_id:
            return v['coco_id']
    return None


class mimicfull_VQAFeatureDataset(Dataset):
    def __init__(self, name, dataset, dictionary, relation_type, dataroot='data',
                 adaptive=False, pos_emb_dim=64, nongt_dim=36, pure_classification = True, args=None):
        super(mimicfull_VQAFeatureDataset, self).__init__()
        self.name = name
        self.test_spa_adj_thr = args.test_spa_adj_thr if args is not None else 0
        self.dataset = dataset
        # assert name in ['train', 'val']
        if dataset == 'mimic-full':
            ans2label_path = os.path.join(dataroot, 'mimic',
                                          'mimic_ans2label_full.pkl')
            label2ans_path = os.path.join(dataroot, 'mimic',
                                          'mimic_label2ans_full.pkl')
        elif dataset == 'mimic-vqa':
            ans2label_path = os.path.join(dataroot, 'mimic_vqa',
                                          'mimic_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'mimic_vqa',
                                          'mimic_label2ans.pkl')
        elif dataset == 'vqamed':
            ans2label_path = os.path.join(dataroot, 'vqamed',
                                          'vqamed_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'vqamed',
                                          'vqamed_label2ans.pkl')
        elif dataset == 'vqarad':
            ans2label_path = os.path.join(dataroot, 'vqarad',
                                          'vqarad_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'vqarad',
                                          'vqarad_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        self.relation_type = relation_type
        self.adaptive = False
        self.pure_classification = pure_classification
        self.node_size = self.get_node_size()

        if 'mimic' in dataset:
            node_labels_path = 'data/mimic_vqa/node_labels.pkl'
            node_label2id_path = 'data/mimic_vqa/node_label2id.pkl'
            self.node_labels = pickle.load(open(node_labels_path, 'rb'))
            self.ori_node_label2id = pickle.load(open(node_label2id_path, 'rb'))

            self.new_node_label2id = self.dictionary.node_names()
            # self.node_mapping = {}
            # for node_name in self.ori_node_label2id:
            #     self.node_mapping[self.ori_node_label2id[node_name]] = self.new_node_label2id[node_name]


        h5_dataroot = dataroot+"/mimic"

        if 'mimic' in dataset:
            # h5_path = os.path.join(h5_dataroot, 'cmb_bbox_features_full.hdf5') # 60 bbx # this file is feature file. doesn't need to be updated every time.
            # h5_path = os.path.join('/home/xinyue/faster-rcnn', 'output/mimic_ana_box/cmb_bbox_features_full.hdf5') # 51 bbx in old server
            h5_path = 'data/mimic/cmb_bbox_features_full.hdf5' # 51 bbx in new server
        elif dataset == 'vqamed':
            h5_dataroot = dataroot + "/vqamed"
            h5_path = os.path.join(h5_dataroot, 'cmb_bbox_features.hdf5')
        elif dataset == 'vqarad':
            h5_dataroot = dataroot + "/vqarad"
            h5_path = os.path.join(h5_dataroot, 'cmb_bbox_features.hdf5')
        # h5_path = '/home/xinyue/faster-rcnn/output/mimic_ana_box/ana_bbox_features_full.hdf5' # 51 bbx

        print('loading features from h5 file %s' % h5_path)
        hf = h5py.File(h5_path, 'r')
        self.hf = hf
        # self.features = np.array(hf.get('image_features')).reshape(-1,1024)
        self.features = hf['image_features']
        self.normalized_bb = hf['spatial_features']
        self.bb = hf['image_bb']
        self.bb_label = hf['bbox_label']
        if "semantic_adj_matrix" in hf.keys() \
           and self.relation_type == "semantic" or self.relation_type == "my_semantic":
            self.semantic_adj_matrix = hf['semantic_adj_matrix']
            print("Loaded semantic adj matrix from file...",
                  self.semantic_adj_matrix.shape)
        else:
            self.semantic_adj_matrix = None
            print("Setting semantic adj matrix to None...")
        if "image_adj_matrix" in hf.keys() and self.relation_type == "spatial":
            self.spatial_adj_matrix = hf['image_adj_matrix']
            print("Loaded spatial adj matrix from file...",
                  self.spatial_adj_matrix.shape)
        else:
            self.spatial_adj_matrix = None
            print("Setting spatial adj matrix to None...")

        self.sem_region_feature = args.sem_region_feature if args is not None else False
        if self.sem_region_feature and dataset == 'mimic-vqa':
            filename = 'data/mimic_vqa/gradcam/gradcam_bbxs_{}.h5'.format(name)
            grad_hf = h5py.File(filename, 'r')
            self.gradcam_features = grad_hf['gradcam_feats']
            self.gradcam_bb = grad_hf['gradcam_bbxs']
            self.gradcam_logits = grad_hf['gradcam_logits']

        self.pos_boxes = None
        if self.adaptive:
            self.pos_boxes = hf['pos_boxes']
        if dataset == 'mimic-vqa':
            if name == 'train':
                dataset_path = '/mimic_vqa/mimic_dataset_train.pkl'
            elif name == 'val':
                dataset_path = '/mimic_vqa/mimic_dataset_val.pkl'
            elif name == 'test':
                dataset_path = '/mimic_vqa/mimic_dataset_test.pkl'
        elif dataset == 'vqamed':
            if name == 'train':
                dataset_path = '/vqamed/vqamed_dataset_train.pkl'
            elif name == 'val':
                dataset_path = '/vqamed/vqamed_dataset_val.pkl'
            elif name == 'test':
                dataset_path = '/vqamed/vqamed_dataset_val.pkl'
        elif dataset == 'vqarad':
            if name == 'train':
                dataset_path = '/vqarad/vqarad_dataset_train.pkl'
            elif name == 'val':
                dataset_path = '/vqarad/vqarad_dataset_test.pkl'
            elif name == 'test':
                dataset_path = '/vqarad/vqarad_dataset_test.pkl'
        with open(dataroot + dataset_path, 'rb') as f:
            self.entries = pickle.load(f)

        self.entries = self.entries

        print('tokenizing')
        self.tokenize()
        print('tensorizing')
        self.tensorize()
        self.nongt_dim = nongt_dim
        self.emb_dim = pos_emb_dim
        self.v_dim = self.features.shape[-1]
        self.s_dim = self.normalized_bb.shape[-1]


        # import csv
        # csv_columns = ['subject_id', 'study_id','dicom_id','question', 'answer', 'height','width','image', 'answer2','q_token', 'a_token']
        # with open('entries.csv', 'w') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        #     writer.writeheader()
        #     for data in self.entries:
        #         writer.writerow(data)

    def get_node_size(self):
        disease_lib_path = 'dataset_construction/lib/disease_lib_llm_full.csv'
        disease_lib = pd.read_csv(disease_lib_path)
        disease_names = disease_lib['official_name'].tolist()

        ana_names = list(self.get_kg_ana_only().keys())

        return len(disease_names+ana_names)
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
    def enrich_answer(self, anss):
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

    def sub_tokenize(self, text, max_length=14):
        tokens = self.dictionary.tokenize(text, False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad to the back of the sentence
            padding = [self.dictionary.padding_idx] * \
                      (max_length - len(tokens))
            tokens = tokens + padding
        utils.assert_eq(len(tokens), max_length)
        return tokens

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            # q_tokens = self.dictionary.tokenize(entry['question'], False)
            # q_tokens = q_tokens[:max_length]
            # if len(q_tokens) < max_length:
            #     # Note here we pad to the back of the sentence
            #     padding = [self.dictionary.padding_idx] * \
            #               (max_length - len(q_tokens))
            #     q_tokens = q_tokens + padding
            # utils.assert_eq(len(q_tokens), max_length)
            # entry['q_token'] = q_tokens

            entry['q_token'] = self.sub_tokenize(entry['question'], max_length=max_length)
            #######===================================
            entry['a_token'] = self.sub_tokenize(self.enrich_answer(entry['answer']['answer']), max_length=max_length)


    def tensorize(self):
        # self.features = torch.from_numpy(self.features)
        # self.normalized_bb = torch.from_numpy(self.normalized_bb)
        # self.bb = torch.from_numpy(self.bb)
        # if self.semantic_adj_matrix is not None:
        #     self.semantic_adj_matrix = torch.from_numpy(
        #                                 self.semantic_adj_matrix).double()
        # if self.spatial_adj_matrix is not None:
        #     self.spatial_adj_matrix = torch.from_numpy(
        #                                 self.spatial_adj_matrix).double()
        # if self.pos_boxes is not None:
        #     self.pos_boxes = torch.from_numpy(self.pos_boxes)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            entry['a_token'] = torch.from_numpy(np.array(entry['a_token']))

            answer = entry['answer']
            # answer2 = entry['answer2']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                # labels2 = np.array(answer2['labels'])
                # scores2 = np.array(answer2['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                    # labels2 = torch.from_numpy(labels2)
                    # scores2 = torch.from_numpy(scores2)
                    # entry['answer2']['labels'] = labels2
                    # entry['answer2']['scores'] = scores2
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def get_img(self, img_id):
        if 'mimic' in self.dataset:
            img_path = os.path.join('/home/xinyue/dataset/mimic-cxr-png/', str(img_id) + '.png')
        elif self.dataset == 'vqarad':
            img_path = os.path.join('/drive/xinyue/dataset/manually_select_vqarad', str(img_id) + '.jpg')
        elif self.dataset == 'vqamed':
            img_path = os.path.join('/drive/xinyue/dataset/manually_select_imageclef', str(img_id) + '.jpg')
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512))
        data_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        img = data_transform(img)
        return img

    def __getitem__(self, index):
        # if index == 49:
        #     print('a')
        entry = self.entries[index]
        raw_question = entry["question"]
        image_id = entry["dicom_id"]

        if 'mimic' in self.dataset:
            # img = self.get_img(image_id)
            img = torch.tensor(0)
            node_label = self.node_labels[str(entry['study_id'])]
            node_label = torch.from_numpy(np.array(node_label))
            node_label_onehot = torch.zeros(self.node_size)
            if len(node_label) > 0:
                node_label_onehot.scatter_(0, node_label, 1)
        elif self.dataset == 'vqarad' or self.dataset == 'vqamed':
            img = self.get_img(image_id)
            node_label_onehot = torch.tensor(0)
        else:
            img = torch.tensor(0)
            node_label_onehot = torch.tensor(0)




        question = entry['q_token']
        # bbx_labels = self.bbox_labels[index]
        # question_id = entry['question_id']
        question_id = 0
        if self.spatial_adj_matrix is not None:
            spatial_adj_matrix = torch.from_numpy(self.spatial_adj_matrix[entry["image"]]).double()
        else:
            spatial_adj_matrix = torch.zeros(1).double()
        if self.semantic_adj_matrix is not None:
            semantic_adj_matrix = torch.from_numpy(self.semantic_adj_matrix[entry["image"]]).double()
        else:
            semantic_adj_matrix = torch.zeros(1).double()
        if not self.adaptive: # yes here
            # fixed number of bounding boxes
            features = torch.from_numpy(self.features[entry['image']])
            normalized_bb = torch.from_numpy(self.normalized_bb[entry['image']])
            bb = torch.from_numpy(self.bb[entry["image"]])
        else: # no
            features = self.features[
                self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            normalized_bb = self.normalized_bb[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            bb = self.bb[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]

        if self.test_spa_adj_thr != 0:
            spatial_adj_matrix = torch.from_numpy(get_adj_matrix(np.array(bb.unsqueeze(0)), thr= self.test_spa_adj_thr)).double().squeeze()
        # if self.semantic_adj_matrix is not None:
        #     features = torch.cat([features, torch.mean(features, 0).unsqueeze(0)])
        #     normalized_bb = torch.cat([normalized_bb, torch.tensor([[0, 0, 1, 1, 0, 0]]).type(torch.float64)])
        #     bb = torch.cat([bb, torch.tensor([[0, 0, 1024, 1024]]).type(torch.float32)])

        sem_region_logits = torch.zeros(1)
        if self.sem_region_feature:
            sem_region_logits = (torch.from_numpy(np.average(self.gradcam_features[index], 1)),  torch.from_numpy(self.gradcam_logits[index]))
            # sem_region_logits = (torch.from_numpy(self.gradcam_features[index][:,0,:]),  torch.from_numpy(self.gradcam_logits[index]))
        bbx_label = self.bb_label[entry['image']]

        answer = entry['answer']
        # answer2 = entry['answer2']
        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            # labels2 = answer2['labels']
            # scores2 = answer2['scores']
            target = entry['a_token']  # words of answer
            target_ori = torch.zeros(self.num_ans_candidates) # there seems be no difference between target_ori and target2 right now. but they are designed to be different for vqa/classification
            if self.pure_classification:
                target2 = torch.zeros(self.num_ans_candidates)
            else:
                target2 = torch.zeros(self.num_ans_candidates-1)
            if labels is not None:
                target_ori.scatter_(0, labels, scores)
            # if labels2 is not None:
            #     try:
            #         target2.scatter_(0, labels2, scores2)
            #     except:
            #         print('a')
            return features, normalized_bb, question, (target,target2, target_ori),\
                question_id, image_id, bb, spatial_adj_matrix,\
                semantic_adj_matrix, node_label_onehot, img, sem_region_logits, bbx_label

        else:
            return features, normalized_bb, question, question_id,\
                question_id, image_id, bb, spatial_adj_matrix,\
                semantic_adj_matrix

    def __len__(self):
        return len(self.entries)


