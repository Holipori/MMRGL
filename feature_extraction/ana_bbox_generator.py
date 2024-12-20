# comparing to bbox_generator.py, this file is for anatomical bboxes. and some code is optimized.

import json
import os
import pickle
from pathlib import Path
from typing import Any, Union, Dict, List
from math import ceil
from numpy import ndarray
import math
import h5py
from os.path import exists
import yaml

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm

from get_bbox_id import inference
from train_vindr import get_vindr_shape, get_vindr_label2id
from train_anatomy import get_kg2

setup_logger()

os.environ['CUDA_LAUNCH_BLOCKING']='1'


# --- setup ---

def format_pred(labels: ndarray, boxes: ndarray, scores: ndarray) -> str:
    pred_strings = []
    for label, score, bbox in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = bbox.astype(np.int64)
        pred_strings.append(f"{label} {score} {xmin} {ymin} {xmax} {ymax}")
    return " ".join(pred_strings)



def predict_batch(predictor: DefaultPredictor, im_list: List[ndarray]) -> List:
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        inputs_list = []
        for original_image in im_list:
            # Apply pre-processing to image.
            if predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            inputs_list.append(inputs)
        predictions = predictor.model(inputs_list)
    return predictions



def get_mimic_dict(full=True):
    datadir = '/home/xinyue/dataset/mimic-cxr-png/'
    datadict = []
    name = '../data/mimic_shape_full.pkl'
    with open(name, 'rb') as f:
        mimic_dataset = pickle.load(f)
    for i,row in enumerate(mimic_dataset):
        record = {}
        filename = datadir + row['image'] + '.png'
        record["file_name"] = filename
        record["image_id"] = row['image']
        record["height"] = 1024
        record["width"] = 1024
        datadict.append(record)

    return datadict


def save_yaml(filepath: Union[str, Path], content: Any, width: int = 120):
    with open(filepath, "w") as f:
        yaml.dump(content, f, width=width)


def load_yaml(filepath: Union[str, Path]) -> Any:
    with open(filepath, "r") as f:
        content = yaml.full_load(f)
    return content

def save_features(mod, inp, outp):
    feats = inp[0]
    for i in range(batch_size):
        features.append(feats[i*1000: (i+1)*1000])
    predictions.append(outp)
    # feature = inp[0]
    # prediction = outp

def save_features1(mod, inp, outp):
    proposals.append(inp[2])
    # prposal = inp[2]

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou
def get_center(bb1):
    c1 = [(bb1[2] + bb1[0])/2, (bb1[3] + bb1[1])/2]
    return c1

def get_distance(bb1, bb2):
    c1 = get_center(bb1)
    c2 = get_center(bb2)
    dx = np.abs(c2[0] - c1[0])
    dy = np.abs(c2[1] - c1[1])
    d = np.sqrt(np.square(dx) + np.square(dy))
    return d

def get_angle(coor):
    x1,y1, x2, y2 = coor
    angle = math.atan2(  y2-y1, x2-x1 ) /math.pi *180
    if angle < 0:
        angle += 360
    return angle

def cal_angle(bb1,bb2):
    c1 = get_center(bb1)
    c2 = get_center(bb2)
    return get_angle(c1+c2)

def bbox_relation_type(bb1, bb2, lx=1024, ly=1024):
    if bb1[0]< bb2[0] and bb1[1]< bb2[1] and bb1[2]> bb2[2] and bb1[3]>bb2[3]:
        return 1
    elif bb1[0]> bb2[0] and bb1[1]> bb2[1] and bb1[2]< bb2[2] and bb1[3]<bb2[3]:
        return 2
    elif get_iou(bb1, bb2) >= 0.5:
        return 3
    elif get_distance(bb1, bb2) >= (lx+ly)/3:
        return 0
    angle = cal_angle(bb1,bb2)
    return math.ceil(angle/45) + 3

def reverse_type(type):
    if type == 0:
        return 0
    elif type == 1:
        return 2
    elif type == 2:
        return 1
    elif type == 3:
        return 3
    elif type == 4:
        return 8
    elif type == 5:
        return 9
    elif type == 6:
        return 10
    elif type == 7:
        return 11
    elif type == 8:
        return 4
    elif type == 9:
        return 5
    elif type == 10:
        return 6
    elif type == 11:
        return 7



def get_adj_matrix(bboxes, adj_matrix = None):

    num_pics = len(bboxes)
    n = len(bboxes[0])
    if adj_matrix is None:
        adj_matrix = np.zeros([num_pics,100,100],int)
    for idx in tqdm(range(num_pics)):
        bbs = bboxes[idx]
        for i in range(n):
            for j in range(i,n):
                if adj_matrix[idx,i,j] != 0:
                    continue
                type = bbox_relation_type(bbs[i],bbs[j])
                adj_matrix[idx,i,j] = type
                adj_matrix[idx,j,i] = reverse_type((type))
    return adj_matrix

def save_h5(final_features, normalized_bboxes,bboxes, pos_boxes, adj_matrix, test_topk_per_image, pred_classes, full=True, times=0, length = 100):
    filename = './output/mimic_ana_box/ana_bbox_features_full.hdf5'
    if times == 0:
        h5f = h5py.File(filename, 'w')
        image_features_dataset = h5f.create_dataset("image_features", (length, test_topk_per_image, 1024),
                                                    maxshape=(None, test_topk_per_image, 1024),
                                                    chunks=(100, test_topk_per_image, 1024),
                                                    dtype='float32')
        spatial_features_dataset = h5f.create_dataset("spatial_features", (length, test_topk_per_image, 6),
                                                    maxshape=(None, test_topk_per_image, 6),
                                                    chunks=(100, test_topk_per_image, 6),
                                                    dtype='float64')
        image_bb_dataset = h5f.create_dataset("image_bb", (length, test_topk_per_image, 4),
                                                    maxshape=(None, test_topk_per_image, 4),
                                                    chunks=(100, test_topk_per_image, 4),
                                                    dtype='float32')
        pos_boxes_dataset = h5f.create_dataset("pos_boxes", (length, 2),
                                                    maxshape=(None, 2),
                                                    chunks=(100, 2),
                                                    dtype='int64')
        image_adj_matrix_dataset = h5f.create_dataset("image_adj_matrix", (length, 100, 100),
                                                    maxshape=(None, 100, 100),
                                                    chunks=(100, 100, 100),
                                                    dtype='int64')
        bbox_label_dataset = h5f.create_dataset("bbox_label", (length, test_topk_per_image),
                                                         maxshape=(None, test_topk_per_image),
                                                         chunks=(100, test_topk_per_image),
                                                         dtype='int64')
    else:
        h5f = h5py.File(filename, 'a')
        image_features_dataset = h5f['image_features']
        spatial_features_dataset = h5f['spatial_features']
        image_bb_dataset = h5f['image_bb']
        pos_boxes_dataset = h5f['pos_boxes']
        image_adj_matrix_dataset = h5f['image_adj_matrix']
        bbox_label_dataset = h5f['bbox_label']

    if len(final_features) != length:
        adding = len(final_features)
    else:
        adding = length
    image_features_dataset.resize([times*length+adding, test_topk_per_image, 1024])
    image_features_dataset[times*length:times*length+adding] = final_features

    spatial_features_dataset.resize([times*length+adding, test_topk_per_image, 6])
    spatial_features_dataset[times*length:times*length+adding] = normalized_bboxes

    image_bb_dataset.resize([times*length+adding, test_topk_per_image, 4])
    image_bb_dataset[times*length:times*length+adding] = bboxes

    pos_boxes_dataset.resize([times*length+adding, 2])
    pos_boxes_dataset[times*length:times*length+adding] = pos_boxes

    image_adj_matrix_dataset.resize([times*length+adding, 100, 100])
    image_adj_matrix_dataset[times*length:times*length+adding] = adj_matrix

    bbox_label_dataset.resize([times * length + adding, test_topk_per_image])
    bbox_label_dataset[times * length:times * length + adding] = pred_classes

    h5f.close()

if __name__ == '__main__':
    dataset = 'mimic'

    gold_weights=True
    if gold_weights:
        dd = get_kg2()
        category = {}
        for item in dd:
            category[item] = len(category)
    else:
        with open('./dictionary/category_ana.pkl', "rb") as tf:
            category = pickle.load(tf)

    thing_classes = list(category)
    category_name_to_id = category


    #===============================



    cfg = get_cfg()
    cfg.OUTPUT_DIR = 'results/mimic_ana_bbx'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("mimic_ana_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 200000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(category)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    ### --- Inference & Evaluation ---
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = str("checkpoints/model_final_for_anatomy_gold.pth")
    print("Original thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)  # 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set a custom testing threshold
    print("Changed  thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    predictor = DefaultPredictor(cfg)

    # hook
    features = []
    proposals = []
    predictions = []


    ### hook ##########
    layer_to_hook1 = 'roi_heads'
    layer_to_hook2 = 'box_predictor'
    layer_to_hook3 = 'fc_relu2'
    for name, layer in predictor.model.named_modules():
        if name == layer_to_hook1:
            layer.register_forward_hook(save_features1)
            for name2, layer2 in layer.named_modules():
                if name2 == layer_to_hook2:
                    layer2.register_forward_hook(save_features)


    DatasetCatalog.register("mimic", get_mimic_dict)
    MetadataCatalog.get("mimic").set(thing_classes=thing_classes)
    dataset_dicts = get_mimic_dict(full=True)

    full = True
    #### test/val
    # metadata = MetadataCatalog.get("mimic")

    results_list = []
    index = 0
    out_ids = []
    final_features = []
    bboxes = []
    normalized_bboxes = []
    pos_boxes = []
    pred_classes = []
    n = 0
    test_topk_per_image = 26

    batch_size = 1 # batch size must be dividable by length
    length = 5000
    times = 0
    flag = 0
    order = True
    if order:
        pre_extract_num = 100




    resume = False # remember to check before running
    if resume:
        stopped_batch_num = 72500  # the number you see in the terminal when stooped
        # stopped_batch_num = 75000
        stopped_img_num = stopped_batch_num * batch_size
        continue_i = (stopped_img_num - length) / batch_size
        times = int((stopped_img_num - length)/length)
        n = int(continue_i * test_topk_per_image)
    for i in tqdm(range(ceil(len(dataset_dicts) / batch_size))):
        if i == 452:
            print('here')
        if resume:
            if i < continue_i:
                continue
        inds = list(range(batch_size * i, min(batch_size * (i + 1), len(dataset_dicts))))
        dataset_dicts_batch = [dataset_dicts[i] for i in inds]
        # print(dataset_dicts_batch[0]['file_name'])

        im_list = [cv2.imread(d["file_name"]) for d in dataset_dicts_batch]
        outputs_list = predict_batch(predictor, im_list)

        if order:
            out_ids = (
                inference(predictions[0], proposals[0], predictor.model.roi_heads.box_predictor.box2box_transform,
                          pre_extract_num))
        else:
            out_ids = (inference(predictions[0], proposals[0], predictor.model.roi_heads.box_predictor.box2box_transform, test_topk_per_image))
        predictions = []
        proposals = []
        for feats, ids, dicts, outputs in zip(features, out_ids, dataset_dicts_batch,outputs_list):
            ids = (ids / len(category)).type(torch.long)
            if order:
                feat = []
                bb = []
                pred_class = []
                for id in range(test_topk_per_image):
                    for idx in range(pre_extract_num):
                        if outputs['instances']._fields['pred_classes'][idx] == id:
                            bb.append(outputs['instances']._fields['pred_boxes'].tensor[idx].cpu())
                            pred_class.append(outputs['instances']._fields['pred_classes'][idx].cpu())
                            feat.append(feats[ids[idx]].cpu())
                            break
                        elif idx == pre_extract_num-1:
                            bb.append(torch.zeros(4))
                            pred_class.append(torch.zeros(1))
                            feat.append(torch.zeros(1024))
                feat = torch.stack(feat)
                bb = torch.stack(bb)
                pred_class = torch.tensor(pred_class)
                final_features.append(np.array(feat))
                bb = np.array(bb)
                pred_class = np.array(pred_class)
            else:
                feat = feats[ids].cpu()
                final_features.append(np.array(feat))
                 # f[dicts['image_id']] = feat
                bb = np.array(outputs['instances']._fields['pred_boxes'].tensor.cpu())[:test_topk_per_image]
                pred_class = np.array(outputs['instances']._fields['pred_classes'].cpu())[:test_topk_per_image]
            bboxes.append(bb)
            pred_classes.append(pred_class)
            normalized_bb = np.concatenate((bb/1024,np.zeros((bb.shape[0],2))),1)
            normalized_bboxes.append(normalized_bb)
            pos_boxes.append([n, n+len(normalized_bb)])
            n += len(normalized_bb)
        features = []
        if len(final_features) == length or i == ceil(len(dataset_dicts) / batch_size)-1:
            final_features = np.array(final_features)
            bboxes = np.array(bboxes)
            pred_classes = np.array(pred_classes)
            normalized_bboxes = np.array(normalized_bboxes)
            pos_boxes = np.array(pos_boxes)
            adj_matrix = get_adj_matrix(bboxes)
            save_h5(final_features, normalized_bboxes,bboxes, pos_boxes, adj_matrix,test_topk_per_image, pred_classes, full=full, times= times, length=length)
            final_features = []
            bboxes = []
            normalized_bboxes = []
            pos_boxes = []
            pred_classes = []
            times += 1



    print('finished writing')


# this file use gold standard to generate bounding box and features

