# combine anatomical bbox and disease bbox
import h5py
import numpy as np
from ana_bbox_generator import get_adj_matrix
from tqdm import tqdm
import pickle
from train_vindr import get_vindr_label2id

def get_kg():
    kg_dict = {}
    # anatomical part
    kg_dict['Right lung'] = 'Lung'
    kg_dict['Right upper lung zone'] = 'Lung'
    kg_dict['Right mid lung zone'] = 'Lung'
    kg_dict['Right lower lung zone'] = 'Lung'
    kg_dict['Hilar Area of the Right Lung'] = 'Lung'
    kg_dict['Apical zone of right lung'] = 'Lung'
    kg_dict['right costophrenic sulcus;;Right costodiaphragmatic recess'] = 'Pleural'
    kg_dict['right cardiophrenic sulcus'] = 'Pleural'
    kg_dict['Right hemidiaphragm'] = 'Pleural' # probably
    kg_dict['Left lung'] = 'Lung'
    kg_dict['Left upper lung zone'] = 'Lung'
    kg_dict['Left mid lung zone'] = 'Lung'
    kg_dict['Left lower lung zone'] = 'Lung'
    kg_dict['Hilar Area of the Left Lung'] = 'Lung'
    kg_dict['Apical zone of left lung'] = 'Lung'
    kg_dict['left costophrenic sulcus;;Left costodiaphragmatic recess'] = 'Pleural'
    kg_dict['Left hemidiaphragm'] = 'Pleural' # probably

    kg_dict['Trachea&&Main Bronchus'] = 'Lung'
    kg_dict['Vertebral column'] = 'Spine'
    kg_dict['Right clavicle'] = 'Bone'
    kg_dict['Left clavicle'] = 'Bone'
    kg_dict['Aortic arch structure'] = 'Heart'
    kg_dict['Mediastinum'] = 'Mediastinum'
    kg_dict['Superior mediastinum'] = 'Mediastinum'
    kg_dict['Superior vena cava structure'] = 'Heart'
    kg_dict['Cardiac shadow viewed radiologically;;Heart'] = 'Heart'
    kg_dict['Structure of left margin of heart'] = 'Heart'
    kg_dict['Right border of heart viewed radiologically'] = 'Heart'
    kg_dict['cavoatrial'] = 'Heart'
    kg_dict['Right atrial structure'] = 'Heart'
    kg_dict['Descending aorta'] = 'Heart'
    kg_dict['Structure of carina'] = 'Lung'

    kg_dict['Structure of left upper quadrant of abdomen'] = 'Abdomen' # new group
    kg_dict['Structure of right upper quadrant of abdomen'] = 'Abdomen'# new group
    kg_dict['Abdominal cavity'] = 'Abdomen'# new group
    kg_dict['left cardiophrenic sulcus'] = 'Pleural'

    # disease part
    kg_dict['Aortic enlargement'] = 'Heart'
    kg_dict['Atelectasis'] = 'Lung'
    kg_dict['Calcification'] = 'Bone'
    kg_dict['Cardiomegaly'] = 'Heart'
    kg_dict['Consolidation'] = 'Lung'
    kg_dict['ILD'] = 'Lung'
    kg_dict['Infiltration'] = 'Lung' # probably https://en.wikipedia.org/wiki/Pulmonary_infiltrate
    kg_dict['Lung Opacity'] = 'Lung'
    kg_dict['Nodule/Mass'] = 'Lung' # probably
    kg_dict['Other lesion'] = 'Lung' # probably
    kg_dict['Pleural effusion'] = 'Pleural'
    kg_dict['Pleural thickening'] = 'Pleural'
    kg_dict['Pneumothorax'] = 'Pleural'
    kg_dict['Pulmonary fibrosis'] = 'Lung'
    kg_dict['Clavicle fracture'] = 'Bone'
    kg_dict['Emphysema'] = 'Lung'
    kg_dict['Enlarged PA'] = 'Heart'
    kg_dict['Lung cavity'] = 'Lung'
    kg_dict['Lung cyst'] = 'Lung'
    kg_dict['Mediastinal shift'] = 'Mediastinum'
    kg_dict['Rib fracture'] = 'Bone'
    kg_dict['Fracture'] = 'Bone'




    # this part is not using.
    kg_idx = {}
    kg_idx['Lung'] = 1
    kg_idx['Pleural'] = 2
    kg_idx['Spine'] = 3
    kg_idx['Bone'] = 4
    kg_idx['Mediastinum'] = 5
    kg_idx['Heart'] = 6
    kg_idx['Abdomen'] = 7


    return kg_dict, kg_idx




def save_h5(final_features, normalized_bboxes,bboxes, pos_boxes, adj_matrix, test_topk_per_image, pred_classes, semantic_adj, full=True, times=0, length = 100):
    filename = './output/mimic_ana_box/cmb_bbox_features_full.hdf5'
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
        semantic_adj_matrix_dataset = h5f.create_dataset("semantic_adj_matrix", (length, 100, 100),
                                                         maxshape=(None, 100, 100),
                                                         chunks=(100, 100, 100),
                                                         dtype='int64')
        # semantic_adj_matrix_dataset2 = h5f.create_dataset("semantic_adj_matrix2", (length, 100, 100),
        #                                                 maxshape=(None, 100, 100),
        #                                                  dtype='float64')
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
        semantic_adj_matrix_dataset = h5f['semantic_adj_matrix']
        # semantic_adj_matrix_dataset2 = h5f['semantic_adj_matrix2']
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

    semantic_adj_matrix_dataset.resize([times * length + adding, 100, 100])
    semantic_adj_matrix_dataset[times * length:times * length + adding] = semantic_adj

    # semantic_adj_matrix_dataset2.resize([times * length + adding, 100, 100])
    # semantic_adj_matrix_dataset2[times * length:times * length + adding] = semantic_adj2

    bbox_label_dataset.resize([times * length + adding, test_topk_per_image])
    bbox_label_dataset[times * length:times * length + adding] = pred_classes

    h5f.close()

def get_semantic_adj(pred_classes_ana, pred_classes_di, ana_thing_classes, di_thing_classes,kg_dict, small_adj, small_name2index, kg_idx):
    '''

    :param pred_classes_ana: 36
    :param pred_classes_di: 25
    :return:
    '''

    pred_classes_di += 36

    for i in range(len(di_thing_classes)):
        # di_thing_classes[i] = di_thing_classes[i].lower().replace(' ', '_')
        if 'fracture' in di_thing_classes[i]:
            di_thing_classes[i] = 'fracture'

    thing_classes = ana_thing_classes + di_thing_classes
    pred_classes = np.hstack((pred_classes_ana,pred_classes_di))[:test_topk_per_image]

    ana_thing_classes_set = set(ana_thing_classes)
    di_thing_classes_set = set(di_thing_classes)
    adj_matrix = np.zeros([100, 100], int)
    for i in range(test_topk_per_image):
        for j in range(i,test_topk_per_image):
            try:
                if kg_dict[thing_classes[pred_classes[i]]] == kg_dict[thing_classes[pred_classes[j]]]:
                    # adj_matrix[i,j] = kg_idx[kg_dict[thing_classes[pred_classes[i]]]]
                    # adj_matrix[j, i] = adj_matrix[i,j]
                    if thing_classes[pred_classes[i]] in ana_thing_classes_set and thing_classes[pred_classes[j]] in di_thing_classes_set or thing_classes[pred_classes[j]] in ana_thing_classes_set and thing_classes[pred_classes[i]] in di_thing_classes_set:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1
                if thing_classes[pred_classes[i]].lower().replace(' ', '_') in small_name2index and thing_classes[pred_classes[j]].lower().replace(' ', '_') in small_name2index:
                    value = max(small_adj[small_name2index[thing_classes[pred_classes[i]].lower()], small_name2index[thing_classes[pred_classes[j]].lower()]],adj_matrix[i,j] )
                    adj_matrix[i,j] = value
                    adj_matrix[j, i] = value
            except:
                pass
    # adj_matrix[test_topk_per_image,:test_topk_per_image]  = 1
    # adj_matrix[:test_topk_per_image, test_topk_per_image] = 1

    return adj_matrix

def cmb_pred_classes(pred_classes_ana, pred_classes_di):
    pred_classes_di += 36

    thing_classes = ana_thing_classes + di_thing_classes
    pred_classes = np.hstack((pred_classes_ana, pred_classes_di))[:test_topk_per_image]

    return pred_classes

def get_countingAdj_name2index():

    with open('dictionary/mimic_ans2label_full.pkl', 'rb') as f:
        ans2label = pickle.load(f)
    return ans2label


def get_semantic_adj2(pred_classes_ana, pred_classes_di, ana_thing_classes, di_thing_classes, small_adj,small_name2index):
    # deprecated. because semantic graph has been fused up into a single graph with different label. label is determined by the relationships.
    pred_classes = cmb_pred_classes(pred_classes_ana, pred_classes_di)
    thing_classes = ana_thing_classes + di_thing_classes

    # thing_classes = ana_thing_classes + di_thing_classes
    adj_matrix = np.zeros([100, 100], float)
    for i in range(test_topk_per_image):
        if thing_classes[pred_classes[i]].lower() in small_name2index:
            for j in range(i, test_topk_per_image):
                if thing_classes[pred_classes[j]].lower() in small_name2index:
                    adj_matrix[i,j] = small_adj[small_name2index[thing_classes[pred_classes[i]].lower()], small_name2index[thing_classes[pred_classes[j]].lower()]]
                    adj_matrix[j, i] = adj_matrix[i,j]
    return adj_matrix




h5_path1 = '/home/xinyue/faster-rcnn/output/mimic_ana_box/ana_bbox_features_full.hdf5'
h5_path2 = '/home/xinyue/faster-rcnn/output/mimic_box_coords/bbox_disease_features_by_coords.hdf5'

f1 = h5py.File(h5_path1,'r')   #打开h5文件
f2 = h5py.File(h5_path2,'r')


di_thing_classes = list(get_vindr_label2id())

with open('dictionary/category_ana.pkl', "rb") as tf:
    ana_thing_classes = list(pickle.load(tf))

with open('dictionary/GT_counting_adj.pkl', "rb") as tf:
    small_counting_adj = pickle.load(tf)
    for i in range(len(small_counting_adj)):
        small_counting_adj[i]  = small_counting_adj[i]/small_counting_adj[i][i]
    small_counting_adj = np.where(small_counting_adj > 0.15, 2, 0) # set threshold to 0.2, label = 2.
kg_dict, kg_idx = get_kg()
small_name2index = get_countingAdj_name2index()


full = True
results_list = []
out_ids = []
final_features = []
bboxes = []
normalized_bboxes = []
pos_boxes = []
adj_matrix = []
pred_classes = []
semantic_adj = []
# semantic_adj2 = []
n = 0

length = 5000
l1 = len(f1['image_features'][0])
l2 = len(f2['image_features'][0])
test_topk_per_image = l1 + l2 # need to change if need to change
times = 0
# 1 is the anatomical bbox, 2 is the disease bbox

resume = False # remember to check before running
if resume:
    stopped_batch_num = 135000  # the number you see in the terminal when stooped
    stopped_img_num = stopped_batch_num
    continue_i = stopped_img_num - length
    times = int((stopped_img_num - length)/length)
    n = int(continue_i * test_topk_per_image)
for i in tqdm(range(len(f1['image_adj_matrix']))):
    if resume:
        if i < continue_i:
            continue
    final_features.append(np.vstack((f1['image_features'][i],f2['image_features'][i]))[:test_topk_per_image])
    normalized_bboxes.append(np.vstack((f1['spatial_features'][i],f2['spatial_features'][i]))[:test_topk_per_image])
    bboxes.append(np.vstack((f1['image_bb'][i],f2['image_bb'][i]))[:test_topk_per_image])
    pos_boxes.append((f1['pos_boxes'][i]+f2['pos_boxes'][i])[:test_topk_per_image])
    matrix = f1['image_adj_matrix'][i]
    matrix[l1:l1+l2, l1:l1+l2] = f2['image_adj_matrix'][i][:l2, :l2] # need to change if need to change
    adj_matrix.append(matrix)
    pred_classes.append(cmb_pred_classes(f1['bbox_label'][i], f2['bbox_label'][i]))
    semantic_adj.append(get_semantic_adj(f1['bbox_label'][i], f2['bbox_label'][i], ana_thing_classes, di_thing_classes,kg_dict, small_counting_adj, small_name2index, kg_idx))

    if len(final_features) == length or i == len(f1['image_adj_matrix'])-1:
        final_features = np.array(final_features)
        bboxes = np.array(bboxes)
        normalized_bboxes = np.array(normalized_bboxes)
        pos_boxes = np.array(pos_boxes)
        pred_classes = np.array(pred_classes)
        adj_matrix = np.array(adj_matrix)
        semantic_adj = np.array(semantic_adj)
        # semantic_adj2 = np.array(semantic_adj2)
        adj_matrix = get_adj_matrix(bboxes, adj_matrix) # need to be modified
        save_h5(final_features, normalized_bboxes,bboxes, pos_boxes, adj_matrix,test_topk_per_image,pred_classes, semantic_adj, full=full, times= times, length=length)
        final_features = []
        bboxes = []
        normalized_bboxes = []
        pos_boxes = []
        adj_matrix = []
        pred_classes = []
        semantic_adj = []
        # semantic_adj2 = []
        times += 1


f1.close()
f2.close()
