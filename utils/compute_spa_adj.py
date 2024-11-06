import numpy as np
import math

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

def bbox_relation_type(bb1, bb2, lx=1024, ly=1024, thr=3):
    if bb1[0]< bb2[0] and bb1[1]< bb2[1] and bb1[2]> bb2[2] and bb1[3]>bb2[3]:
        return 1
    elif bb1[0]> bb2[0] and bb1[1]> bb2[1] and bb1[2]< bb2[2] and bb1[3]<bb2[3]:
        return 2
    elif get_iou(bb1, bb2) >= 0.5:
        return 3
    elif get_distance(bb1, bb2) >= (lx+ly)/thr:
        return 0
    angle = cal_angle(bb1,bb2)
    return math.ceil(angle/45) + 3

def get_adj_matrix(bboxes, thr=3):

    num_pics = len(bboxes)
    n = len(bboxes[0])
    # if adj_matrix is None:
    adj_matrix = np.zeros([num_pics,100,100],int)
    for idx in range(num_pics):
        bbs = bboxes[idx]
        for i in range(n):
            for j in range(i,n):
                if adj_matrix[idx,i,j] != 0:
                    continue
                type = bbox_relation_type(bbs[i],bbs[j], thr=thr)
                adj_matrix[idx,i,j] = type
                adj_matrix[idx,j,i] = reverse_type((type))
    return adj_matrix