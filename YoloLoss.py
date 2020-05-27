import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import sys

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max

'''
    pred_mins, pred_maxes: shape(? * S * S * 3 * 2)
    true_mins, true_maxes: shape(? * S * S * 3 * 2)
    
    returns [? * S * S * 3]
'''
def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    # pred와 true 각각 최소값 중 큰값
    intersect_mins = K.maximum(pred_mins, true_mins)  # ? * S * S * 3 * 2

    # pred와 true 각각 최대값 중 작은값
    intersect_maxes = K.minimum(pred_maxes, true_maxes)  # ? * S * S * 3 * 2

    # wh는 max에서 min을 뺀 값(intersact_xmax - intersact_xmin = intersact_w, intersact_ymax - intersact_ymin = intersact_h)
    # wh -> intersact_wh
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)  # ? * S * S * 3 * 2

    # 그 공간에서 w와 h를 곱하면 그 공간의 면적
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  # ? * S * S * 3

    # pred와 true 각각 wh -> area 순으로 구함
    pred_wh = pred_maxes - pred_mins  # ? * S * S * 3 * 2
    true_wh = true_maxes - true_mins  # ? * S * S * 3 * 2
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # ? * S * S * 3
    true_areas = true_wh[..., 0] * true_wh[..., 1]  # ? * S * S * 3

    # A union B = A + B - (A intersect B)
    union_areas = pred_areas + true_areas - intersect_areas  # ? * S * S * 3

    # IOU = Intersact of Union = intersact_area / union_area
    # 교집합이 합집합의 어느정도를 차지하는지를 의미
    # intersact가 전체인경우 -> 1
    # intersact가 전체보다 작은 경우 -> ( < 1 )
    # intersact가 없는 경우 -> 0
    iou_scores = intersect_areas / union_areas  # ? * S * S * 3

    return iou_scores  # ? * S * S * 3


'''
    box_xy, box_wh -> [B * S * S * 3 * 1 * 2]
'''
ANCHOR_BOXES = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHOR_BOXES_TF = tf.reshape(
    tf.constant(ANCHOR_BOXES, dtype=tf.float32),
    shape=(1, 1, 1, 3, 3, 2)
)
SCALES = [13, 26, 52]
# SCALES = [26]

# Returns 1 * S  * S * 1 * 1 * 2
def cell_offset_table(scale_size):
    # Dynamic implementation of conv dims for fully convolutional model.
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=scale_size)
    conv_width_index = K.arange(0, stop=scale_size)
    conv_height_index = K.tile(conv_height_index, [scale_size]) # 늘어놓는 함수  tile -> 같은걸 N번 반복함
    # 결과 -> 0~12, 0~12, ...., 0~12

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [scale_size, 1]) # tile을 [n, m] 쓰면 dims 2로 만들어줌
    # 결과 -> [0~12], [0~12], [0~12], ...

    conv_width_index = K.flatten(K.transpose(conv_width_index))
    # 결과 -> 0, 0, 0, 0, 0, 0, 0 (13개), 1, 1, 1, 1, 1, 1, 1 (13개), ...

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    # 결과 -> [0, 0], [1, 0], [2, 0], ..., [11, 12], [12, 12]

    conv_index = K.reshape(conv_index, [1, scale_size, scale_size, 1, 2])
    # 결과 -> 1 * 13 * 13 에 있는 [1 * 2]의 conv index item이 만들어짐
    # 각각 [1 * 2]의 값은 [0, 0], [1, 0], [2, 0], ..., [11, 12], [12, 12]
    # 이런 식으로 이루어져 있음 -> Mask를 만들기 위한 과정
    # 결과 shape -> 1, 13, 13, 1, 2

    conv_index = K.cast(conv_index, tf.float32)

    diff = (1 / scale_size * 416)
    conv_index = conv_index * diff

    return conv_index

def generate_index_matrix(scale_size):
    scale_range = tf.range(scale_size, dtype=tf.float32)

    x_table = tf.reshape(tf.tile(scale_range, [scale_size]), (scale_size, scale_size))
    y_table = tf.transpose(x_table)

    index_matrix = tf.stack([x_table, y_table], axis=2)  # 그냥 사용할때는 axis=1, reshape시 axis=2
    return tf.reshape(index_matrix, (1, scale_size, scale_size, 1, 2))  # B * S * S * A * xy(2)


def get_anchor_box(scale_index):
    return ANCHOR_BOXES_TF[:, :, :, scale_index, :, :]

def CreateYolov3Loss(scale_size, scale_index, verbose=False):

    '''
    scale_size = 13 -> 32
    scale_size = 26 -> 16
    scale_size = 52 -> 8
    '''
    # cell_size = 32
    cell_size = 416 / scale_size

    # loss multiply
    scale_index_divider = scale_index + 1
    # volume_loss_multiply = 1. / (scale_index_divider * scale_index_divider)  # /1, /4, /9
    volume_loss_multiply = 1.

    image_size = 416

    # Anchor box
    anchor_box = get_anchor_box(scale_index)

    def __loss_dev_v3__(y_true, y_pred):
        '''
        YOLOv3 Loss Function
        총 3개의 Part로 구성된, Multi-Part Loss이다.
        1. xywh Loss (Box Loss): 해당 Anchor Cell이 책임지는 박스의 Scale을 예상한 값이다.
        2. Confidence Loss: 박스가 해당 셀에 실재하는 정도를 예상한 값이다.
        3. Class Loss: 각 클래스(VOC 총 20개) Probability는 실제 해당 셀의
                       Class Probability와 동일해야 한다.
        :param y_true: [B * S * S * 3 * 25]
        :param y_pred: [B * S * S * 3 * 25]
        :return: 단일 Floating-Point Value. 이 값으로 전체 Weight를 업데이트한다.
        '''
        # if verbose: tf.print("Anchor box of anchor idx " + str(scale_index) + ": ", anchor_box)
        pred_class = y_pred[..., :20]
        pred_bbox = y_pred[..., 20:24]
        pred_conf = y_pred[..., 24:25]

        label_class = y_true[..., :20]
        label_bbox = y_true[..., 20:24]
        label_conf = y_true[..., 24:25]

        # Create IOU between two bboxes
        # Convert cell-relative xy to global-relative xy (glr)
        index_matrix = generate_index_matrix(scale_size)

        pred_bbox_xy_glr = (tf.sigmoid(pred_bbox[..., :2]) + index_matrix) / scale_size  # normalized from 0.0 to 1.0
        pred_bbox_wh_glr = (anchor_box * tf.exp(pred_bbox[..., 2:])) / image_size
        #! Sigmoid of t_o goes to YoloModel

        label_bbox_xy_glr = (label_bbox[..., :2] + index_matrix) / scale_size
        label_bbox_wh = label_bbox[..., 2:]

        # if verbose: tf.print("pred_bbox_xy_glr[0, 0]:", pred_bbox_xy_glr[0, 0, 0, 0, :] * scale_size)
        # if verbose: tf.print("label_bbox_xy_glr[12, 12]:", label_bbox_xy_glr[0, 12, 12, 2, :] * scale_size)
        pred_bbox_xy_min = pred_bbox_xy_glr - (pred_bbox_wh_glr / 2)
        pred_bbox_xy_max = pred_bbox_xy_glr + (pred_bbox_wh_glr / 2)
        pred_volume = pred_bbox_wh_glr[..., 0] * pred_bbox_wh_glr[..., 1]

        label_bbox_xy_min = label_bbox_xy_glr - (label_bbox_wh / 2)
        label_bbox_xy_max = label_bbox_xy_glr + (label_bbox_wh / 2)
        label_volume = label_bbox_wh[..., 0] * label_bbox_wh[..., 1]

        intersact_min = tf.maximum(pred_bbox_xy_min, label_bbox_xy_min)
        intersact_max = tf.minimum(pred_bbox_xy_max, label_bbox_xy_max)
        intersact_wh = tf.maximum(intersact_max - intersact_min, 0)  # clip to zero
        intersact_volume = intersact_wh[..., 0] * intersact_wh[..., 1]

        union_volume = pred_volume + label_volume - intersact_volume
        iou_volume = intersact_volume / union_volume
        max_iou_anchor_idx = tf.argmax(iou_volume, axis=3)
        max_iou_anchor_mask = tf.one_hot(max_iou_anchor_idx, depth=3, dtype=tf.float32)
        max_iou_anchor_mask = tf.expand_dims(max_iou_anchor_mask, axis=4)  # B * S * S * 3 * 1

        # if verbose: tf.print("max_iou_anchor_idx:", tf.shape(max_iou_anchor_idx))
        # if verbose: tf.print("max_iou_anchor_mask:", tf.shape(max_iou_anchor_mask))
        # if verbose: tf.print("iou_volume:", tf.shape(iou_volume))
        # if verbose: tf.print("max_iou_anchor_value:", tf.shape(max_iou_anchor_value))

        # pred_object_mask → 1(obj)
        label_object_mask = label_conf * max_iou_anchor_mask
        # pred_no_object_mask → 1(noobj)
        label_no_object_mask = 1 - label_object_mask

        # gt_object_mask = pred_conf
        # gt_no_object_mask = 1 - pred_conf

        # if verbose: tf.print("label_object_mask:", tf.shape(label_object_mask))
        # if verbose: tf.print("label_no_object_mask:", tf.shape(label_no_object_mask))
        # if verbose: tf.print("gt_object_mask:", tf.shape(gt_object_mask))
        # if verbose: tf.print("gt_no_object_mask:", tf.shape(gt_no_object_mask))

        '''
        Constants
        '''
        lambda_coord = 5
        lambda_noobj = 0.5

        '''
        Box Loss
        '''
        # if verbose: tf.print("shape of label_object_mask:", tf.shape(label_object_mask))
        # if verbose: tf.print("s hape of pred_xy:", tf.shape(pred_xy))
        # if verbose: tf.print("shape of label_xy:", tf.shape(label_xy))
        # if verbose: tf.print("shape of pred_wh:", tf.shape(pred_wh))
        # if verbose: tf.print("shape of label_wh:", tf.shape(label_wh))
        # box_scale = label_bbox[..., 2:3] * label_bbox[..., 3:4]

        xy_loss = lambda_coord * label_object_mask * tf.square(pred_bbox_xy_glr - label_bbox_xy_glr)
        wh_loss = lambda_coord * label_object_mask * tf.square(tf.sqrt(pred_bbox_wh_glr) - tf.sqrt(label_bbox_wh))

        # if verbose: tf.print("tf.sqrt(pred_wh):", tf.shape(tf.sqrt(pred_wh)))
        # if verbose: tf.print("tf.sqrt(label_wh):", tf.shape(tf.sqrt(label_wh)))
        # if verbose: tf.print("xy_loss:", tf.reduce_sum(xy_loss))
        # if verbose: tf.print("wh_loss:", tf.reduce_sum(wh_loss))

        box_loss = (tf.reduce_sum(xy_loss) + tf.reduce_sum(wh_loss)) * volume_loss_multiply
        if verbose: tf.print("[%d] box_loss:" % scale_size, box_loss)

        '''
        Confidence Loss
        '''
        pred_cell_class_prob = pred_conf * pred_class
        label_cell_class_prob = label_conf * label_class

        # if verbose: tf.print("pred_cell_class_prob:", tf.shape(pred_cell_class_prob))
        # if verbose: tf.print("label_cell_class_prob:", tf.shape(label_cell_class_prob))

        sqrt_cell_class_prob = tf.square(pred_cell_class_prob - label_cell_class_prob)

        object_loss = label_object_mask * sqrt_cell_class_prob
        noobj_loss = lambda_noobj * label_no_object_mask * sqrt_cell_class_prob
        if verbose: tf.print("object_loss:", tf.reduce_sum(object_loss))
        if verbose: tf.print("noobj_loss:", tf.reduce_sum(noobj_loss))

        confidence_loss = (tf.reduce_sum(object_loss) + tf.reduce_sum(noobj_loss)) * volume_loss_multiply
        if verbose: tf.print("[%d] confidence_loss:" % scale_size, confidence_loss)

        '''
        Class Loss
        [X] 클래스 확률만 봤을때 이상무
        [X] 클래스 확률 * confidence 곱해서 볼 때 이상무
          → confidence가 어떤 값이든 클래스 확률이 제일 큰 값(chair)을 가져오므로 이상 무
       '''
        class_loss = label_object_mask * tf.square(pred_class - label_class)
        class_loss = tf.reduce_sum(class_loss) * volume_loss_multiply
        if verbose: tf.print("[%d] class_loss:" % scale_size, class_loss, "\n")

        return box_loss + confidence_loss + class_loss


    # def __dummy__(y_true, y_pred):
    #     if verbose: tf.print("y_true shape: ", tf.shape(y_true))
    #     if verbose: tf.print("y_pred shape: ", tf.shape(y_pred))
    #
    #     return tf.reduce_mean(y_pred)

    return __loss_dev_v3__
    # return __dummy__

